import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn.init import constant_

from core.config import HF_DIR, SCORE_THRESHOLD
from models.backbone import TIME_STRIDE, ASTBackbone, MultiScalePyramid
from models.base import Detector
from models.criterion import Outputs, SetCriterion
from models.deformable_detr import (
    FRONTEND,
    N_QUERIES,
    PRIOR_PROB,
    inverse_sigmoid,
    sigmoid_detections,
)
from models.frontend import build_frontend
from utils.boxes import Detections, Target

if TYPE_CHECKING:
    from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

logger = logging.getLogger(__name__)

# Deformable DETR de dos etapas con refinamiento de cajas, entrenado en COCO (Zhu et al. 2021)
DETR_CHECKPOINT = "SenseTime/deformable-detr-with-box-refine-two-stage"
D_MODEL = 256


def local_detr_dir(checkpoint: str = DETR_CHECKPOINT) -> Path:
    return HF_DIR / checkpoint.replace("/", "__")


def placeholder_config(source: str | Path) -> "DeformableDetrConfig":
    from transformers import DeformableDetrConfig, ResNetConfig

    config = DeformableDetrConfig.from_pretrained(source)
    if not config.two_stage:
        raise ValueError(f"{source} no es de dos etapas; `outputs` cuenta con las propuestas")
    # El ResNet-50 de COCO se descarta (y pediría `timm`): un ResNet de relleno con cuatro salidas
    # de D_MODEL canales hace que HF arme `input_proj` como cuatro convoluciones 1x1 para la
    # pirámide del AST, que ocupa su lugar en `CocoDeformableDETR`.
    config.backbone_config = ResNetConfig(
        embedding_size=D_MODEL,
        hidden_sizes=[D_MODEL] * 4,
        depths=[1] * 4,
        out_features=[f"stage{i}" for i in range(1, 5)],
    )
    return config


def load_detr(
    n_classes: int, n_queries: int, checkpoint: str = DETR_CHECKPOINT
) -> "DeformableDetrForObjectDetection":
    # `transformers` es el extra `detr`: sólo hace falta si se carga esta arquitectura.
    from transformers import DeformableDetrForObjectDetection

    local_dir = local_detr_dir(checkpoint)
    if not local_dir.is_dir():
        logger.info("Descargando Deformable DETR '%s' desde HuggingFace...", checkpoint)
        pristine = DeformableDetrForObjectDetection.from_pretrained(
            checkpoint, config=placeholder_config(checkpoint), ignore_mismatched_sizes=True
        )
        pristine.save_pretrained(local_dir)
        logger.info("Deformable DETR guardado en %s", local_dir)

    config = placeholder_config(local_dir)
    config.num_labels = n_classes
    config.num_queries = config.two_stage_num_proposals = n_queries
    # Las cabezas de clase (91 clases de COCO) se reinician; el resto carga tal cual.
    return DeformableDetrForObjectDetection.from_pretrained(
        local_dir, config=config, ignore_mismatched_sizes=True
    )


class ASTPyramid(nn.Module):
    # Ocupa el lugar del ResNet de HF: (mapa, máscara) por nivel, del 4x al 1/2x de los tokens del AST.
    def __init__(self, n_frames: int | None, time_stride: int, frontend: str, dim: int = D_MODEL):
        super().__init__()
        self.backbone = ASTBackbone(n_frames=n_frames, time_stride=time_stride)
        self.frontend = build_frontend(frontend, self.backbone.n_mels)
        # Media 0 y varianza 1 por lote, a la mitad: el rango con el que el AST se preentrenó.
        self.input_norm = nn.BatchNorm2d(1, affine=False)
        self.proj = nn.Linear(self.backbone.hidden_size, dim)
        self.pyramid = MultiScalePyramid(dim)
        self.pyramid.check_input_size(self.backbone.freq_out, self.backbone.time_out)

    def forward(self, mel: Tensor, pixel_mask: Tensor | None = None) -> list[tuple[Tensor, Tensor]]:
        tokens = self.backbone(self.input_norm(self.frontend(mel)) / 2)
        features = self.proj(tokens).transpose(1, 2)
        features = features.unflatten(-1, (self.backbone.freq_out, self.backbone.time_out))
        return [
            (level, level.new_ones(level.shape[0], *level.shape[-2:], dtype=torch.bool))
            for level in self.pyramid(features)
        ]


class CocoDeformableDETR(Detector):
    # Sin NMS: el matching húngaro ya es uno a uno
    nms_iou = None
    clip_grad = 0.1

    def __init__(
        self,
        n_classes: int,
        n_frames: int | None = None,
        time_stride: int = TIME_STRIDE,
        n_queries: int = N_QUERIES,
        frontend: str = FRONTEND,
        checkpoint: str = DETR_CHECKPOINT,
    ):
        super().__init__()
        self.detr = load_detr(n_classes, n_queries, checkpoint)
        self.detr.model.backbone = ASTPyramid(n_frames, time_stride, frontend)
        # Cabezas nuevas con el prior de Zhu et al. 2021; la última puntúa las propuestas del encoder.
        for head in self.detr.class_embed:
            constant_(head.bias, -math.log((1 - PRIOR_PROB) / PRIOR_PROB))
        self.criterion = SetCriterion()

    def outputs(self, mel: Tensor) -> Outputs:
        core = self.detr.model(pixel_values=mel)
        hidden = core.intermediate_hidden_states  # (B, capas, Q, C)
        per_layer: list[dict[str, Tensor]] = []
        for layer in range(hidden.shape[1]):
            # Cada capa refina la caja con la que entró, en espacio logit.
            reference = (
                core.init_reference_points
                if layer == 0
                else core.intermediate_reference_points[:, layer - 1]
            )
            delta = self.detr.bbox_embed[layer](hidden[:, layer])
            per_layer.append(
                {
                    "pred_logits": self.detr.class_embed[layer](hidden[:, layer]),
                    "pred_boxes": (delta + inverse_sigmoid(reference)).sigmoid(),
                }
            )
        outputs: Outputs = dict(per_layer[-1])
        outputs["aux_outputs"] = per_layer[:-1]
        # Propuestas del encoder (segunda etapa): una por posición de la pirámide
        outputs["enc_outputs"] = {
            "pred_logits": core.enc_outputs_class,
            "pred_boxes": core.enc_outputs_coord_logits.sigmoid(),
        }
        return outputs

    def forward(
        self, mel: Tensor, targets: list[Target] | None = None
    ) -> Outputs | dict[str, Tensor]:
        outputs = self.outputs(mel)
        if targets is None:
            return outputs
        losses = self.criterion(outputs, targets)
        # Las propuestas se entrenan con una sola etiqueta, "hay algo" (Zhu et al. 2021, A.4).
        binary = [{**target, "labels": torch.zeros_like(target["labels"])} for target in targets]
        for key, value in self.criterion.losses(outputs["enc_outputs"], binary).items():
            losses[key] = losses[key] + value
        return losses

    @torch.no_grad()
    def detect(self, mel: Tensor, score_threshold: float = SCORE_THRESHOLD) -> list[Detections]:
        return sigmoid_detections(self(mel), score_threshold)
