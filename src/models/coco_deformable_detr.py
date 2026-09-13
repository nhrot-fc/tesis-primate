import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.init import constant_

from core.config import HF_DIR, SCORE_THRESHOLD
from models.base import Detector
from models.criterion import Outputs, SetCriterion
from models.deformable_detr import PRIOR_PROB, inverse_sigmoid, sigmoid_detections
from models.faster_rcnn import MAX_SIZE, MIN_SIZE
from utils.audio import mel_to_unit
from utils.boxes import Detections, Target

if TYPE_CHECKING:
    from transformers import DeformableDetrForObjectDetection

logger = logging.getLogger(__name__)

# Deformable DETR de dos etapas con refinamiento de cajas, entrenado en COCO (Zhu et al. 2021)
DETR_CHECKPOINT = "SenseTime/deformable-detr-with-box-refine-two-stage"
# Queries y propuestas con las que se entrenó en COCO
COCO_QUERIES = 300
# Con lo que se normalizaron las imágenes que vio su ResNet-50
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def local_detr_dir(checkpoint: str = DETR_CHECKPOINT) -> Path:
    return HF_DIR / checkpoint.replace("/", "__")


def load_detr(
    n_classes: int, n_queries: int, checkpoint: str = DETR_CHECKPOINT
) -> "DeformableDetrForObjectDetection":
    # `transformers` y `timm` (su ResNet-50) son el extra `detr`: sólo hacen falta si se carga
    # esta arquitectura.
    from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection

    local_dir = local_detr_dir(checkpoint)
    if not local_dir.is_dir():
        logger.info("Descargando Deformable DETR '%s' desde HuggingFace...", checkpoint)
        DeformableDetrForObjectDetection.from_pretrained(checkpoint).save_pretrained(local_dir)
        logger.info("Deformable DETR guardado en %s", local_dir)

    config = DeformableDetrConfig.from_pretrained(local_dir)
    if not config.two_stage:
        raise ValueError(f"{checkpoint} no es de dos etapas; `outputs` cuenta con las propuestas")
    config.num_labels = n_classes
    config.num_queries = config.two_stage_num_proposals = n_queries
    # Las cabezas de clase (91 clases de COCO) se reinician; el resto carga tal cual.
    return DeformableDetrForObjectDetection.from_pretrained(
        local_dir, config=config, ignore_mismatched_sizes=True
    )


class ResNetDeformableDETR(Detector):
    # El Deformable DETR de COCO tal cual sale de HF, ResNet-50 incluido, con la receta de Zhu
    # et al. 2021: el mel pintado como imagen ImageNet, 300 queries, stem y layer1 congelados.
    # Sin NMS: el matching húngaro ya es uno a uno
    nms_iou = None
    clip_grad = 0.1
    needs_db_range = True
    mean: Tensor
    std: Tensor

    def __init__(
        self,
        n_classes: int,
        db_low: float,
        db_high: float,
        n_queries: int = COCO_QUERIES,
        min_size: int = MIN_SIZE,
        max_size: int = MAX_SIZE,
        checkpoint: str = DETR_CHECKPOINT,
    ):
        super().__init__()
        self.db_low, self.db_high = db_low, db_high
        self.min_size, self.max_size = min_size, max_size
        self.detr = load_detr(n_classes, n_queries, checkpoint)
        for name, parameter in self.detr.model.backbone.model.named_parameters():
            parameter.requires_grad_(any(f"layer{i}" in name for i in (2, 3, 4)))
        # Cabezas nuevas con el prior de Zhu et al. 2021; la última puntúa las propuestas del encoder.
        for head in self.detr.class_embed:
            assert isinstance(head, nn.Linear)
            constant_(head.bias, -math.log((1 - PRIOR_PROB) / PRIOR_PROB))
        self.criterion = SetCriterion()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def pixels(self, mel: Tensor) -> Tensor:
        # Como el Faster R-CNN: el mel en dB a [0, 1], escalado a 396 x 1024 y en tres canales
        image = mel_to_unit(mel, self.db_low, self.db_high)
        height, width = image.shape[-2:]
        scale = min(self.min_size / min(height, width), self.max_size / max(height, width))
        size = (round(height * scale), round(width * scale))
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        return (image.expand(-1, 3, -1, -1) - self.mean) / self.std

    def outputs(self, mel: Tensor) -> Outputs:
        core = self.detr.model(pixel_values=self.pixels(mel))
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
