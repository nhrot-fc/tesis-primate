"""Faster R-CNN de torchvision sobre el espectrograma mel.

El caché guarda el mel en potencia y de un canal; el modelo espera 3 canales en [0, 1],
así que la conversión a dB con rango global va adentro del `forward`. El eje de
frecuencia no se invierte: la fila 0 es la banda más grave, la misma `y` normalizada que
usan `utils.audio.hz_to_y` y el Deformable-DETR.
"""

import torch
from torch import Tensor, nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.ops import box_convert

from architectures.deformable_detr import Detections
from core.config import P, Parameters
from utils.audio import mel_to_unit

# El alto/ancho real de las cajas va de 0.05 a 9 (p1, p95=6.2): con los (0.5, 1, 2)
# de fábrica el RPN se pierde los tonos angostos y las bandas largas.
ANCHOR_RATIOS: tuple[float, ...] = (0.05, 0.15, 0.5, 1.5, 5.0)
ANCHOR_SIZES: tuple[tuple[int], ...] = ((32,), (64,), (128,), (256,), (512,))
# Sobre el mel de 128 x 331 esto deja la imagen en 396 x 1024, conservando la relación
# de aspecto y con ella la validez de `ANCHOR_RATIOS`.
MIN_SIZE, MAX_SIZE = 512, 1024
# Torchvision descarta adentro todo lo que baje de 0.05, y con eso la cola de la curva
# de AP queda cortada mientras el DETR y YOLO llegan hasta 0.001. Se baja para que los
# tres reporten el mismo rango de scores.
SCORE_THRESH = 0.001


class SpectrogramFasterRCNN(nn.Module):
    def __init__(
        self,
        n_classes: int,
        db_low: float,
        db_high: float,
        pretrained: bool = True,
        min_size: int = MIN_SIZE,
        max_size: int = MAX_SIZE,
        anchor_ratios: tuple[float, ...] = ANCHOR_RATIOS,
        trainable_layers: int = 3,
        score_thresh: float = SCORE_THRESH,
    ) -> None:
        super().__init__()
        self.db_low, self.db_high = db_low, db_high

        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
            trainable_backbone_layers=trainable_layers if pretrained else None,
            min_size=min_size,
            max_size=max_size,
            box_score_thresh=score_thresh,
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes + 1)

        # Otros anchors son otra cantidad por posición: la cabeza del RPN se reconstruye
        # y pierde COCO. El backbone y la cabeza de cajas, el grueso, se conservan.
        anchors = AnchorGenerator(ANCHOR_SIZES, (tuple(anchor_ratios),) * len(ANCHOR_SIZES))
        model.rpn.anchor_generator = anchors
        model.rpn.head = RPNHead(
            model.backbone.out_channels, anchors.num_anchors_per_location()[0], conv_depth=2
        )
        self.model = model

    def to_images(self, mel: Tensor) -> list[Tensor]:
        unit = mel_to_unit(mel, self.db_low, self.db_high)
        return list(unit.expand(-1, 3, -1, -1))

    def forward(
        self, mel: Tensor, targets: list[dict[str, Tensor]] | None = None
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """En train devuelve el diccionario de pérdidas; en eval, las detecciones."""
        return self.model(self.to_images(mel), targets)


def to_torchvision_targets(
    targets: list[dict[str, Tensor]], params: Parameters = P
) -> list[dict[str, Tensor]]:
    """Cajas cxcywh normalizadas -> xyxy en píxeles, con la clase 0 reservada al fondo."""
    scale = torch.tensor(
        [params.n_frames, params.n_mels, params.n_frames, params.n_mels], dtype=torch.float32
    )
    return [
        {
            "boxes": box_convert(target["boxes"], "cxcywh", "xyxy")
            * scale.to(target["boxes"].device),
            "labels": target["labels"].to(torch.int64) + 1,
        }
        for target in targets
    ]


def postprocess(
    outputs: list[dict[str, Tensor]], score_threshold: float = 0.5, params: Parameters = P
) -> list[Detections]:
    scale = torch.tensor(
        [params.n_frames, params.n_mels, params.n_frames, params.n_mels], dtype=torch.float32
    )

    detections = []
    for output in outputs:
        keep = output["scores"] >= score_threshold
        boxes = output["boxes"][keep] / scale.to(output["boxes"].device)
        detections.append(
            Detections(
                boxes=box_convert(boxes, "xyxy", "cxcywh"),
                scores=output["scores"][keep],
                labels=output["labels"][keep] - 1,
            )
        )
    return detections


@torch.no_grad()
def detect(
    model: SpectrogramFasterRCNN, images: Tensor, score_threshold: float = 0.5
) -> list[Detections]:
    training = model.training
    model.eval()
    outputs = model(images)
    model.train(training)
    return postprocess(outputs, score_threshold)
