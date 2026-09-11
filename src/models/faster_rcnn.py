import torch
from torch import Tensor
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

from core.config import NMS_IOU, SCORE_FLOOR, SCORE_THRESHOLD
from models.base import Detector
from utils.audio import mel_to_unit
from utils.boxes import Detections, Target, to_pixel_xyxy, to_unit_cxcywh

# Alto/ancho de las cajas: va de 0.05 a 9
ANCHOR_RATIOS: tuple[float, ...] = (0.05, 0.15, 0.5, 1.5, 5.0)
ANCHOR_SIZES: tuple[tuple[int], ...] = ((32,), (64,), (128,), (256,), (512,))
# El mel de 128 x 331 queda en 396 x 1024
MIN_SIZE, MAX_SIZE = 512, 1024
# Capas del ResNet que se afinan, de 5
TRAINABLE_LAYERS = 3


class SpectrogramFasterRCNN(Detector):
    # Cabeza densa: trae duplicados
    nms_iou = NMS_IOU
    clip_grad = 10.0
    needs_db_range = True

    def __init__(
        self,
        n_classes: int,
        db_low: float,
        db_high: float,
        pretrained: bool = True,
        min_size: int = MIN_SIZE,
        max_size: int = MAX_SIZE,
        anchor_ratios: tuple[float, ...] = ANCHOR_RATIOS,
        trainable_layers: int = TRAINABLE_LAYERS,
        # torchvision descarta por debajo de esto (de fábrica 0.05): el piso del protocolo
        score_thresh: float = SCORE_FLOOR,
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
        predictor = model.roi_heads.box_predictor
        assert isinstance(predictor, FastRCNNPredictor)
        model.roi_heads.box_predictor = FastRCNNPredictor(
            predictor.cls_score.in_features, n_classes + 1
        )
        anchors = AnchorGenerator(ANCHOR_SIZES, (tuple(anchor_ratios),) * len(ANCHOR_SIZES))
        model.rpn.anchor_generator = anchors
        out_channels = model.backbone.out_channels
        assert isinstance(out_channels, int)
        model.rpn.head = RPNHead(out_channels, anchors.num_anchors_per_location()[0], conv_depth=2)
        self.model = model

    def to_images(self, mel: Tensor) -> list[Tensor]:
        return list(mel_to_unit(mel, self.db_low, self.db_high).expand(-1, 3, -1, -1))

    @staticmethod
    def to_torchvision(targets: list[Target]) -> list[Target]:
        # torchvision quiere píxeles xyxy y reserva la clase 0 para el fondo.
        return [
            {"boxes": to_pixel_xyxy(t["boxes"]), "labels": t["labels"].to(torch.int64) + 1}
            for t in targets
        ]

    def forward(
        self, mel: Tensor, targets: list[Target] | None = None
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        images = self.to_images(mel)
        if targets is None:
            return self.model(images)
        targets = self.to_torchvision(targets)
        if self.training:
            return self.model(images, targets)
        # torchvision sólo devuelve pérdidas en modo train: se fuerza en las cabezas, no en las BN.
        for module in (self.model, self.model.rpn, self.model.roi_heads):
            module.training = True
        try:
            return self.model(images, targets)
        finally:
            self.model.eval()

    @torch.no_grad()
    def detect(self, mel: Tensor, score_threshold: float = SCORE_THRESHOLD) -> list[Detections]:
        detections = []
        for output in self(mel):
            above = output["scores"] >= score_threshold
            detections.append(
                Detections(
                    boxes=to_unit_cxcywh(output["boxes"][above]),
                    scores=output["scores"][above],
                    labels=output["labels"][above] - 1,
                )
            )
        return detections
