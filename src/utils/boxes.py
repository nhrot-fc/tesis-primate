from typing import NamedTuple

import torch
from torch import Tensor
from torchvision.ops import batched_nms, box_area, box_convert

from core.config import P

# 'boxes' (N,4) cxcywh en [0,1] + 'labels' (N,)
Target = dict[str, Tensor]
IOMIN_THRESHOLD = 0.8  # intersección sobre el área de la caja chica


class Detections(NamedTuple):
    boxes: Tensor  # (K, 4) cxcywh en [0,1]
    scores: Tensor  # (K,)
    labels: Tensor  # (K,) id de clase en `LabelSet`

    def select(self, index: Tensor) -> "Detections":
        return Detections(*(field[index] for field in self))


def suppress_nested(boxes_xyxy: Tensor, scores: Tensor, labels: Tensor, nms_iou: float) -> Tensor:
    keep = batched_nms(boxes_xyxy, scores, labels, nms_iou)
    kept, kept_labels = boxes_xyxy[keep], labels[keep]

    # Intersección sobre la caja chica: vale 1 si una contiene a la otra, cosa que el IoU no ve.
    top_left = torch.max(kept[:, None, :2], kept[None, :, :2])
    bottom_right = torch.min(kept[:, None, 2:], kept[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    area = box_area(kept)
    overlap = intersection / torch.min(area[:, None], area[None]).clamp(min=1e-6)

    # `keep` viene por score: la triangular superior deja sólo las anidadas en otra mejor.
    same_class = kept_labels[:, None] == kept_labels[None, :]
    nested = ((overlap.triu(diagonal=1) > IOMIN_THRESHOLD) & same_class).any(dim=0)
    return keep[~nested]


def postprocess(
    detections: Detections, nms_iou: float | None, max_detections: int | None
) -> Detections:
    boxes, scores, labels = detections
    if nms_iou is not None and len(boxes):
        keep = suppress_nested(box_convert(boxes, "cxcywh", "xyxy"), scores, labels, nms_iou)
        detections = detections.select(keep)
    if max_detections is not None and len(detections.boxes) > max_detections:
        detections = detections.select(detections.scores.topk(max_detections).indices)
    return detections


def to_pixel_xyxy(boxes: Tensor) -> Tensor:
    scale = boxes.new_tensor([P.n_frames, P.n_mels] * 2)
    return box_convert(boxes, "cxcywh", "xyxy") * scale


def to_unit_cxcywh(boxes_xyxy: Tensor) -> Tensor:
    scale = boxes_xyxy.new_tensor([P.n_frames, P.n_mels] * 2)
    return box_convert(boxes_xyxy / scale, "xyxy", "cxcywh")
