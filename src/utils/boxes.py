from typing import NamedTuple

import torch
from torch import Tensor
from torchvision.ops import batched_nms, box_area, box_convert

from core.config import P, Parameters


# Lo que devuelve cualquier detector del proyecto, ya decodificado.
class Detections(NamedTuple):
    boxes: Tensor  # (K, 4) cxcywh normalizado
    scores: Tensor  # (K,)
    labels: Tensor  # (K,) id de clase en `data.species.LabelSet`


def _min_area_iou(boxes_xyxy: Tensor) -> Tensor:
    # Intersección sobre el área de la caja más chica: vale 1 si una está contenida en otra,
    # donde el IoU clásico baja con la diferencia de tamaño y deja pasar el duplicado.
    top_left = torch.max(boxes_xyxy[:, None, :2], boxes_xyxy[None, :, :2])
    bottom_right = torch.min(boxes_xyxy[:, None, 2:], boxes_xyxy[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    area = box_area(boxes_xyxy)
    return intersection / torch.min(area[:, None], area[None]).clamp(min=1e-6)


def suppress_nested(
    boxes_xyxy: Tensor,
    scores: Tensor,
    labels: Tensor,
    nms_iou: float,
    iomin_threshold: float = 0.8,
) -> Tensor:
    # El NMS saca los solapes parciales; el IoMin, la caja anidada en otra de más score.
    keep = batched_nms(boxes_xyxy, scores, labels, nms_iou)
    kept_labels = labels[keep]
    overlap = _min_area_iou(boxes_xyxy[keep]).triu(diagonal=1)  # `keep` viene por score
    same_class = kept_labels[:, None] == kept_labels[None, :]
    nested = ((overlap > iomin_threshold) & same_class).any(dim=0)
    return keep[~nested]


def to_pixel_xyxy(boxes: Tensor, params: Parameters = P) -> Tensor:
    scale = boxes.new_tensor([params.n_frames, params.n_mels] * 2)
    return box_convert(boxes, "cxcywh", "xyxy") * scale


def to_unit_cxcywh(boxes_xyxy: Tensor, params: Parameters = P) -> Tensor:
    scale = boxes_xyxy.new_tensor([params.n_frames, params.n_mels] * 2)
    return box_convert(boxes_xyxy / scale, "xyxy", "cxcywh")
