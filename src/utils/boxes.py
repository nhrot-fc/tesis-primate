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


def suppress_nested(
    boxes_xyxy: Tensor,
    scores: Tensor,
    labels: Tensor,
    nms_iou: float,
    iomin_threshold: float = 0.8,
) -> Tensor:
    keep = batched_nms(boxes_xyxy, scores, labels, nms_iou)  # saca los solapes parciales
    kept, kept_labels = boxes_xyxy[keep], labels[keep]

    # Intersección sobre el área de la caja más chica: vale 1 si una está contenida en la
    # otra, donde el IoU clásico baja con la diferencia de tamaño y deja pasar el duplicado.
    top_left = torch.max(kept[:, None, :2], kept[None, :, :2])
    bottom_right = torch.min(kept[:, None, 2:], kept[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    area = box_area(kept)
    iomin = intersection / torch.min(area[:, None], area[None]).clamp(min=1e-6)

    # `keep` viene ordenado por score, así que la triangular superior deja sólo las cajas
    # anidadas en otra mejor.
    same_class = kept_labels[:, None] == kept_labels[None, :]
    nested = ((iomin.triu(diagonal=1) > iomin_threshold) & same_class).any(dim=0)
    return keep[~nested]


def to_pixel_xyxy(boxes: Tensor, params: Parameters = P) -> Tensor:
    scale = boxes.new_tensor([params.n_frames, params.n_mels] * 2)
    return box_convert(boxes, "cxcywh", "xyxy") * scale


def to_unit_cxcywh(boxes_xyxy: Tensor, params: Parameters = P) -> Tensor:
    scale = boxes_xyxy.new_tensor([params.n_frames, params.n_mels] * 2)
    return box_convert(boxes_xyxy / scale, "xyxy", "cxcywh")
