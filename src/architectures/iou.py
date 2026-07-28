"""Criterios de solape entre cajas, en formato `xyxy` y con firma `(N, 4), (M, 4) -> (N, M)`.

Todos devuelven una similitud (1.0 = solape perfecto), así que la pérdida es
`1 - sim` y el coste del matcher `-sim`.

- `iou`, `giou`: `torchvision.ops` (Rezatofighi et al., 2019).
- `iomin`: intersección sobre el área menor (Hexeberg et al., 2021).
- `eiou`: Efficient IoU (Zhang et al., 2022), https://arxiv.org/abs/2101.08158
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torchvision.ops import box_area, box_iou, generalized_box_iou

EPS = 1e-6


def min_area_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)

    return intersection / torch.min(area1[:, None], area2[None]).clamp(min=EPS)


def efficient_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing = (bottom_right - top_left).clamp(min=0)
    enclosing_width = enclosing[..., 0]
    enclosing_height = enclosing[..., 1]
    enclosing_diagonal = enclosing_width.pow(2) + enclosing_height.pow(2)

    centers1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
    centers2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
    center_distance = (centers1[:, None] - centers2[None]).pow(2).sum(-1)

    sizes1 = boxes1[:, 2:] - boxes1[:, :2]
    sizes2 = boxes2[:, 2:] - boxes2[:, :2]
    width_distance = (sizes1[:, None, 0] - sizes2[None, :, 0]).pow(2)
    height_distance = (sizes1[:, None, 1] - sizes2[None, :, 1]).pow(2)

    return (
        iou
        - center_distance / enclosing_diagonal.clamp(min=EPS)
        - width_distance / enclosing_width.pow(2).clamp(min=EPS)
        - height_distance / enclosing_height.pow(2).clamp(min=EPS)
    )


IOU_FUNCTIONS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "iou": box_iou,
    "giou": generalized_box_iou,
    "iomin": min_area_box_iou,
    "eiou": efficient_box_iou,
}


def get_iou_fn(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    if name not in IOU_FUNCTIONS:
        raise ValueError(
            f"criterio de solape desconocido: {name!r}; opciones: {sorted(IOU_FUNCTIONS)}"
        )
    return IOU_FUNCTIONS[name]
