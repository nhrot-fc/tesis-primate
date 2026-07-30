from collections.abc import Callable

import torch
from torch import Tensor
from torchvision.ops import box_area, box_iou

EPS = 1e-6


def min_area_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)

    return intersection / torch.min(area1[:, None], area2[None]).clamp(min=EPS)


def box_iou_pairwise(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """`box_iou` para pares ya emparejados: `boxes1[i]` contra `boxes2[i]` -> `(N,)`."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.max(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=EPS)


def min_area_box_iou_pairwise(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.max(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    return intersection / torch.min(area1, area2).clamp(min=EPS)


IOU_FUNCTIONS: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "iou": box_iou,
    "iomin": min_area_box_iou,
}

IOU_FUNCTIONS_PAIRWISE: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "iou": box_iou_pairwise,
    "iomin": min_area_box_iou_pairwise,
}


def get_iou_fn(name: str, pairwise: bool = False) -> Callable[[Tensor, Tensor], Tensor]:
    functions = IOU_FUNCTIONS_PAIRWISE if pairwise else IOU_FUNCTIONS
    if name not in functions:
        raise ValueError(f"criterio de solape desconocido: {name!r}; opciones: {sorted(functions)}")
    return functions[name]
