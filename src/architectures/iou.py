import torch
from torch import Tensor
from torchvision.ops import batched_nms, box_area

EPS = 1e-6


def box_iou_pairwise(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.max(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=EPS)


def min_area_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    return intersection / torch.min(area1[:, None], area2[None]).clamp(min=EPS)


def suppress_nested(
    boxes_xyxy: Tensor,
    scores: Tensor,
    labels: Tensor,
    nms_iou: float,
    iomin_threshold: float = 0.8,
) -> Tensor:
    keep = batched_nms(boxes_xyxy, scores, labels, nms_iou)
    kept = boxes_xyxy[keep]
    overlap = min_area_box_iou(kept, kept).triu(diagonal=1)
    same_class = labels[keep][:, None] == labels[keep][None, :]
    nested = ((overlap > iomin_threshold) & same_class).any(dim=0)
    return keep[~nested]
