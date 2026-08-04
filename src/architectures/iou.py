import torch
from torch import Tensor
from torchvision.ops import box_area

EPS = 1e-6


def box_iou_pairwise(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.max(boxes1[:, :2], boxes2[:, :2])
    bottom_right = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    union = area1 + area2 - intersection
    return intersection / union.clamp(min=EPS)
