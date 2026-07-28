"""Matcher húngaro y pérdida de conjunto de DETR (Carion et al., 2020).

Sigue la implementación de referencia de facebookresearch/detr: el canal
`n_classes` del head es el "no-objeto" y las cajas se comparan en `xyxy`.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops import box_convert

from architectures.iou import get_iou_fn

Target = dict[str, Tensor]
Indices = list[tuple[Tensor, Tensor]]


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_iou: float = 2.0,
        iou_type: str = "iou",
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_iou = cost_iou
        self.iou_type = iou_type
        self.iou_fn = get_iou_fn(iou_type)

    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor], targets: list[Target]) -> Indices:
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device

        probabilities = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        predicted_boxes = outputs["pred_boxes"].flatten(0, 1)
        target_labels = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets])

        cost_class = -probabilities[:, target_labels]
        cost_bbox = torch.cdist(predicted_boxes, target_boxes, p=1)
        cost_iou = -self.iou_fn(
            box_convert(predicted_boxes, "cxcywh", "xyxy"),
            box_convert(target_boxes, "cxcywh", "xyxy"),
        )

        cost_matrix = (
            self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_iou * cost_iou
        )
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(target["boxes"]) for target in targets]
        assignments = [
            linear_sum_assignment(cost[index])
            for index, cost in enumerate(cost_matrix.split(sizes, dim=-1))
        ]
        return [
            (
                torch.as_tensor(query_index, dtype=torch.int64, device=device),
                torch.as_tensor(target_index, dtype=torch.int64, device=device),
            )
            for query_index, target_index in assignments
        ]


class SetCriterion(nn.Module):
    empty_weight: Tensor

    def __init__(
        self,
        n_classes: int = 1,
        matcher: nn.Module | None = None,
        eos_coef: float = 0.1,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_iou: float = 2.0,
        iou_type: str = "iou",
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.background_id = n_classes
        self.matcher = matcher or HungarianMatcher(iou_type=iou_type)
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_iou = weight_iou
        self.iou_type = iou_type
        self.iou_fn = get_iou_fn(iou_type)

        empty_weight = torch.ones(n_classes + 1)
        empty_weight[self.background_id] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _permutation_index(indices: Indices) -> tuple[Tensor, Tensor]:
        batch_index = torch.cat(
            [torch.full_like(query_index, batch) for batch, (query_index, _) in enumerate(indices)]
        )
        query_index = torch.cat([query_index for query_index, _ in indices])
        return batch_index, query_index

    def forward(self, outputs: dict[str, Tensor], targets: list[Target]) -> dict[str, Tensor]:
        indices = self.matcher(outputs, targets)
        index = self._permutation_index(indices)
        num_boxes = max(sum(len(target["labels"]) for target in targets), 1)

        logits = outputs["pred_logits"]
        target_classes = torch.full(
            logits.shape[:2], self.background_id, dtype=torch.int64, device=logits.device
        )
        target_classes[index] = torch.cat(
            [
                target["labels"][target_index]
                for target, (_, target_index) in zip(targets, indices, strict=True)
            ]
        )
        loss_class = F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)

        predicted_boxes = outputs["pred_boxes"][index]
        matched_boxes = torch.cat(
            [
                target["boxes"][target_index]
                for target, (_, target_index) in zip(targets, indices, strict=True)
            ]
        )
        loss_bbox = F.l1_loss(predicted_boxes, matched_boxes, reduction="sum") / num_boxes
        iou = self.iou_fn(
            box_convert(predicted_boxes, "cxcywh", "xyxy"),
            box_convert(matched_boxes, "cxcywh", "xyxy"),
        )
        loss_iou = (1 - torch.diag(iou)).sum() / num_boxes

        return {
            "loss_cls": loss_class,
            "loss_bbox": loss_bbox,
            "loss_iou": loss_iou,
            "loss_total": (
                self.weight_class * loss_class
                + self.weight_bbox * loss_bbox
                + self.weight_iou * loss_iou
            ),
        }
