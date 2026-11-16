from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import (
    box_convert,
    generalized_box_iou,
    generalized_box_iou_loss,
    sigmoid_focal_loss,
)

from utils.boxes import Target

# pred_logits, pred_boxes y aux_outputs (una salida por capa del decodificador)
Outputs = dict[str, Any]
Indices = list[tuple[Tensor, Tensor]]

# Focal sigmoide por clase, sin canal de no-objeto
FOCAL_ALPHA, FOCAL_GAMMA = 0.25, 2.0
# clase / L1 / GIoU
COST_CLASS, COST_BBOX, COST_IOU = 2.0, 5.0, 2.0
WEIGHT_CLASS, WEIGHT_BBOX, WEIGHT_IOU = 1.0, COST_BBOX, COST_IOU


def focal_cost(probabilities: Tensor, alpha: float, gamma: float) -> Tensor:
    # Costo de asignar una clase: focal positiva menos la negativa que se ahorra (Zhu et al. 2021).
    positive = alpha * (1 - probabilities) ** gamma * -(probabilities + 1e-8).log()
    negative = (1 - alpha) * probabilities**gamma * -(1 - probabilities + 1e-8).log()
    return positive - negative


class HungarianMatcher(nn.Module):
    @torch.no_grad()
    def forward(self, outputs: Outputs, targets: list[Target]) -> Indices:
        from scipy.optimize import linear_sum_assignment

        batch_size, n_queries = outputs["pred_logits"].shape[:2]
        device = outputs["pred_logits"].device

        logits = outputs["pred_logits"].flatten(0, 1)
        predicted_boxes = outputs["pred_boxes"].flatten(0, 1)
        target_labels = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets])

        class_cost = focal_cost(logits.sigmoid(), FOCAL_ALPHA, FOCAL_GAMMA)[:, target_labels]
        box_cost = torch.cdist(predicted_boxes, target_boxes, p=1)
        iou_cost = -generalized_box_iou(
            box_convert(predicted_boxes, "cxcywh", "xyxy"),
            box_convert(target_boxes, "cxcywh", "xyxy"),
        )

        cost_matrix = COST_CLASS * class_cost + COST_BBOX * box_cost + COST_IOU * iou_cost
        cost_matrix = cost_matrix.view(batch_size, n_queries, target_boxes.shape[0]).cpu()

        box_counts = [len(target["boxes"]) for target in targets]
        assignments = [
            linear_sum_assignment(cost[index])
            for index, cost in enumerate(cost_matrix.split(box_counts, dim=-1))
        ]
        return [
            (
                torch.as_tensor(query_index, dtype=torch.int64, device=device),
                torch.as_tensor(target_index, dtype=torch.int64, device=device),
            )
            for query_index, target_index in assignments
        ]


class SetCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matcher = HungarianMatcher()

    @staticmethod
    def matched_positions(indices: Indices) -> tuple[Tensor, Tensor]:
        batch_index = torch.cat(
            [torch.full_like(query_index, batch) for batch, (query_index, _) in enumerate(indices)]
        )
        query_index = torch.cat([query_index for query_index, _ in indices])
        return batch_index, query_index

    def losses(self, outputs: Outputs, targets: list[Target]) -> dict[str, Tensor]:
        matched_indices: Indices = self.matcher(outputs, targets)
        matched = self.matched_positions(matched_indices)
        n_matched = max(sum(len(query_index) for query_index, _ in matched_indices), 1)

        logits = outputs["pred_logits"]
        matched_labels = torch.cat(
            [
                target["labels"][target_index]
                for target, (_, target_index) in zip(targets, matched_indices, strict=True)
            ]
        )
        target_scores = torch.zeros_like(logits)
        target_scores[matched[0], matched[1], matched_labels] = 1.0
        loss_class = (
            sigmoid_focal_loss(logits, target_scores, FOCAL_ALPHA, FOCAL_GAMMA, reduction="sum")
            / n_matched
        )

        predicted_boxes = outputs["pred_boxes"][matched]
        matched_boxes = torch.cat(
            [
                target["boxes"][target_index]
                for target, (_, target_index) in zip(targets, matched_indices, strict=True)
            ]
        )
        loss_bbox = F.l1_loss(predicted_boxes, matched_boxes, reduction="sum") / n_matched
        loss_iou = (
            generalized_box_iou_loss(
                box_convert(predicted_boxes, "cxcywh", "xyxy"),
                box_convert(matched_boxes, "cxcywh", "xyxy"),
                reduction="sum",
            )
            / n_matched
        )

        # ya ponderadas
        return {
            "loss_cls": WEIGHT_CLASS * loss_class,
            "loss_bbox": WEIGHT_BBOX * loss_bbox,
            "loss_iou": WEIGHT_IOU * loss_iou,
        }

    def forward(self, outputs: Outputs, targets: list[Target]) -> dict[str, Tensor]:
        losses = self.losses(outputs, targets)
        for aux_outputs in outputs.get("aux_outputs", []):
            for key, value in self.losses(aux_outputs, targets).items():
                losses[key] = losses[key] + value
        return losses
