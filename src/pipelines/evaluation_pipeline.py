from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from tqdm.auto import tqdm

from architectures.deformable_detr import predict_scores
from architectures.iou import box_iou_pairwise, suppress_nested
from pipelines.common import LOSS_KEYS, Losses, Target, to_device
from pipelines.metrics import (
    AP_THRESHOLDS,
    Boxes,
    concat,
    detection_metrics,
    sort_by_score,
)


class EvalMetrics(NamedTuple):
    losses: Losses
    mean_iou: float  # encuadre: sobre los pares del matcher, sin umbral de score
    accuracy: float  # clasificación: sobre los mismos pares, sin umbral de score
    ap_agnostic: dict[float, float | None]
    recall_agnostic: float | None
    precision_agnostic: float | None
    recall_per_class: dict[int, float | None]
    confusion: Tensor  # (n_classes, n_classes + 1); la última columna es "no-objeto"


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    matcher: nn.Module,
    device: torch.device | str = "cpu",
    n_classes: int = 1,
    iou_threshold: float = 0.5,
    ap_thresholds: tuple[float, ...] = AP_THRESHOLDS,
    score_threshold: float = 0.5,
    nms_iou: float | None = 0.3,
    detailed: bool = True,
    desc: str = "val",
) -> EvalMetrics:
    model.eval()
    if getattr(criterion, "matcher", matcher) is not matcher:
        raise ValueError("`matcher` tiene que ser el mismo objeto que `criterion.matcher`")

    totals = dict.fromkeys(LOSS_KEYS, 0.0)
    matched = matched_correct = 0
    iou_sum = 0.0
    confusion = torch.zeros(n_classes, n_classes + 1, dtype=torch.int64)
    predicted_chunks: list[Boxes] = []
    truth_chunks: list[Boxes] = []
    next_image_id = 0

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        images, targets = to_device(batch, device)
        outputs = model(images)
        losses: dict[str, Tensor] = criterion(outputs, targets)
        for key in LOSS_KEYS:
            totals[key] += losses[f"loss_{key}"].item()

        pred_boxes = outputs["pred_boxes"]
        scores, labels = predict_scores(outputs)
        decided = outputs["pred_logits"].argmax(-1)  # incluye el canal de "no-objeto"

        for b, (query_idx, target_idx) in enumerate(matcher(outputs, targets)):
            image_id = next_image_id + b
            target = targets[b]

            if len(query_idx):
                predicted_classes = decided[b, query_idx]
                true_classes = target["labels"][target_idx]
                matched += len(query_idx)
                matched_correct += int((predicted_classes == true_classes).sum())
                confusion.index_put_(
                    (true_classes.cpu(), predicted_classes.cpu()),
                    torch.ones(len(query_idx), dtype=torch.int64),
                    accumulate=True,
                )
                iou_sum += float(
                    box_iou_pairwise(  # ya emparejadas: N solapes, no una matriz N x N
                        box_convert(pred_boxes[b, query_idx], "cxcywh", "xyxy"),
                        box_convert(target["boxes"][target_idx], "cxcywh", "xyxy"),
                    ).sum()
                )

            keep = (
                suppress_nested(
                    box_convert(pred_boxes[b], "cxcywh", "xyxy"), scores[b], labels[b], nms_iou
                )
                if nms_iou is not None
                else slice(None)
            )
            kept = pred_boxes[b][keep]
            predicted_chunks.append(
                Boxes(
                    kept,
                    torch.full((len(kept),), image_id, device=kept.device),
                    labels[b][keep],
                    scores[b][keep],
                )
            )
            n_target = len(target["labels"])
            truth_chunks.append(
                Boxes(
                    target["boxes"],
                    torch.full((n_target,), image_id, device=target["boxes"].device),
                    target["labels"],
                    torch.ones(n_target, device=target["boxes"].device),
                )
            )

        next_image_id += len(targets)

    detections = detection_metrics(
        sort_by_score(concat(predicted_chunks)),
        concat(truth_chunks),
        n_classes=n_classes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        ap_thresholds=ap_thresholds,
        detailed=detailed,
    )
    return EvalMetrics(
        losses=Losses(**{key: value / max(len(loader), 1) for key, value in totals.items()}),
        mean_iou=iou_sum / max(matched, 1),
        accuracy=matched_correct / max(matched, 1),
        ap_agnostic=detections.ap_agnostic,
        recall_agnostic=detections.recall_agnostic,
        precision_agnostic=detections.precision_agnostic,
        recall_per_class=detections.recall_per_class,
        confusion=confusion,
    )
