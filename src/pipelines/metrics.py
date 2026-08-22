"""Métricas de detección compartidas por todos los detectores.

Trabajan sobre cajas ya decodificadas (cxcywh normalizado, score y clase) sin saber de
qué arquitectura salieron, así que los tres modelos se miden con el mismo código.
"""

from collections import defaultdict
from typing import NamedTuple

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

Overlaps = list[tuple[list[int], Tensor]]
# por clip: (filas de predictions, matriz P x G)

AP_THRESHOLDS: tuple[float, ...] = (0.25, 0.5)


class Boxes(NamedTuple):
    boxes: Tensor  # (N, 4) cxcywh normalizado
    image_ids: Tensor  # (N,) id del clip, no de la posición en el batch
    labels: Tensor
    scores: Tensor

    def select(self, index: Tensor) -> "Boxes":
        return Boxes(*(field[index] for field in self))


class DetectionMetrics(NamedTuple):
    ap_agnostic: dict[float, float | None]
    recall_agnostic: float | None
    precision_agnostic: float | None
    recall_per_class: dict[int, float | None]
    n_gt: int
    n_predictions: int
    n_above_threshold: int


def concat(chunks: list[Boxes]) -> Boxes:
    if not chunks:  # loader vacío: métricas indefinidas, no un TypeError
        return Boxes(torch.zeros(0, 4), *(torch.zeros(0) for _ in range(3)))
    return Boxes(*(torch.cat(fields).cpu() for fields in zip(*chunks, strict=True)))


def sort_by_score(predictions: Boxes) -> Boxes:
    return predictions.select(predictions.scores.argsort(descending=True, stable=True))


def rows_by_image(image_ids: Tensor) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for row, image_id in enumerate(image_ids.tolist()):
        rows[image_id].append(row)
    return rows


def overlaps(predictions: Boxes, truth: Boxes, class_aware: bool = False) -> Overlaps:
    if not len(predictions.boxes) or not len(truth.boxes):
        return []

    predicted_xyxy = box_convert(predictions.boxes, "cxcywh", "xyxy")
    truth_xyxy = box_convert(truth.boxes, "cxcywh", "xyxy")
    truth_rows = rows_by_image(truth.image_ids)

    matrices: Overlaps = []
    for image_id, rows in rows_by_image(predictions.image_ids).items():
        columns = truth_rows.get(image_id)
        if columns is None:
            continue
        matrix = box_iou(predicted_xyxy[rows], truth_xyxy[columns])
        if class_aware:
            same = predictions.labels[rows][:, None] == truth.labels[columns][None, :]
            matrix = matrix.masked_fill(~same, -1.0)
        matrices.append((rows, matrix))
    return matrices


def hits(matrices: Overlaps, n_predictions: int, iou_threshold: float) -> Tensor:
    found = torch.zeros(n_predictions)
    for rows, matrix in matrices:
        available = matrix.masked_fill(matrix < iou_threshold, -1.0)
        for _ in range(available.shape[1]):
            best = available.max(dim=1)
            candidates = best.values >= iou_threshold
            if not candidates.any():
                break
            row = int(candidates.to(torch.uint8).argmax())  # la de mayor score
            found[rows[row]] = 1.0
            available[row] = -1.0  # una predicción cuenta por un solo GT
            available[:, int(best.indices[row])] = -1.0  # y un GT se consume una vez
    return found


def average_precision(found: Tensor, n_gt: int) -> float | None:
    if n_gt == 0:
        return None
    if not len(found):
        return 0.0
    recall = found.cumsum(0) / n_gt
    precision = found.cumsum(0) / torch.arange(1, len(found) + 1)
    return float(((recall - torch.cat([recall.new_zeros(1), recall[:-1]])) * precision).sum())


def detection_metrics(
    predictions: Boxes,
    truth: Boxes,
    n_classes: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    ap_thresholds: tuple[float, ...] = AP_THRESHOLDS,
    detailed: bool = True,
) -> DetectionMetrics:
    """`predictions` tiene que venir ordenado por score descendente (`sort_by_score`)."""
    n_gt = len(truth.boxes)
    n_predictions = len(predictions.boxes)
    # `k` = detecciones sobre el punto de operación; como están ordenadas por score, son
    # las primeras k y `found[:k]` es su resultado sin necesidad de rehacer el greedy.
    k = int((predictions.scores >= score_threshold).sum())

    agnostic = overlaps(predictions, truth)
    tp = int(hits(agnostic, n_predictions, iou_threshold)[:k].sum())

    ap_agnostic: dict[float, float | None] = {}
    recall_per_class: dict[int, float | None] = {}
    if detailed:
        ap_agnostic = {
            threshold: average_precision(hits(agnostic, n_predictions, threshold), n_gt)
            for threshold in ap_thresholds
        }
        class_hits = hits(
            overlaps(predictions, truth, class_aware=True), n_predictions, iou_threshold
        )
        predicted_labels = predictions.labels[:k]
        for class_id in range(n_classes):
            class_gt = int((truth.labels == class_id).sum())
            recall_per_class[class_id] = (
                float(class_hits[:k][predicted_labels == class_id].sum()) / class_gt
                if class_gt
                else None
            )

    return DetectionMetrics(
        ap_agnostic=ap_agnostic,
        recall_agnostic=tp / n_gt if n_gt else None,
        precision_agnostic=tp / k if k else None,
        recall_per_class=recall_per_class,
        n_gt=n_gt,
        n_predictions=n_predictions,
        n_above_threshold=k,
    )
