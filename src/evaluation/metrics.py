from collections import defaultdict
from typing import NamedTuple

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

# Mínimo IoU para considerar una predicción como acierto
MATCH_IOU = 0.3
# Mínimo score para considerar una predicción
SCORE_FLOOR = 0.001
# Cantidad común de detecciones por clip
MAX_DETECTIONS = 100
# Ancho a partir de la cual una clase ocupa la ventana entera
WINDOW_CLASS_WIDTH = 0.95


class Boxes(NamedTuple):
    boxes: Tensor
    image_ids: Tensor
    labels: Tensor
    scores: Tensor

    def select(self, index: Tensor) -> "Boxes":
        return Boxes(*(field[index] for field in self))


class ImageOverlap(NamedTuple):
    prediction_rows: list[int]
    truth_rows: list[int]
    iou: Tensor


Overlaps = list[ImageOverlap]
PerClassAP = dict[int, float | None]


class DetectionMetrics(NamedTuple):
    recall: float | None
    precision: float | None
    map_30: float | None


def concat(chunks: list[Boxes]) -> Boxes:
    if not chunks:
        return Boxes(torch.zeros(0, 4), *(torch.zeros(0) for _ in range(3)))
    return Boxes(*(torch.cat(fields).cpu() for fields in zip(*chunks, strict=True)))


def sort_by_score(predictions: Boxes) -> Boxes:
    return predictions.select(predictions.scores.argsort(descending=True, stable=True))


def rows_by_image(image_ids: Tensor) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for row, image_id in enumerate(image_ids.tolist()):
        rows[image_id].append(row)
    return rows


def overlaps(predictions: Boxes, truth: Boxes, class_aware: bool) -> Overlaps:
    if not len(predictions.boxes) or not len(truth.boxes):
        return []
    predicted_xyxy = box_convert(predictions.boxes, "cxcywh", "xyxy")
    truth_xyxy = box_convert(truth.boxes, "cxcywh", "xyxy")
    truth_rows = rows_by_image(truth.image_ids)

    per_image: Overlaps = []
    for image_id, rows in rows_by_image(predictions.image_ids).items():
        columns = truth_rows.get(image_id)
        if columns is None:
            continue
        iou = box_iou(predicted_xyxy[rows], truth_xyxy[columns])
        if class_aware:
            same_class = predictions.labels[rows][:, None] == truth.labels[columns][None, :]
            iou = iou.masked_fill(~same_class, -1.0)
        per_image.append(ImageOverlap(rows, columns, iou))
    return per_image


def assignments(per_image: Overlaps, iou_threshold: float) -> list[tuple[int, int]]:
    # Cada GT se asigna a lo sumo a una predicción.
    pairs: list[tuple[int, int]] = []
    for overlap in per_image:
        available = overlap.iou.masked_fill(overlap.iou < iou_threshold, -1.0)
        for _ in range(available.shape[1]):
            best = available.max(dim=1)
            candidates = best.values >= iou_threshold
            if not candidates.any():
                break
            highest_scoring = int(candidates.to(torch.uint8).argmax())
            claimed_truth = int(best.indices[highest_scoring])
            pairs.append(
                (overlap.prediction_rows[highest_scoring], overlap.truth_rows[claimed_truth])
            )
            available[highest_scoring] = -1.0
            available[:, claimed_truth] = -1.0
    return pairs


def hits(per_image: Overlaps, n_predictions: int, iou_threshold: float) -> Tensor:
    found = torch.zeros(n_predictions)
    for prediction_row, _ in assignments(per_image, iou_threshold):
        found[prediction_row] = 1.0
    return found


def average_precision(found: Tensor, n_gt: int) -> float | None:
    if n_gt == 0:
        return None
    if not len(found):
        return 0.0
    recall = found.cumsum(0) / n_gt
    precision = found.cumsum(0) / torch.arange(1, len(found) + 1)
    envelope = precision.flip(0).cummax(0).values.flip(0)
    previous_recall = torch.cat([recall.new_zeros(1), recall[:-1]])
    return float(((recall - previous_recall) * envelope).sum())


def average_precision_per_class(predictions: Boxes, truth: Boxes, n_classes: int) -> PerClassAP:
    ap: PerClassAP = {}
    for class_id in range(n_classes):
        class_predictions = predictions.select(predictions.labels == class_id)
        class_truth = truth.select(truth.labels == class_id)
        found = hits(
            overlaps(class_predictions, class_truth, False), len(class_predictions.boxes), MATCH_IOU
        )
        ap[class_id] = average_precision(found, len(class_truth.boxes))
    return ap


def mean_average_precision(ap: PerClassAP, classes: list[int] | None = None) -> float | None:
    scored = [v for v in (ap[c] for c in (ap if classes is None else classes)) if v is not None]
    return sum(scored) / len(scored) if scored else None


def window_classes(truth: Boxes, n_classes: int) -> list[int]:
    # Clases cuya llamada dura más que el clip: la caja ocupa la ventana y no es detección.
    saturated = []
    for class_id in range(n_classes):
        widths = truth.boxes[truth.labels == class_id][:, 2]
        if len(widths) and float(widths.median()) >= WINDOW_CLASS_WIDTH:
            saturated.append(class_id)
    return saturated


def detection_classes(truth: Boxes, n_classes: int) -> list[int]:
    saturated = set(window_classes(truth, n_classes))
    return [c for c in range(n_classes) if c not in saturated]


def detection_metrics(
    predictions: Boxes, truth: Boxes, n_classes: int, score_threshold: float = 0.5
) -> DetectionMetrics:
    n_gt = len(truth.boxes)
    n_above = int((predictions.scores >= score_threshold).sum())
    found = hits(overlaps(predictions, truth, class_aware=True), len(predictions.boxes), MATCH_IOU)
    true_positives = int(found[:n_above].sum())
    ap = average_precision_per_class(predictions, truth, n_classes)
    return DetectionMetrics(
        recall=true_positives / n_gt if n_gt else None,
        precision=true_positives / n_above if n_above else None,
        map_30=mean_average_precision(ap, detection_classes(truth, n_classes)),
    )
