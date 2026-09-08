from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from core.config import P

MATCH_IOU = 0.3
COCO_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
MAP_THRESHOLDS: tuple[float, ...] = (MATCH_IOU, *COCO_THRESHOLDS)
BETA = 2.0
SCORE_FLOOR = 0.001


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
    f_beta: float | None
    fp_per_hour: float | None
    map_30: float | None
    map_50: float | None
    map_50_95: float | None
    n_gt: int
    n_predictions: int
    n_above_threshold: int


def f_beta(precision: float | None, recall: float | None, beta: float = BETA) -> float | None:
    if precision is None or recall is None or precision + recall <= 0:
        return None
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def false_positives_per_hour(false_positives: int, n_images: int) -> float | None:
    hours = n_images * P.clip_len_s / 3600
    return false_positives / hours if hours else None


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
    previous_recall = torch.cat([recall.new_zeros(1), recall[:-1]])
    return float(((recall - previous_recall) * precision).sum())


def class_average_precision(predictions: Boxes, truth: Boxes, class_id: int) -> float | None:
    class_predictions = predictions.select(predictions.labels == class_id)
    class_truth = truth.select(truth.labels == class_id)
    return average_precision(
        # Ya está filtrado a una clase, así que enmascarar por clase no cambiaría nada.
        hits(
            overlaps(class_predictions, class_truth, False), len(class_predictions.boxes), MATCH_IOU
        ),
        len(class_truth.boxes),
    )


def average_precision_per_class(
    predictions: Boxes, truth: Boxes, n_classes: int
) -> dict[float, PerClassAP]:
    per_threshold: dict[float, PerClassAP] = {t: {} for t in MAP_THRESHOLDS}
    for class_id in range(n_classes):
        class_predictions = predictions.select(predictions.labels == class_id)
        class_truth = truth.select(truth.labels == class_id)
        per_image = overlaps(class_predictions, class_truth, False)  # ya filtrado a una clase
        for threshold in MAP_THRESHOLDS:
            per_threshold[threshold][class_id] = average_precision(
                hits(per_image, len(class_predictions.boxes), threshold), len(class_truth.boxes)
            )
    return per_threshold


def mean_average_precision(
    ap_per_class: PerClassAP, classes: Iterable[int] | None = None
) -> float | None:
    selected = ap_per_class if classes is None else {c: ap_per_class[c] for c in classes}
    scored = [ap for ap in selected.values() if ap is not None]
    return sum(scored) / len(scored) if scored else None


def mean_average_precision_over(
    ap: dict[float, PerClassAP],
    thresholds: Iterable[float],
    classes: Iterable[int] | None = None,
) -> float | None:
    classes = None if classes is None else list(classes)
    scored = [
        value
        for value in (mean_average_precision(ap[threshold], classes) for threshold in thresholds)
        if value is not None
    ]
    return sum(scored) / len(scored) if scored else None


def detection_metrics(
    predictions: Boxes,
    truth: Boxes,
    n_classes: int,
    n_images: int = 0,
    iou_threshold: float = MATCH_IOU,
    score_threshold: float = 0.5,
    beta: float = BETA,
) -> DetectionMetrics:
    # `predictions` viene ordenado por score descendente (`sort_by_score`): de ahí que las
    # que superan el umbral sean exactamente las primeras `n_above_threshold` filas.
    n_gt = len(truth.boxes)
    n_predictions = len(predictions.boxes)
    n_above_threshold = int((predictions.scores >= score_threshold).sum())

    found = hits(overlaps(predictions, truth, class_aware=True), n_predictions, iou_threshold)
    true_positives = int(found[:n_above_threshold].sum())
    recall = true_positives / n_gt if n_gt else None
    precision = true_positives / n_above_threshold if n_above_threshold else None

    ap = average_precision_per_class(predictions, truth, n_classes)
    return DetectionMetrics(
        recall=recall,
        precision=precision,
        f_beta=f_beta(precision, recall, beta),
        fp_per_hour=false_positives_per_hour(n_above_threshold - true_positives, n_images),
        map_30=mean_average_precision(ap[MATCH_IOU]),
        map_50=mean_average_precision(ap[0.5]),
        map_50_95=mean_average_precision_over(ap, COCO_THRESHOLDS),
        n_gt=n_gt,
        n_predictions=n_predictions,
        n_above_threshold=n_above_threshold,
    )
