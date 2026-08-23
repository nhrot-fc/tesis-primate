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
# COCO promedia la mAP sobre 0.50, 0.55, ..., 0.95: la primera mide "encontró el
# evento", la última "lo encuadró". Reportar las dos separa detección de encuadre.
MAP_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))


class Boxes(NamedTuple):
    boxes: Tensor  # (N, 4) cxcywh normalizado
    image_ids: Tensor  # (N,) id del clip, no de la posición en el batch
    labels: Tensor
    scores: Tensor

    def select(self, index: Tensor) -> "Boxes":
        return Boxes(*(field[index] for field in self))


class DetectionMetrics(NamedTuple):
    ap_agnostic: dict[float, float | None]
    map_50: float | None  # mAP por clase con IoU >= 0.5
    map_50_95: float | None  # la misma, promediada sobre `MAP_THRESHOLDS`
    map_per_threshold: dict[float, float | None]
    ap_per_class_50: dict[int, float | None]
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


def average_precision_per_class(
    predictions: Boxes,
    truth: Boxes,
    n_classes: int,
    thresholds: tuple[float, ...] = MAP_THRESHOLDS,
) -> dict[float, dict[int, float | None]]:
    """AP por umbral de IoU y por clase, al estilo COCO.

    Cada clase se mide sobre sus propias cajas: una predicción sólo puede acertarle a un
    GT de la misma clase, y las clases sin cajas anotadas quedan en `None` para que no
    entren al promedio. `predictions` tiene que venir ordenado por score descendente.
    """
    per_threshold: dict[float, dict[int, float | None]] = {
        threshold: {} for threshold in thresholds
    }
    for class_id in range(n_classes):
        class_predictions = predictions.select(predictions.labels == class_id)
        class_truth = truth.select(truth.labels == class_id)
        # El solape se calcula una vez por clase y se reusa en los diez umbrales.
        matrices = overlaps(class_predictions, class_truth)
        n_gt = len(class_truth.boxes)
        n_predictions = len(class_predictions.boxes)
        for threshold in thresholds:
            per_threshold[threshold][class_id] = average_precision(
                hits(matrices, n_predictions, threshold), n_gt
            )
    return per_threshold


def mean_average_precision(ap_per_class: dict[int, float | None]) -> float | None:
    """Promedio sobre las clases que tienen cajas anotadas; las demás no puntúan."""
    values = [ap for ap in ap_per_class.values() if ap is not None]
    return sum(values) / len(values) if values else None


def detection_metrics(
    predictions: Boxes,
    truth: Boxes,
    n_classes: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    ap_thresholds: tuple[float, ...] = AP_THRESHOLDS,
    map_thresholds: tuple[float, ...] = MAP_THRESHOLDS,
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

    # La mAP no depende del punto de operación ni de `detailed`: es la métrica con la
    # que se comparan las corridas entre sí, así que se calcula en todas las épocas.
    ap_per_class = average_precision_per_class(predictions, truth, n_classes, map_thresholds)
    map_per_threshold = {
        threshold: mean_average_precision(per_class)
        for threshold, per_class in ap_per_class.items()
    }
    scored = [value for value in map_per_threshold.values() if value is not None]

    return DetectionMetrics(
        ap_agnostic=ap_agnostic,
        map_50=map_per_threshold.get(0.5),
        map_50_95=sum(scored) / len(scored) if scored else None,
        map_per_threshold=map_per_threshold,
        ap_per_class_50=ap_per_class.get(0.5, {}),
        recall_agnostic=tp / n_gt if n_gt else None,
        precision_agnostic=tp / k if k else None,
        recall_per_class=recall_per_class,
        n_gt=n_gt,
        n_predictions=n_predictions,
        n_above_threshold=k,
    )
