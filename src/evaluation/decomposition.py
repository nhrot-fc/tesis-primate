from typing import NamedTuple

import torch
from torch import Tensor

from evaluation.metrics import (
    MATCH_IOU,
    Boxes,
    assignments,
    average_precision,
    class_average_precision,
    hits,
    mean_average_precision,
    overlaps,
)

TIME_AXIS, FREQUENCY_AXIS = 0, 1
BOX_ORACLE_IOU = 0.1


class Decomposition(NamedTuple):
    ap_agnostic: float | None
    iou_time: float | None
    iou_freq: float | None
    top1: float | None
    species_top1: float | None
    label_oracle_gain: float | None
    box_oracle_gain: float | None
    n_matched: int


def macro_ap(predictions: Boxes, truth: Boxes, classes: list[int]) -> float | None:
    return mean_average_precision(
        {class_id: class_average_precision(predictions, truth, class_id) for class_id in classes}
    )


def edges(boxes: Tensor, axis: int) -> tuple[Tensor, Tensor]:
    center, size = boxes[:, axis], boxes[:, axis + 2]
    return center - size / 2, center + size / 2


def axis_iou(predicted: Tensor, truth: Tensor, axis: int) -> Tensor:
    low, high = edges(predicted, axis)
    truth_low, truth_high = edges(truth, axis)
    intersection = (torch.minimum(high, truth_high) - torch.maximum(low, truth_low)).clamp(min=0)
    union = torch.maximum(high, truth_high) - torch.minimum(low, truth_low)
    return intersection / union.clamp(min=1e-9)


def indices(pairs: list[tuple[int, int]]) -> tuple[Tensor, Tensor]:
    rows, columns = zip(*pairs, strict=True) if pairs else ((), ())
    return torch.tensor(rows, dtype=torch.int64), torch.tensor(columns, dtype=torch.int64)


def oracle(predictions: Boxes, truth: Boxes, pairs: list[tuple[int, int]], field: str) -> Boxes:
    rows, columns = indices(pairs)
    corrected = getattr(predictions, field).clone()
    corrected[rows] = getattr(truth, field)[columns]
    return predictions._replace(**{field: corrected})


def species_of(class_names: list[str]) -> Tensor:
    species = sorted({name.split("/")[0] for name in class_names})
    return torch.tensor([species.index(name.split("/")[0]) for name in class_names])


def decompose(
    predictions: Boxes, truth: Boxes, class_names: list[str], classes: list[int]
) -> Decomposition:
    truth = truth.select(torch.isin(truth.labels, torch.tensor(classes, dtype=truth.labels.dtype)))
    agnostic = overlaps(predictions, truth, class_aware=False)
    pairs = assignments(agnostic, MATCH_IOU)
    ap_agnostic = average_precision(
        hits(agnostic, len(predictions.boxes), MATCH_IOU), len(truth.boxes)
    )
    if not pairs:
        return Decomposition(ap_agnostic, None, None, None, None, None, None, 0)

    rows, columns = indices(pairs)
    predicted_boxes, truth_boxes = predictions.boxes[rows], truth.boxes[columns]
    predicted_labels, truth_labels = predictions.labels[rows], truth.labels[columns]
    species = species_of(class_names)
    base = macro_ap(predictions, truth, classes)

    def gain(corrected: Boxes) -> float | None:
        recovered = macro_ap(corrected, truth, classes)
        return None if base is None or recovered is None else recovered - base

    box_pairs = assignments(overlaps(predictions, truth, class_aware=True), BOX_ORACLE_IOU)
    return Decomposition(
        ap_agnostic=ap_agnostic,
        iou_time=float(axis_iou(predicted_boxes, truth_boxes, TIME_AXIS).median()),
        iou_freq=float(axis_iou(predicted_boxes, truth_boxes, FREQUENCY_AXIS).median()),
        top1=float((predicted_labels == truth_labels).float().mean()),
        species_top1=float((species[predicted_labels] == species[truth_labels]).float().mean()),
        label_oracle_gain=gain(oracle(predictions, truth, pairs, "labels")),
        box_oracle_gain=gain(oracle(predictions, truth, box_pairs, "boxes")),
        n_matched=len(pairs),
    )
