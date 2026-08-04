from collections import defaultdict
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.ops import box_convert, box_iou
from tqdm.auto import tqdm

from architectures.deformable_detr import predict_scores
from architectures.iou import box_iou_pairwise, suppress_nested
from pipelines.common import LOSS_KEYS, Losses, Target, to_device

Overlaps = list[tuple[list[int], Tensor]]
# por clip: (filas de predictions, matriz P x G)


class EvalMetrics(NamedTuple):
    losses: Losses
    mean_iou: float  # encuadre: sobre los pares del matcher, sin umbral de score
    accuracy: float  # clasificación: sobre los mismos pares, sin umbral de score
    ap_agnostic: dict[float, float | None]
    recall_agnostic: float | None
    precision_agnostic: float | None
    recall_per_class: dict[int, float | None]
    confusion: Tensor  # (n_classes, n_classes + 1); la última columna es "no-objeto"


class Boxes(NamedTuple):
    boxes: Tensor  # (N, 4) cxcywh normalizado
    image_ids: Tensor  # (N,) id del clip, no de la posición en el batch
    labels: Tensor
    scores: Tensor

    def select(self, index: Tensor) -> "Boxes":
        return Boxes(*(field[index] for field in self))


def _concat(chunks: list[Boxes]) -> Boxes:
    if not chunks:  # loader vacío: métricas indefinidas, no un TypeError
        return Boxes(torch.zeros(0, 4), *(torch.zeros(0) for _ in range(3)))
    return Boxes(*(torch.cat(fields).cpu() for fields in zip(*chunks, strict=True)))


def _rows_by_image(image_ids: Tensor) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for row, image_id in enumerate(image_ids.tolist()):
        rows[image_id].append(row)
    return rows


def _overlaps(predictions: Boxes, truth: Boxes, class_aware: bool = False) -> Overlaps:
    if not len(predictions.boxes) or not len(truth.boxes):
        return []

    predicted_xyxy = box_convert(predictions.boxes, "cxcywh", "xyxy")
    truth_xyxy = box_convert(truth.boxes, "cxcywh", "xyxy")
    truth_rows = _rows_by_image(truth.image_ids)

    overlaps: Overlaps = []
    for image_id, rows in _rows_by_image(predictions.image_ids).items():
        columns = truth_rows.get(image_id)
        if columns is None:
            continue
        matrix = box_iou(predicted_xyxy[rows], truth_xyxy[columns])
        if class_aware:
            same = predictions.labels[rows][:, None] == truth.labels[columns][None, :]
            matrix = matrix.masked_fill(~same, -1.0)
        overlaps.append((rows, matrix))
    return overlaps


def _hits(overlaps: Overlaps, n_predictions: int, iou_threshold: float) -> Tensor:
    hits = torch.zeros(n_predictions)
    for rows, matrix in overlaps:
        available = matrix.masked_fill(matrix < iou_threshold, -1.0)
        for _ in range(available.shape[1]):
            best = available.max(dim=1)
            candidates = best.values >= iou_threshold
            if not candidates.any():
                break
            row = int(candidates.to(torch.uint8).argmax())  # la de mayor score
            hits[rows[row]] = 1.0
            available[row] = -1.0  # una predicción cuenta por un solo GT
            available[:, int(best.indices[row])] = -1.0  # y un GT se consume una vez
    return hits


def _average_precision(hits: Tensor, n_gt: int) -> float | None:
    if n_gt == 0:
        return None
    if not len(hits):
        return 0.0
    recall = hits.cumsum(0) / n_gt
    precision = hits.cumsum(0) / torch.arange(1, len(hits) + 1)
    return float(((recall - torch.cat([recall.new_zeros(1), recall[:-1]])) * precision).sum())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    matcher: nn.Module,
    device: torch.device | str = "cpu",
    n_classes: int = 1,
    iou_threshold: float = 0.5,
    ap_thresholds: tuple[float, ...] = (0.25, 0.5),
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

    predictions = _concat(predicted_chunks)
    predictions = predictions.select(predictions.scores.argsort(descending=True, stable=True))
    truth = _concat(truth_chunks)

    n_gt = len(truth.boxes)
    n_predictions = len(predictions.boxes)
    # `k` = detecciones sobre el punto de operación; como están ordenadas por score, son
    # las primeras k y `hits[:k]` es su resultado sin necesidad de rehacer el greedy.
    k = int((predictions.scores >= score_threshold).sum())

    agnostic = _overlaps(predictions, truth)
    tp = int(_hits(agnostic, n_predictions, iou_threshold)[:k].sum())

    ap_agnostic: dict[float, float | None] = {}
    recall_per_class: dict[int, float | None] = {}
    if detailed:
        ap_agnostic = {
            threshold: _average_precision(_hits(agnostic, n_predictions, threshold), n_gt)
            for threshold in ap_thresholds
        }
        by_class = _overlaps(predictions, truth, class_aware=True)
        class_hits = _hits(by_class, n_predictions, iou_threshold)[:k]
        predicted_labels = predictions.labels[:k]
        for class_id in range(n_classes):
            class_gt = int((truth.labels == class_id).sum())
            recall_per_class[class_id] = (
                float(class_hits[predicted_labels == class_id].sum()) / class_gt
                if class_gt
                else None
            )

    return EvalMetrics(
        losses=Losses(**{key: value / max(len(loader), 1) for key, value in totals.items()}),
        mean_iou=iou_sum / max(matched, 1),
        accuracy=matched_correct / max(matched, 1),
        ap_agnostic=ap_agnostic,
        recall_agnostic=tp / n_gt if n_gt else None,
        precision_agnostic=tp / k if k else None,
        recall_per_class=recall_per_class,
        confusion=confusion,
    )
