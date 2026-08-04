from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms, box_convert, box_iou
from tqdm.auto import tqdm

from architectures.deformable_detr import predict_scores
from architectures.iou import box_iou_pairwise
from pipelines.common import LOSS_KEYS, Losses, Target, to_device

IouFn = Callable[[Tensor, Tensor], Tensor]


class EvalMetrics(NamedTuple):
    losses: Losses
    mean_iou: float
    accuracy: float
    ap_agnostic: dict[float, float | None]
    recall_agnostic: float | None
    precision_agnostic: float | None
    fp_per_tp_agnostic: float | None
    recall_per_class: dict[int, float | None]
    precision_per_class: dict[int, float | None]
    # (n_classes, n_classes + 1): filas=clase real, columnas=predicha (+ "no-objeto")
    confusion: Tensor


class Boxes(NamedTuple):
    boxes: Tensor  # (N, 4) cxcywh normalizado
    image_ids: Tensor  # (N,) id del clip, no del batch
    labels: Tensor  # (N,)
    scores: Tensor  # (N,)

    def select(self, mask: Tensor) -> "Boxes":
        return Boxes(*(field[mask] for field in self))

    def to_cpu(self) -> "Boxes":
        return Boxes(*(field.cpu() for field in self))

    def sorted_by_score(self) -> "Boxes":
        return self.select(self.scores.argsort(descending=True, stable=True))

    @property
    def box_count(self) -> int:
        return len(self.boxes)


def _concat(chunks: list[Boxes]) -> Boxes:
    if not chunks:  # loader vacío: mejor métricas en cero que un TypeError
        return Boxes(torch.zeros(0, 4), *(torch.zeros(0) for _ in range(3)))
    return Boxes(*(torch.cat(fields) for fields in zip(*chunks, strict=True)))


def _rows_by_image(image_ids: Tensor) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = defaultdict(list)
    for row, image_id in enumerate(image_ids.tolist()):
        rows[image_id].append(row)
    return rows


class Matching:
    def __init__(self, predictions: Boxes, truth: Boxes, iou_fn: IouFn) -> None:
        self.n_predictions = predictions.box_count
        self.per_image: list[tuple[list[int], Tensor]] = []
        if predictions.box_count == 0 or truth.box_count == 0:
            return

        predicted_xyxy = box_convert(predictions.boxes, "cxcywh", "xyxy")
        truth_xyxy = box_convert(truth.boxes, "cxcywh", "xyxy")
        truth_rows = _rows_by_image(truth.image_ids)
        for image_id, rows in _rows_by_image(predictions.image_ids).items():
            columns = truth_rows.get(image_id)
            if columns is not None:
                self.per_image.append((rows, iou_fn(predicted_xyxy[rows], truth_xyxy[columns])))

    def hits(self, iou_threshold: float) -> Tensor:
        """1.0 en cada predicción que captura un GT todavía libre de su mismo clip.

        Greedy por score: cada GT se lo lleva la predicción de mayor score que lo
        supere. `predictions` tiene que venir ordenado descendente.
        """
        hits = torch.zeros(self.n_predictions)
        for rows, overlaps in self.per_image:
            # Se itera sobre los GT (uno o dos por clip) y no sobre las predicciones
            # (una por query): mismo resultado greedy, muchas menos vueltas de Python.
            available = overlaps.masked_fill(overlaps < iou_threshold, -1.0)
            for _ in range(available.shape[1]):
                best_per_row = available.max(dim=1)
                candidates = best_per_row.values >= iou_threshold
                if not candidates.any():
                    break
                row = int(candidates.to(torch.uint8).argmax())  # la de mayor score
                column = int(best_per_row.indices[row])  # su GT libre de mayor IoU
                hits[rows[row]] = 1.0
                available[row] = -1.0  # una predicción cuenta por un solo GT
                available[:, column] = -1.0  # y un GT se consume una sola vez
        return hits


def _average_precision(hits: Tensor, n_gt: int) -> float | None:
    """AP sin interpolar (como `sklearn.metrics.average_precision_score`): rankea por
    score y suma la precisión en cada acierto nuevo."""
    if n_gt == 0:
        return None
    if len(hits) == 0:
        return 0.0

    recall = torch.cumsum(hits, 0) / n_gt
    precision = torch.cumsum(hits, 0) / torch.arange(1, len(hits) + 1)
    return float(((recall - torch.cat([recall.new_zeros(1), recall[:-1]])) * precision).sum())


class PointMetrics(NamedTuple):
    """`None` = indefinida, no cero. Un split sin GT no tiene recall, un umbral que no
    deja pasar nada no tiene precisión, y sin ningún TP los FP por acierto son infinitos.
    Devolver 0.0 en esos casos hacía pasar por buenas métricas que no existen."""

    recall: float | None
    precision: float | None
    fp_per_tp: float | None


def _point_metrics(hits: Tensor, n_gt: int, k: int) -> PointMetrics:
    """Recall, precisión y falsos positivos por acierto en el punto de operación fijado
    por `score_threshold`, quedándose con las `k` mejores detecciones."""
    tp = int(hits[:k].sum())
    return PointMetrics(
        recall=tp / n_gt if n_gt else None,
        precision=tp / k if k else None,
        fp_per_tp=(k - tp) / tp if tp else None,
    )


def _class_point_metrics(
    predictions: Boxes,
    truth: Boxes,
    iou_fn: IouFn,
    iou_threshold: float,
    score_threshold: float,
) -> PointMetrics | None:
    """`None` cuando la clase no tiene ground truth en el split (métricas indefinidas)."""
    if truth.box_count == 0:
        return None
    hits = Matching(predictions, truth, iou_fn).hits(iou_threshold)  # el filtro mantiene el orden
    k = int((predictions.scores >= score_threshold).sum())
    return _point_metrics(hits, truth.box_count, k)


def _keep_after_nms(boxes: Tensor, scores: Tensor, labels: Tensor, nms_iou: float) -> Tensor:
    """Índices que sobreviven al NMS por clase, en el mismo espacio normalizado que usa
    `inference_pipeline.predict`: si acá no se suprime nada, `fp_per_tp` cuenta como
    falsos positivos duplicados que en producción el NMS ya eliminó.

    La diferencia que queda con producción es de alcance: acá el NMS es por clip, y en
    `predict` es sobre el archivo entero, donde además suprime la misma vocalización
    detectada en dos ventanas solapadas. Eso no se puede medir por ventana."""
    return batched_nms(box_convert(boxes, "cxcywh", "xyxy"), scores, labels, nms_iou)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    matcher: nn.Module,
    device: torch.device | str = "cpu",
    n_classes: int = 1,
    iou_threshold: float = 0.5,
    ap_thresholds: tuple[float, ...] = (0.25, 0.3, 0.5, 0.75),
    score_threshold: float = 0.5,
    nms_iou: float | None = 0.3,
    detailed: bool = True,
    desc: str = "val",
) -> EvalMetrics:
    """Corre un split completo y devuelve las métricas agregadas.

    `detailed=False` se salta el AP por umbral y el desglose por clase: cada uno repite
    el emparejamiento greedy (una vez por umbral, una vez por clase), y son las dos
    partes caras de esta función. `train.py` sólo los necesita cada `DETAIL_EVERY`
    épocas; el resto del tiempo pasar `detailed=False` evita ese trabajo.
    """
    model.eval()

    # `criterion` empareja por su cuenta (una vez por capa del decoder). Si su matcher no
    # es este, la matriz de confusión y el IoU medio describirían un emparejamiento que no
    # es el que produjo la pérdida que se reporta al lado.
    criterion_matcher = getattr(criterion, "matcher", None)
    if criterion_matcher is not None and criterion_matcher is not matcher:
        raise ValueError(
            "`matcher` tiene que ser el mismo objeto que `criterion.matcher`: si no, las "
            "métricas de emparejamiento y la pérdida hablan de asignaciones distintas."
        )

    iou_fn = box_iou
    pairwise_iou_fn = box_iou_pairwise
    totals = dict.fromkeys(LOSS_KEYS, 0.0)
    matched, matched_correct, iou_sum = 0, 0, 0.0
    # filas=clase real, columnas=clase predicha + "no-objeto" (background_id = n_classes)
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
        # clase incluyendo el canal de "no-objeto": es lo que el modelo realmente afirma
        decided = outputs["pred_logits"].argmax(-1)

        for b, (query_idx, target_idx) in enumerate(matcher(outputs, targets)):
            image_id = next_image_id + b  # identifica el clip, no la posición en el batch
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
                    pairwise_iou_fn(  # ya emparejadas: N solapes, no una matriz N x N
                        box_convert(pred_boxes[b, query_idx], "cxcywh", "xyxy"),
                        box_convert(target["boxes"][target_idx], "cxcywh", "xyxy"),
                    ).sum()
                )

            keep = (
                _keep_after_nms(pred_boxes[b], scores[b], labels[b], nms_iou)
                if nms_iou is not None
                else slice(None)
            )
            kept_boxes = pred_boxes[b][keep]
            predicted_chunks.append(
                Boxes(
                    kept_boxes,
                    torch.full((len(kept_boxes),), image_id, device=kept_boxes.device),
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

    predictions = _concat(predicted_chunks).to_cpu().sorted_by_score()
    truth = _concat(truth_chunks).to_cpu()

    matching = Matching(predictions, truth, iou_fn)
    hits = matching.hits(iou_threshold)
    k = int((predictions.scores >= score_threshold).sum())
    point = _point_metrics(hits, truth.box_count, k)

    ap_agnostic = (
        {
            threshold: _average_precision(matching.hits(threshold), truth.box_count)
            for threshold in ap_thresholds
        }
        if detailed
        else {}
    )
    class_metrics = (
        {
            class_id: _class_point_metrics(
                predictions.select(predictions.labels == class_id),
                truth.select(truth.labels == class_id),
                iou_fn,
                iou_threshold,
                score_threshold,
            )
            for class_id in range(n_classes)
        }
        if detailed
        else {}
    )

    return EvalMetrics(
        losses=Losses(**{key: value / max(len(loader), 1) for key, value in totals.items()}),
        mean_iou=iou_sum / max(matched, 1),
        accuracy=matched_correct / max(matched, 1),
        ap_agnostic=ap_agnostic,
        recall_agnostic=point.recall,
        precision_agnostic=point.precision,
        fp_per_tp_agnostic=point.fp_per_tp,
        recall_per_class={
            class_id: metrics.recall if metrics else None
            for class_id, metrics in class_metrics.items()
        },
        precision_per_class={
            class_id: metrics.precision if metrics else None
            for class_id, metrics in class_metrics.items()
        },
        confusion=confusion,
    )
