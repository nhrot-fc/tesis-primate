from collections import defaultdict
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from tqdm.auto import tqdm

from architectures.deformable_detr import predict_scores
from architectures.iou import get_iou_fn

Target = dict[str, Tensor]
Batch = tuple[Tensor, list[Target]]
Metrics = dict[str, float]
Detection = tuple[float, int, Tensor]  # (score, image_id, box cxcywh), CPU tensors only

LOSS_KEYS: tuple[str, ...] = ("total", "cls", "bbox", "iou")


class EvalMetrics(NamedTuple):
    total: float
    cls: float
    bbox: float
    iou: float
    cls_acc: float  # sobre queries emparejadas por el HungarianMatcher
    mean_iou: float  # IoU plano promedio, mismas queries emparejadas
    recall_agn: float  # recall agnóstico de clase @ `iou_threshold`, en `score_threshold` -- la métrica objetivo
    fp_per_tp: float  # falsos positivos por verdadero, mismo punto de operación -- carga de revisión humana
    map: float  # mean AP sobre todos los umbrales de score, bajo `ap_iou_type`
    ap_iou_type: str  # tipo de IoU usado para `map`/`ap_per_class` (p.ej. "iou", "eiou")
    ap_per_class: dict[int, float | None]
    ap_agnostic: dict[float, float | None]  # AP ignorando clase, por umbral de `ap_iou_type`: "¿hay algo ahí?"
    confusion: Tensor  # (n_classes, n_classes), filas=real, cols=predicho


def _to_device(batch: Batch, device: torch.device | str) -> Batch:
    images, targets = batch
    return (
        images.to(device),
        [{key: value.to(device) for key, value in target.items()} for target in targets],
    )


def _epoch_desc(prefix: str, epoch: int | None, epochs: int | None) -> str:
    return prefix if epoch is None else f"{prefix} {epoch + 1}/{epochs}"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    device: torch.device | str = "cpu",
    clip_grad: float = 0.1,
    epoch: int | None = None,
    epochs: int | None = None,
) -> Metrics:
    model.train()
    totals: Metrics = dict.fromkeys(LOSS_KEYS, 0.0)

    progress = tqdm(loader, desc=_epoch_desc("train", epoch, epochs), unit="batch", leave=False)
    for step, batch in enumerate(progress, start=1):
        images, targets = _to_device(batch, device)
        losses: dict[str, Tensor] = criterion(model(images), targets)

        optimizer.zero_grad()
        losses["loss_total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for key in LOSS_KEYS:
            totals[key] += losses[f"loss_{key}"].item()
        progress.set_postfix(loss=totals["total"] / step, lr=optimizer.param_groups[0]["lr"])

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


def _average_precision(
    detections: list[Detection],
    gt_by_image: dict[int, Tensor],
    n_gt: int,
    iou_fn: Callable[[Tensor, Tensor], Tensor],
    iou_threshold: float,
) -> float | None:
    """AP de una clase (sin interpolar, como `sklearn.metrics.average_precision_score`):
    rankea las detecciones por score y suma la precisión en cada acierto nuevo.
    """
    if n_gt == 0:
        return None
    if not detections:
        return 0.0

    detections.sort(key=lambda d: d[0], reverse=True)
    unused = {
        image_id: torch.ones(len(boxes), dtype=torch.bool)
        for image_id, boxes in gt_by_image.items()
    }
    tp = torch.zeros(len(detections))

    for rank, (_, image_id, box) in enumerate(detections):
        gt_boxes = gt_by_image.get(image_id)
        if gt_boxes is None:
            continue
        overlaps = iou_fn(
            box_convert(box[None], "cxcywh", "xyxy"), box_convert(gt_boxes, "cxcywh", "xyxy")
        )
        overlaps = overlaps[0].masked_fill(~unused[image_id], -1.0)
        best = int(overlaps.argmax())
        if overlaps[best] >= iou_threshold:
            unused[image_id][best] = False
            tp[rank] = 1

    recall = torch.cumsum(tp, 0) / n_gt
    precision = torch.cumsum(tp, 0) / torch.arange(1, len(tp) + 1)
    return float(((recall - torch.cat([recall.new_zeros(1), recall[:-1]])) * precision).sum())


def _recall_and_fp(
    detections: list[Detection],
    gt_by_image: dict[int, Tensor],
    n_gt: int,
    iou_fn: Callable[[Tensor, Tensor], Tensor],
    iou_threshold: float,
    score_threshold: float,
) -> tuple[float, float]:
    """Recall y FP/TP en el punto de operación `score_threshold`.

    A diferencia de `_average_precision` (que integra sobre todos los umbrales de
    score), esto fija el umbral que de verdad se usaría en producción y cuenta
    cuántos GT atrapa el modelo ahí y cuántos falsos positivos carga por cada
    verdadero -- la carga de revisión humana que el investigador tiene que aprobar.
    """
    if n_gt == 0:
        return 0.0, 0.0

    kept = [d for d in detections if d[0] >= score_threshold]
    kept.sort(key=lambda d: d[0], reverse=True)
    unused = {
        image_id: torch.ones(len(boxes), dtype=torch.bool)
        for image_id, boxes in gt_by_image.items()
    }
    tp = 0
    for _, image_id, box in kept:
        gt_boxes = gt_by_image.get(image_id)
        if gt_boxes is None:
            continue
        overlaps = iou_fn(
            box_convert(box[None], "cxcywh", "xyxy"), box_convert(gt_boxes, "cxcywh", "xyxy")
        )
        overlaps = overlaps[0].masked_fill(~unused[image_id], -1.0)
        best = int(overlaps.argmax())
        if overlaps[best] >= iou_threshold:
            unused[image_id][best] = False
            tp += 1

    fp = len(kept) - tp
    return tp / n_gt, fp / max(tp, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    matcher: nn.Module,
    device: torch.device | str = "cpu",
    n_classes: int = 1,
    iou_threshold: float = 0.5,
    ap_iou_type: str = "iou",
    agnostic_thresholds: tuple[float, ...] = (0.25, 0.3, 0.5, 0.75),
    score_threshold: float = 0.5,
    epoch: int | None = None,
    epochs: int | None = None,
) -> EvalMetrics:
    """Pérdidas + métricas de detección.

    `cls_acc`/`mean_iou`/`confusion` se calculan sobre las queries que el
    `HungarianMatcher` empareja con un ground truth, sin depender de ningún
    umbral de score. `ap_per_class`/`map`/`ap_agnostic` sí rankean por score,
    pero integran sobre todos los umbrales en vez de fijar uno.

    `recall_agn`/`fp_per_tp` son la métrica objetivo del proyecto (maximizar recall
    a IoU 0.25-0.3 tolerando falsos positivos): a diferencia de `ap_agnostic`, se
    miden en un único punto de operación (`score_threshold`, el umbral que de verdad
    se usaría en producción) en vez de integrar sobre todos los umbrales de score.
    Siempre con `iou_threshold` e IoU plano (nunca `eiou`: no está acotado en [0,1]
    y no es comparable con la literatura).

    `ap_iou_type` es independiente del `iou_type` del `criterion`/`matcher`:
    ese controla el coste de asignación y la pérdida de entrenamiento, este
    solo decide qué overlap cuenta como TP al calcular AP/mAP/`ap_agnostic`.

    `image_id` identifica el clip, no el batch: cada muestra del loader necesita su
    propia clave, o las cajas normalizadas de clips distintos (que comparten el mismo
    [0,1]²) se comparan entre sí en `_average_precision` y contaminan el AP.
    """
    model.eval()
    ap_iou_fn = get_iou_fn(ap_iou_type)
    plain_iou_fn = get_iou_fn("iou")
    totals: Metrics = dict.fromkeys(LOSS_KEYS, 0.0)

    confusion = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    matched_total = matched_correct = 0
    iou_sum = 0.0
    matched_box_count = 0

    AGNOSTIC = -1  # clave adicional en los diccionarios de abajo: "cualquier clase"
    detections_by_class: dict[int, list[Detection]] = defaultdict(list)
    gt_boxes_by_class: dict[int, dict[int, list[Tensor]]] = defaultdict(lambda: defaultdict(list))
    n_gt_by_class: dict[int, int] = defaultdict(int)

    progress = tqdm(loader, desc=_epoch_desc("val", epoch, epochs), unit="batch", leave=False)
    sample_id = 0
    for batch in progress:
        images, targets = _to_device(batch, device)
        outputs = model(images)
        losses: dict[str, Tensor] = criterion(outputs, targets)
        for key in LOSS_KEYS:
            totals[key] += losses[f"loss_{key}"].item()

        indices = matcher(outputs, targets)
        pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        scores, labels = predict_scores(outputs)

        for b, (query_idx, target_idx) in enumerate(indices):
            image_id = sample_id + b
            target = targets[b]

            if len(query_idx):
                pred_classes = pred_logits[b, query_idx, :-1].argmax(-1)
                true_classes = target["labels"][target_idx]
                matched_correct += int((pred_classes == true_classes).sum())
                matched_total += len(query_idx)
                for t, p in zip(true_classes.tolist(), pred_classes.tolist(), strict=True):
                    confusion[t, p] += 1

                matched_pred_boxes = pred_boxes[b, query_idx]
                matched_target_boxes = target["boxes"][target_idx]
                matched_pred_xyxy = box_convert(matched_pred_boxes, "cxcywh", "xyxy")
                matched_target_xyxy = box_convert(matched_target_boxes, "cxcywh", "xyxy")

                iou_sum += float(torch.diag(plain_iou_fn(matched_pred_xyxy, matched_target_xyxy)).sum())
                matched_box_count += len(query_idx)

            for class_id, box in zip(target["labels"].tolist(), target["boxes"].cpu(), strict=True):
                n_gt_by_class[class_id] += 1
                gt_boxes_by_class[class_id][image_id].append(box)
                n_gt_by_class[AGNOSTIC] += 1
                gt_boxes_by_class[AGNOSTIC][image_id].append(box)

            for q, (score, class_id) in enumerate(
                zip(scores[b].tolist(), labels[b].tolist(), strict=True)
            ):
                detections_by_class[class_id].append((score, image_id, pred_boxes[b, q].cpu()))
                detections_by_class[AGNOSTIC].append((score, image_id, pred_boxes[b, q].cpu()))

        sample_id += len(indices)

    ap_per_class = {
        class_id: _average_precision(
            detections_by_class.get(class_id, []),
            {img: torch.stack(boxes) for img, boxes in gt_boxes_by_class.get(class_id, {}).items()},
            n_gt_by_class.get(class_id, 0),
            ap_iou_fn,
            iou_threshold,
        )
        for class_id in range(n_classes)
    }
    agnostic_gt = {img: torch.stack(boxes) for img, boxes in gt_boxes_by_class.get(AGNOSTIC, {}).items()}
    ap_agnostic = {
        threshold: _average_precision(
            detections_by_class.get(AGNOSTIC, []),
            agnostic_gt,
            n_gt_by_class.get(AGNOSTIC, 0),
            ap_iou_fn,
            threshold,
        )
        for threshold in agnostic_thresholds
    }
    recall_agn, fp_per_tp = _recall_and_fp(
        detections_by_class.get(AGNOSTIC, []),
        agnostic_gt,
        n_gt_by_class.get(AGNOSTIC, 0),
        plain_iou_fn,
        iou_threshold,
        score_threshold,
    )
    valid_ap = [ap for ap in ap_per_class.values() if ap is not None]
    loss_totals = {key: value / max(len(loader), 1) for key, value in totals.items()}

    return EvalMetrics(
        total=loss_totals["total"],
        cls=loss_totals["cls"],
        bbox=loss_totals["bbox"],
        iou=loss_totals["iou"],
        cls_acc=matched_correct / max(matched_total, 1),
        mean_iou=iou_sum / max(matched_box_count, 1),
        recall_agn=recall_agn,
        fp_per_tp=fp_per_tp,
        map=sum(valid_ap) / max(len(valid_ap), 1),
        ap_iou_type=ap_iou_type,
        ap_per_class=ap_per_class,
        ap_agnostic=ap_agnostic,
        confusion=confusion,
    )
