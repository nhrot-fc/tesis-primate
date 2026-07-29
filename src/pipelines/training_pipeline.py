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
    mean_iomin: float  # IoMin promedio, mismas queries emparejadas (fijo, no depende de `ap_iou_type`)
    map: float  # mean AP sobre todos los umbrales de score, bajo `ap_iou_type`
    ap_iou_type: str  # tipo de IoU usado para `map`/`ap_per_class` (p.ej. "iomin", "eiou")
    ap_per_class: dict[int, float | None]
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    matcher: nn.Module,
    device: torch.device | str = "cpu",
    n_classes: int = 1,
    iou_threshold: float = 0.5,
    ap_iou_type: str = "iomin",
    epoch: int | None = None,
    epochs: int | None = None,
) -> EvalMetrics:
    """Pérdidas + métricas de detección.

    `cls_acc`/`mean_iomin`/`confusion` se calculan sobre las queries que el
    `HungarianMatcher` empareja con un ground truth, sin depender de ningún
    umbral de score. `ap_per_class`/`map` sí rankean por score, pero integran
    sobre todos los umbrales en vez de fijar uno.

    `ap_iou_type` es independiente del `iou_type` del `criterion`/`matcher`:
    ese controla el coste de asignación y la pérdida de entrenamiento, este
    solo decide qué overlap cuenta como TP al calcular AP/mAP.
    """
    model.eval()
    ap_iou_fn = get_iou_fn(ap_iou_type)
    iomin_fn = get_iou_fn("iomin")
    totals: Metrics = dict.fromkeys(LOSS_KEYS, 0.0)

    confusion = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    matched_total = matched_correct = 0
    iomin_sum, iomin_count = 0.0, 0

    detections_by_class: dict[int, list[Detection]] = defaultdict(list)
    gt_boxes_by_class: dict[int, dict[int, list[Tensor]]] = defaultdict(lambda: defaultdict(list))
    n_gt_by_class: dict[int, int] = defaultdict(int)

    progress = tqdm(loader, desc=_epoch_desc("val", epoch, epochs), unit="batch", leave=False)
    for image_id, batch in enumerate(progress):
        images, targets = _to_device(batch, device)
        outputs = model(images)
        losses: dict[str, Tensor] = criterion(outputs, targets)
        for key in LOSS_KEYS:
            totals[key] += losses[f"loss_{key}"].item()

        indices = matcher(outputs, targets)
        pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
        scores, labels = predict_scores(outputs)

        for b, (query_idx, target_idx) in enumerate(indices):
            target = targets[b]

            if len(query_idx):
                pred_classes = pred_logits[b, query_idx, :-1].argmax(-1)
                true_classes = target["labels"][target_idx]
                matched_correct += int((pred_classes == true_classes).sum())
                matched_total += len(query_idx)
                for t, p in zip(true_classes.tolist(), pred_classes.tolist(), strict=True):
                    confusion[t, p] += 1

                overlaps = iomin_fn(
                    box_convert(pred_boxes[b, query_idx], "cxcywh", "xyxy"),
                    box_convert(target["boxes"][target_idx], "cxcywh", "xyxy"),
                )
                iomin_sum += float(torch.diag(overlaps).sum())
                iomin_count += len(query_idx)

            for class_id, box in zip(target["labels"].tolist(), target["boxes"].cpu(), strict=True):
                n_gt_by_class[class_id] += 1
                gt_boxes_by_class[class_id][image_id].append(box)

            for q, (score, class_id) in enumerate(
                zip(scores[b].tolist(), labels[b].tolist(), strict=True)
            ):
                detections_by_class[class_id].append((score, image_id, pred_boxes[b, q].cpu()))

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
    valid_ap = [ap for ap in ap_per_class.values() if ap is not None]
    loss_totals = {key: value / max(len(loader), 1) for key, value in totals.items()}

    return EvalMetrics(
        total=loss_totals["total"],
        cls=loss_totals["cls"],
        bbox=loss_totals["bbox"],
        iou=loss_totals["iou"],
        cls_acc=matched_correct / max(matched_total, 1),
        mean_iomin=iomin_sum / max(iomin_count, 1),
        map=sum(valid_ap) / max(len(valid_ap), 1),
        ap_iou_type=ap_iou_type,
        ap_per_class=ap_per_class,
        confusion=confusion,
    )
