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
DetectionRecord = tuple[float, int, Tensor]  # (score, image_id, box cxcywh), CPU tensors only

LOSS_KEYS: tuple[str, ...] = ("total", "cls", "bbox", "iou")
_IOU_FN = get_iou_fn("iou")


class Losses(NamedTuple):
    total: float
    cls: float
    bbox: float
    iou: float


class Framing(NamedTuple):
    mean_iou: float
    ap_agnostic: dict[float, float | None]


class Classification(NamedTuple):
    accuracy: float
    confusion: Tensor
    recall_per_class: dict[int, float | None]


class Detection(NamedTuple):
    recall: float
    fp_per_tp: float


class EvalMetrics(NamedTuple):
    losses: Losses
    framing: Framing
    classification: Classification
    detection: Detection


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


def _sorted_detections(detections: list[DetectionRecord]) -> list[DetectionRecord]:
    return sorted(detections, key=lambda d: d[0], reverse=True)


def _greedy_hits(
    detections: list[DetectionRecord],
    gt_by_image: dict[int, Tensor],
    iou_fn: Callable[[Tensor, Tensor], Tensor],
    iou_threshold: float,
) -> Tensor:
    hits = torch.zeros(len(detections))
    if not detections:
        return hits

    ranks_by_image: dict[int, list[int]] = defaultdict(list)
    for rank, (_, image_id, _) in enumerate(detections):
        ranks_by_image[image_id].append(rank)

    for image_id, ranks in ranks_by_image.items():
        gt_boxes = gt_by_image.get(image_id)
        if gt_boxes is None:
            continue
        boxes = torch.stack([detections[rank][2] for rank in ranks])
        overlaps = iou_fn(
            box_convert(boxes, "cxcywh", "xyxy"), box_convert(gt_boxes, "cxcywh", "xyxy")
        )  # (len(ranks), n_gt), filas ya en orden de score porque `ranks` lo está

        unused = torch.ones(len(gt_boxes), dtype=torch.bool)
        for row, rank in enumerate(ranks):
            row_overlaps = overlaps[row].masked_fill(~unused, -1.0)
            best = int(row_overlaps.argmax())
            if row_overlaps[best] >= iou_threshold:
                unused[best] = False
                hits[rank] = 1.0

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


def _class_recall(
    detections: list[DetectionRecord],
    gt_by_image: dict[int, Tensor],
    n_gt: int,
    iou_fn: Callable[[Tensor, Tensor], Tensor],
    iou_threshold: float,
    score_threshold: float,
) -> float | None:
    if n_gt == 0:
        return None
    sorted_detections = _sorted_detections(detections)
    hits = _greedy_hits(sorted_detections, gt_by_image, iou_fn, iou_threshold)
    k = sum(1 for d in sorted_detections if d[0] >= score_threshold)
    recall, _ = _recall_and_fp(hits, n_gt, k)
    return recall


def _recall_and_fp(hits: Tensor, n_gt: int, k: int) -> tuple[float, float]:
    if n_gt == 0:
        return 0.0, 0.0
    tp = int(hits[:k].sum())
    fp = k - tp
    return tp / n_gt, fp / max(tp, 1)


class LossAcumulator:
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes
        self.loss_totals: Metrics = dict.fromkeys(LOSS_KEYS, 0.0)
        self.n_batches = 0

        self.confusion = torch.zeros(n_classes, n_classes, dtype=torch.int64)
        self.matched_total = 0
        self.matched_correct = 0
        self.iou_sum = 0.0
        self.matched_box_count = 0

        self.detections_by_class: dict[int, list[DetectionRecord]] = defaultdict(list)
        self.gt_boxes_by_class: dict[int, dict[int, list[Tensor]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.n_gt_by_class: dict[int, int] = defaultdict(int)
        self._sample_id = 0

    def update(
        self,
        outputs: dict[str, Tensor],
        targets: list[Target],
        indices: list[tuple[Tensor, Tensor]],
        losses: dict[str, Tensor],
    ) -> None:
        for key in LOSS_KEYS:
            self.loss_totals[key] += losses[f"loss_{key}"].item()
        self.n_batches += 1

        pred_boxes = outputs["pred_boxes"]
        scores, labels = predict_scores(outputs)

        for b, (query_idx, target_idx) in enumerate(indices):
            image_id = self._sample_id + b  # identifica el clip, no el batch
            target = targets[b]

            if len(query_idx):
                pred_classes = labels[b, query_idx]
                true_classes = target["labels"][target_idx]
                self.matched_correct += int((pred_classes == true_classes).sum())
                self.matched_total += len(query_idx)
                for t, p in zip(true_classes.tolist(), pred_classes.tolist(), strict=True):
                    self.confusion[t, p] += 1

                matched_pred_xyxy = box_convert(pred_boxes[b, query_idx], "cxcywh", "xyxy")
                matched_target_xyxy = box_convert(target["boxes"][target_idx], "cxcywh", "xyxy")
                self.iou_sum += float(
                    torch.diag(_IOU_FN(matched_pred_xyxy, matched_target_xyxy)).sum()
                )
                self.matched_box_count += len(query_idx)

            for class_id, box in zip(target["labels"].tolist(), target["boxes"].cpu(), strict=True):
                self.n_gt_by_class[class_id] += 1
                self.gt_boxes_by_class[class_id][image_id].append(box)

            for q, (score, class_id) in enumerate(
                zip(scores[b].tolist(), labels[b].tolist(), strict=True)
            ):
                self.detections_by_class[class_id].append((score, image_id, pred_boxes[b, q].cpu()))

        self._sample_id += len(indices)

    def compute(
        self,
        iou_threshold: float,
        agnostic_thresholds: tuple[float, ...],
        score_threshold: float,
    ) -> EvalMetrics:
        losses = Losses(
            **{key: value / max(self.n_batches, 1) for key, value in self.loss_totals.items()}
        )

        recall_per_class = {
            class_id: _class_recall(
                self.detections_by_class.get(class_id, []),
                {
                    img: torch.stack(boxes)
                    for img, boxes in self.gt_boxes_by_class.get(class_id, {}).items()
                },
                self.n_gt_by_class.get(class_id, 0),
                _IOU_FN,
                iou_threshold,
                score_threshold,
            )
            for class_id in range(self.n_classes)
        }

        # detecciones/GT agnósticos de clase: se derivan al final de los buckets por
        # clase en vez de duplicarse en memoria durante la acumulación (una detección
        # por query, no dos).
        agnostic_detections = _sorted_detections(
            [d for dets in self.detections_by_class.values() for d in dets]
        )
        agnostic_gt_lists: dict[int, list[Tensor]] = defaultdict(list)
        for per_image in self.gt_boxes_by_class.values():
            for image_id, boxes in per_image.items():
                agnostic_gt_lists[image_id].extend(boxes)
        agnostic_gt = {img: torch.stack(boxes) for img, boxes in agnostic_gt_lists.items()}
        n_gt_agnostic = sum(self.n_gt_by_class.values())

        ap_agnostic = {
            threshold: _average_precision(
                _greedy_hits(agnostic_detections, agnostic_gt, _IOU_FN, threshold), n_gt_agnostic
            )
            for threshold in agnostic_thresholds
        }

        hits = _greedy_hits(agnostic_detections, agnostic_gt, _IOU_FN, iou_threshold)
        k = sum(1 for d in agnostic_detections if d[0] >= score_threshold)
        recall, fp_per_tp = _recall_and_fp(hits, n_gt_agnostic, k)

        return EvalMetrics(
            losses=losses,
            framing=Framing(
                mean_iou=self.iou_sum / max(self.matched_box_count, 1),
                ap_agnostic=ap_agnostic,
            ),
            classification=Classification(
                accuracy=self.matched_correct / max(self.matched_total, 1),
                confusion=self.confusion,
                recall_per_class=recall_per_class,
            ),
            detection=Detection(recall=recall, fp_per_tp=fp_per_tp),
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    matcher: nn.Module,
    device: torch.device | str = "cpu",
    n_classes: int = 1,
    iou_threshold: float = 0.5,
    agnostic_thresholds: tuple[float, ...] = (0.25, 0.3, 0.5, 0.75),
    score_threshold: float = 0.5,
    epoch: int | None = None,
    epochs: int | None = None,
) -> EvalMetrics:
    model.eval()
    accumulator = LossAcumulator(n_classes)

    progress = tqdm(loader, desc=_epoch_desc("val", epoch, epochs), unit="batch", leave=False)
    for batch in progress:
        images, targets = _to_device(batch, device)
        outputs = model(images)
        losses: dict[str, Tensor] = criterion(outputs, targets)
        indices = matcher(outputs, targets)
        accumulator.update(outputs, targets, indices, losses)

    return accumulator.compute(iou_threshold, agnostic_thresholds, score_threshold)
