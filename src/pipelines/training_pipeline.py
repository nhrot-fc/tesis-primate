from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

from architectures.deformable_detr import Detections, postprocess
from architectures.iou import get_iou_fn

Target = dict[str, Tensor]
Batch = tuple[Tensor, list[Target]]
Metrics = dict[str, float]

LOSS_KEYS: tuple[str, ...] = ("total", "cls", "bbox", "iou")
EPS = 1e-9


def _to_device(batch: Batch, device: torch.device | str) -> Batch:
    images, targets = batch
    return (
        images.to(device),
        [{key: value.to(device) for key, value in target.items()} for target in targets],
    )


def detection_count(
    detections: Detections,
    target: Target,
    iou_threshold: float,
    iou_fn: Callable[[Tensor, Tensor], Tensor],
) -> tuple[int, int]:
    if len(detections.boxes) == 0 or len(target["boxes"]) == 0:
        return 0, 0

    overlaps = iou_fn(
        box_convert(detections.boxes, "cxcywh", "xyxy"),
        box_convert(target["boxes"], "cxcywh", "xyxy"),
    )
    matched = torch.zeros(len(target["boxes"]), dtype=torch.bool, device=overlaps.device)
    true_positives = 0
    correct_class = 0

    for prediction in range(len(detections.boxes)):
        candidates = overlaps[prediction].masked_fill(matched, float("-inf"))
        best = int(candidates.argmax())
        if candidates[best] < iou_threshold:
            continue
        matched[best] = True
        true_positives += 1
        correct_class += int(detections.labels[prediction] == target["labels"][best])

    return true_positives, correct_class


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    device: torch.device | str = "cpu",
    clip_grad: float = 0.1,
) -> Metrics:
    model.train()
    totals: Metrics = dict.fromkeys(LOSS_KEYS, 0.0)

    for batch in loader:
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

    return {key: value / max(len(loader), 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    device: torch.device | str = "cpu",
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    iou_type: str = "iou",
) -> Metrics:
    model.eval()
    iou_fn = get_iou_fn(iou_type)
    totals: Metrics = dict.fromkeys(LOSS_KEYS, 0.0)
    true_positives = correct_class = num_targets = num_predictions = 0

    for batch in loader:
        images, targets = _to_device(batch, device)
        outputs = model(images)
        losses: dict[str, Tensor] = criterion(outputs, targets)
        for key in LOSS_KEYS:
            totals[key] += losses[f"loss_{key}"].item()

        for detections, target in zip(postprocess(outputs, score_threshold), targets, strict=True):
            matches, correct = detection_count(detections, target, iou_threshold, iou_fn)
            true_positives += matches
            correct_class += correct
            num_predictions += len(detections.boxes)
            num_targets += len(target["boxes"])

    precision = true_positives / max(num_predictions, 1)
    recall = true_positives / max(num_targets, 1)

    metrics = {key: value / max(len(loader), 1) for key, value in totals.items()}
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = 2 * precision * recall / max(precision + recall, EPS)
    metrics["cls_acc"] = correct_class / max(true_positives, 1)
    return metrics
