import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from tqdm.auto import tqdm

from evaluation.metrics import (
    BETA,
    SCORE_FLOOR,
    Boxes,
    DetectionMetrics,
    concat,
    detection_metrics,
    sort_by_score,
)
from utils.boxes import Detections, suppress_nested

logger = logging.getLogger(__name__)

BoundDetect = Callable[[Tensor, float], list[Detections]]
SUFFIX = "_predictions.pt"
VAL, TEST = "val", "test"


class RawPredictions(NamedTuple):
    model: str
    architecture: str
    split: str
    labels: list[str]
    predictions: Boxes
    truth: Boxes
    n_images: int
    recordings: Tensor
    recording_names: list[str]
    nms_iou: float | None
    score_floor: float = SCORE_FLOOR

    @property
    def detections_per_window(self) -> float:
        return len(self.predictions.boxes) / max(self.n_images, 1)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._asdict(), path)
        return path

    @classmethod
    def load(cls, path: Path) -> "RawPredictions":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


@torch.no_grad()
def collect_detections(
    detect: BoundDetect,
    loader: DataLoader,
    device: str | torch.device,
    nms_iou: float | None = 0.3,
    max_detections: int | None = None,
    desc: str = "detectando",
) -> tuple[Boxes, Boxes, int]:
    predicted: list[Boxes] = []
    truth: list[Boxes] = []
    image_id = 0

    for images, targets in tqdm(loader, desc=desc, unit="batch", leave=False):
        detections = detect(images.to(device), SCORE_FLOOR)
        for detection, target in zip(detections, targets, strict=True):
            boxes, scores, labels = (tensor.cpu() for tensor in detection)
            if nms_iou is not None and len(boxes):
                keep = suppress_nested(
                    box_convert(boxes, "cxcywh", "xyxy"), scores, labels, nms_iou
                )
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if max_detections is not None and len(boxes) > max_detections:
                keep = scores.topk(max_detections).indices
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            predicted.append(Boxes(boxes, torch.full((len(boxes),), image_id), labels, scores))
            n_truth = len(target["labels"])
            truth.append(
                Boxes(
                    target["boxes"].cpu(),
                    torch.full((n_truth,), image_id),
                    target["labels"].cpu(),
                    torch.ones(n_truth),
                )
            )
            image_id += 1

    return sort_by_score(concat(predicted)), concat(truth), image_id


def evaluate(
    detect: BoundDetect,
    loader: DataLoader,
    n_classes: int,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.5,
    nms_iou: float | None = 0.3,
    max_detections: int | None = None,
    beta: float = BETA,
    desc: str = "val",
) -> DetectionMetrics:
    predictions, truth, n_images = collect_detections(
        detect, loader, device, nms_iou, max_detections, desc
    )
    return detection_metrics(
        predictions,
        truth,
        n_classes=n_classes,
        n_images=n_images,
        score_threshold=score_threshold,
        beta=beta,
    )


def path_for(directory: Path, model: str, split: str) -> Path:
    return directory / f"{model}_{split}{SUFFIX}"


def load_pairs(paths: Sequence[Path]) -> list[tuple[RawPredictions, RawPredictions]]:
    by_model: dict[str, dict[str, RawPredictions]] = {}
    for path in paths:
        dump = RawPredictions.load(path)
        by_model.setdefault(dump.model, {})[dump.split] = dump
        logger.info(
            "%s %s: %d ventanas, %.1f detecciones por ventana, %d grabaciones",
            dump.model,
            dump.split,
            dump.n_images,
            dump.detections_per_window,
            len(dump.recording_names),
        )

    incomplete = {
        model: sorted(splits)
        for model, splits in by_model.items()
        if not {VAL, TEST} <= splits.keys()
    }
    if incomplete:
        raise ValueError(
            f"faltan volcados: {incomplete}. Cada modelo necesita {VAL} (elige el umbral) y "
            f"{TEST} (lo mide)."
        )
    return [(splits[VAL], splits[TEST]) for splits in by_model.values()]
