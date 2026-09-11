import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from core.config import SCORE_FLOOR, SCORE_THRESHOLD
from core.runtime import progress
from data.cache import TEST, VAL
from data.datasets import Batch, to_device
from evaluation.metrics import Boxes, DetectionMetrics, concat, detection_metrics, sort_by_score
from utils.boxes import Detections, postprocess

logger = logging.getLogger(__name__)

Detect = Callable[[Tensor, float], list[Detections]]
SUFFIX = "_predictions.pt"


# Volcado de un modelo sobre un split, hasta `SCORE_FLOOR`
class RawPredictions(NamedTuple):
    model: str
    split: str
    labels: list[str]
    predictions: Boxes
    truth: Boxes
    n_images: int
    recordings: Tensor  # grabación de cada ventana
    recording_names: list[str]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._asdict(), path)
        return path

    @classmethod
    def load(cls, path: Path) -> "RawPredictions":
        stored = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**{key: stored[key] for key in cls._fields})


@torch.no_grad()
def collect_detections(
    detect: Detect,
    loader: DataLoader,
    device: str | torch.device,
    nms_iou: float | None,
    max_detections: int | None = None,
    desc: str = "detectando",
) -> tuple[Boxes, Boxes, int]:
    predicted: list[Boxes] = []
    truth: list[Boxes] = []
    image_id = 0
    for images, targets in progress(loader, desc):
        detections = detect(images.to(device), SCORE_FLOOR)
        for detection, target in zip(detections, targets, strict=True):
            predicted.append(Boxes.of(postprocess(detection, nms_iou, max_detections), image_id))
            truth.append(Boxes.truth(target, image_id))
            image_id += 1
    return sort_by_score(concat(predicted)), concat(truth), image_id


def evaluate(
    detect: Detect,
    loader: DataLoader,
    n_classes: int,
    device: str | torch.device,
    nms_iou: float | None,
    score_threshold: float = SCORE_THRESHOLD,
    max_detections: int | None = None,
    desc: str = VAL,
) -> DetectionMetrics:
    predictions, truth, _ = collect_detections(
        detect, loader, device, nms_iou, max_detections, desc
    )
    return detection_metrics(predictions, truth, n_classes, score_threshold)


# Los términos de pérdida de un lote más su suma, que es lo que se optimiza y se loguea.
def loss_terms(model: nn.Module, batch: Batch, device: str | torch.device) -> dict[str, Tensor]:
    images, targets = to_device(batch, device)
    terms: dict[str, Tensor] = model(images, targets)
    return {"total": torch.stack(list(terms.values())).sum(), **terms}


def add_losses(totals: dict[str, float], losses: dict[str, Tensor]) -> None:
    for key, value in losses.items():
        totals[key] = totals.get(key, 0.0) + value.item()


def mean_losses(totals: dict[str, float], n_batches: int) -> dict[str, float]:
    n = max(n_batches, 1)
    return {key.removeprefix("loss_"): value / n for key, value in totals.items()}


@torch.no_grad()
def average_loss(
    model: nn.Module, loader: DataLoader, device: str | torch.device, desc: str = "loss"
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    try:
        for batch in progress(loader, desc):
            add_losses(totals, loss_terms(model, batch, device))
    finally:
        model.train(was_training)
    return mean_losses(totals, len(loader))


def path_for(directory: Path, model: str, split: str) -> Path:
    return directory / f"{model}_{split}{SUFFIX}"


# Los dos volcados de un modelo y dónde viven: val elige el umbral y test lo mide.
class ModelDumps(NamedTuple):
    model: str
    directory: Path  # la del volcado de val; ahí va `operating_point.json`
    val: RawPredictions
    test: RawPredictions


def load_dumps(paths: Sequence[Path]) -> list[ModelDumps]:
    by_model: dict[str, dict[str, RawPredictions]] = {}
    directories: dict[str, Path] = {}
    for path in paths:
        dump = RawPredictions.load(path)
        by_model.setdefault(dump.model, {})[dump.split] = dump
        if dump.split == VAL:
            directories[dump.model] = path.parent
        logger.info(
            "%s %s: %d ventanas, %d cajas",
            dump.model,
            dump.split,
            dump.n_images,
            len(dump.predictions.boxes),
        )
    incomplete = {m: sorted(s) for m, s in by_model.items() if not {VAL, TEST} <= s.keys()}
    if incomplete:
        raise ValueError(f"faltan volcados: {incomplete}; cada modelo necesita {VAL} y {TEST}")
    return [
        ModelDumps(model, directories[model], splits[VAL], splits[TEST])
        for model, splits in by_model.items()
    ]
