import json
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor

from core.config import MAX_DETECTIONS, SCORE_FLOOR, SCORE_STEP, SEED, score_grid
from evaluation.evaluator import RawPredictions
from evaluation.metrics import (
    MATCH_IOU,
    Boxes,
    PerClassAP,
    average_precision_per_class,
    hits,
    mean_average_precision,
    overlaps,
    window_classes,
)

THRESHOLD_GRID = score_grid(SCORE_STEP, 1.0 - SCORE_STEP)
# Precisión mínima en val de cada punto de operación
MIN_PRECISIONS = (0.70, 0.50)
N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
# El primer punto de operación de cada modelo, junto a su `best.pt`: es el umbral con el que
# arranca el visor. Lo escribe `compare_models.py` y lo lee `models.registry.load_checkpoint`.
OPERATING_POINT = "operating_point.json"


class Point(NamedTuple):
    threshold: float
    recall: float | None
    precision: float | None
    n_above: int
    n_tp: int


# Umbral elegido en val, medido en test
class Paired(NamedTuple):
    min_precision: float
    threshold: float | None
    val: Point | None
    test: Point | None
    interval: dict[str, list[float]]


class ClassRow(NamedTuple):
    name: str
    n_gt: int
    recall: float | None
    ap: float | None


class WindowLevel(NamedTuple):
    name: str
    windows_gt: int
    windows_predicted: int
    recall: float | None
    precision: float | None


class ModelComparison(NamedTuple):
    model: str
    map_30: float | None
    paired: list[Paired]
    per_class: list[ClassRow]
    window_level: list[WindowLevel]


class Protocol(NamedTuple):
    iou: float
    max_det: int
    score_floor: float
    n_bootstrap: int
    confidence: float
    seed: int


class Split(NamedTuple):
    windows: int
    boxes: int
    recordings: int


class Comparison(NamedTuple):
    protocol: Protocol
    val: Split
    test: Split
    detection_classes: list[str]
    window_classes: list[str]
    models: list[ModelComparison]


def as_json(value: Any) -> Any:
    if hasattr(value, "_asdict"):
        return {key: as_json(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {str(key): as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    return value


def equalize(
    predictions: Boxes, max_det: int = MAX_DETECTIONS, score_floor: float = SCORE_FLOOR
) -> Boxes:
    # Mismo presupuesto para todos: las `max_det` mejores de cada ventana, en orden de score.
    kept = predictions.select(predictions.scores >= score_floor)
    if not len(kept.boxes):
        return kept
    image_ids = kept.image_ids.to(torch.int64)
    by_image = image_ids.argsort(stable=True)
    per_image = torch.bincount(image_ids[by_image])
    starts = per_image.cumsum(0) - per_image
    rank_in_image = torch.arange(len(by_image)) - starts.repeat_interleave(per_image)
    return kept.select(by_image[rank_in_image < max_det].sort().values)


# Predicciones en orden de score, con su acierto y su ventana
class Matched(NamedTuple):
    scores: Tensor
    found: Tensor
    image_ids: Tensor
    gt_per_image: Tensor

    @property
    def n_gt(self) -> int:
        return int(self.gt_per_image.sum())

    def above(self, threshold: float) -> int:
        return int((self.scores >= threshold).sum())


def match(predictions: Boxes, truth: Boxes, n_images: int) -> Matched:
    return Matched(
        scores=predictions.scores,
        found=hits(
            overlaps(predictions, truth, class_aware=True), len(predictions.boxes), MATCH_IOU
        ),
        image_ids=predictions.image_ids.to(torch.int64),
        gt_per_image=torch.bincount(truth.image_ids.to(torch.int64), minlength=n_images),
    )


def operating_point(matched: Matched, threshold: float) -> Point:
    n_above = matched.above(threshold)
    n_tp = int(matched.found[:n_above].sum())
    return Point(
        threshold=threshold,
        recall=n_tp / matched.n_gt if matched.n_gt else None,
        precision=n_tp / n_above if n_above else None,
        n_above=n_above,
        n_tp=n_tp,
    )


def select(matched: Matched, min_precision: float) -> Point | None:
    points = [operating_point(matched, threshold) for threshold in THRESHOLD_GRID]
    feasible = [p for p in points if p.precision is not None and p.precision >= min_precision]
    if not feasible:
        return None
    return max(feasible, key=lambda p: (p.recall or 0.0, p.precision or 0.0))


def bootstrap(
    matched: Matched, recordings: Tensor, threshold: float, n_boot: int
) -> dict[str, list[float]]:
    # IC por remuestreo de grabaciones: las ventanas de una misma grabación no son independientes.
    n_images = len(matched.gt_per_image)
    n_above = matched.above(threshold)
    above = matched.image_ids[:n_above]
    tp_per_window = torch.bincount(above, weights=matched.found[:n_above], minlength=n_images)
    predicted_per_window = torch.bincount(above, minlength=n_images)

    n_recordings = int(recordings.max()) + 1
    per_recording = torch.stack(
        [
            torch.bincount(recordings, weights=series.double(), minlength=n_recordings)
            for series in (tp_per_window, predicted_per_window, matched.gt_per_image)
        ]
    )
    generator = torch.Generator().manual_seed(SEED)
    draws = torch.randint(0, n_recordings, (n_boot, n_recordings), generator=generator)
    tp, predicted, n_gt = per_recording[:, draws].sum(dim=-1)

    samples = {"recall": tp / n_gt.clamp(min=1), "precision": tp / predicted.clamp(min=1)}
    low, high = (1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2
    return {name: [float(v.quantile(low)), float(v.quantile(high))] for name, v in samples.items()}


def window_level(
    predictions: Boxes, truth: Boxes, class_id: int, name: str, threshold: float
) -> WindowLevel:
    annotated = set(truth.image_ids[truth.labels == class_id].tolist())
    above = (predictions.scores >= threshold) & (predictions.labels == class_id)
    detected = set(predictions.image_ids[above].tolist())
    tp = len(annotated & detected)
    return WindowLevel(
        name=name,
        windows_gt=len(annotated),
        windows_predicted=len(detected),
        recall=tp / len(annotated) if annotated else None,
        precision=tp / len(detected) if detected else None,
    )


def class_rows(
    matched: Matched,
    predictions: Boxes,
    truth: Boxes,
    threshold: float,
    ap: PerClassAP,
    names: list[str],
    classes: list[int],
) -> list[ClassRow]:
    n_above = matched.above(threshold)
    predicted_labels = predictions.labels[:n_above]
    found = matched.found[:n_above]
    rows = []
    for class_id in classes:
        n_gt = int((truth.labels == class_id).sum())
        tp = float(found[predicted_labels == class_id].sum())
        rows.append(ClassRow(names[class_id], n_gt, tp / n_gt if n_gt else None, ap[class_id]))
    return rows


def write_operating_point(directory: Path, model: str, paired: Paired, protocol: Protocol) -> Path:
    path = directory / OPERATING_POINT
    record = {"model": model, **as_json(paired), "protocol": as_json(protocol)}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def read_operating_point(directory: Path) -> float | None:
    # -> umbral de score, o None si la corrida no pasó por `compare_models.py`
    path = directory / OPERATING_POINT
    if not path.is_file():
        return None
    threshold = json.loads(path.read_text()).get("threshold")
    return None if threshold is None else float(threshold)


def describe(dump: RawPredictions) -> Split:
    return Split(dump.n_images, len(dump.truth.boxes), len(dump.recording_names))


def check_comparable(models: list[tuple[RawPredictions, RawPredictions]]) -> None:
    reference_val, reference_test = models[0]
    for val, test in models:
        for dump, reference in ((val, reference_val), (test, reference_test)):
            if dump.labels != reference.labels or not torch.equal(
                dump.recordings, reference.recordings
            ):
                raise ValueError(
                    f"{dump.model} y {reference.model} no evaluaron el mismo {dump.split}"
                )


def compare(
    models: list[tuple[RawPredictions, RawPredictions]],
    min_precisions: tuple[float, ...] = MIN_PRECISIONS,
    max_det: int = MAX_DETECTIONS,
    n_boot: int = N_BOOTSTRAP,
) -> Comparison:
    check_comparable(models)
    reference_val, reference_test = models[0]
    names = reference_val.labels
    saturated = window_classes(reference_test.truth, len(names))
    detection = [c for c in range(len(names)) if c not in saturated]
    recordings = reference_test.recordings.to(torch.int64)

    compared: list[ModelComparison] = []
    for val_dump, test_dump in models:
        val = equalize(val_dump.predictions, max_det)
        test = equalize(test_dump.predictions, max_det)
        matched_val = match(val, val_dump.truth, val_dump.n_images)
        matched_test = match(test, test_dump.truth, test_dump.n_images)
        ap = average_precision_per_class(test, test_dump.truth, len(names))

        paired = []
        for min_precision in min_precisions:
            chosen = select(matched_val, min_precision)
            if chosen is None:
                paired.append(Paired(min_precision, None, None, None, {}))
                continue
            paired.append(
                Paired(
                    min_precision=min_precision,
                    threshold=chosen.threshold,
                    val=chosen,
                    test=operating_point(matched_test, chosen.threshold),
                    interval=bootstrap(matched_test, recordings, chosen.threshold, n_boot),
                )
            )

        # Los desgloses por clase van al primer punto de operación.
        threshold = paired[0].threshold
        per_class = window_rows = []
        if threshold is not None:
            per_class = class_rows(
                matched_test, test, test_dump.truth, threshold, ap, names, detection
            )
            window_rows = [
                window_level(test, test_dump.truth, c, names[c], threshold) for c in saturated
            ]

        compared.append(
            ModelComparison(
                model=test_dump.model,
                map_30=mean_average_precision(ap, detection),
                paired=paired,
                per_class=per_class,
                window_level=window_rows,
            )
        )

    return Comparison(
        protocol=Protocol(MATCH_IOU, max_det, SCORE_FLOOR, n_boot, CONFIDENCE, SEED),
        val=describe(reference_val),
        test=describe(reference_test),
        detection_classes=[names[c] for c in detection],
        window_classes=[names[c] for c in saturated],
        models=compared,
    )
