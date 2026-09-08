from typing import Any, Literal, NamedTuple

import torch
from torch import Tensor

from core.config import SEED, P
from evaluation.decomposition import Decomposition, decompose
from evaluation.evaluator import RawPredictions
from evaluation.metrics import (
    BETA,
    COCO_THRESHOLDS,
    MATCH_IOU,
    SCORE_FLOOR,
    Boxes,
    PerClassAP,
    average_precision_per_class,
    f_beta,
    false_positives_per_hour,
    hits,
    mean_average_precision,
    mean_average_precision_over,
    overlaps,
)

EQUALIZED_MAX_DET = 64
THRESHOLD_GRID = torch.arange(0.01, 1.0, 0.01)
N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
TARGET_PRECISION = 0.70
WINDOW_CLASS_WIDTH = 0.99

CriterionKind = Literal["precision", "fp_per_hour", "f_beta"]


class Point(NamedTuple):
    threshold: float
    recall: float | None
    precision: float | None
    f_beta: float | None
    fp_per_hour: float | None
    boxes_per_tp: float | None
    n_above: int
    n_tp: int


class Criterion(NamedTuple):
    kind: CriterionKind
    value: float

    def __str__(self) -> str:
        return {
            "precision": f"precision>={self.value:.2f}",
            "fp_per_hour": f"FP/h<={self.value:g}",
            "f_beta": f"max F{self.value:g}",
        }[self.kind]

    def satisfied_by(self, point: Point) -> bool:
        if self.kind == "precision":
            return point.precision is not None and point.precision >= self.value
        if self.kind == "fp_per_hour":
            return point.fp_per_hour is not None and point.fp_per_hour <= self.value
        return True


DEFAULT_CRITERIA = (
    Criterion("precision", TARGET_PRECISION),
    Criterion("precision", 0.50),
    Criterion("fp_per_hour", 100.0),
    Criterion("f_beta", BETA),
)


class Paired(NamedTuple):
    criterion: str
    threshold: float | None
    val: Point | None
    test: Point | None
    agnostic_recall: float | None
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
    architecture: str
    detections_per_window: float
    map_30: float | None
    map_50: float | None
    map_50_95: float | None
    map_all: float | None
    max_recall: float | None
    paired: list[Paired]
    decomposition: Decomposition
    per_class: list[ClassRow]
    window_level: list[WindowLevel]


class Protocol(NamedTuple):
    iou: float
    max_det: int
    score_floor: float
    beta: float
    n_bootstrap: int
    confidence: float
    reference_criterion: str
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
    predictions: Boxes, max_det: int = EQUALIZED_MAX_DET, score_floor: float = SCORE_FLOOR
) -> Boxes:
    # Deja las `max_det` mejores de cada ventana sin ordenar por ventana: `kept` ya viene en
    # orden de score, así que la posición dentro de su grupo es el ranking, y el `.sort()`
    # final devuelve las filas al orden de score global que el resto del módulo asume.
    kept = predictions.select(predictions.scores >= score_floor)
    if not len(kept.boxes):
        return kept

    image_ids = kept.image_ids.to(torch.int64)
    by_image = image_ids.argsort(stable=True)
    per_image = torch.bincount(image_ids[by_image])
    starts = per_image.cumsum(0) - per_image
    rank_in_image = torch.arange(len(by_image)) - starts.repeat_interleave(per_image)
    rows_in_score_order = by_image[rank_in_image < max_det].sort().values
    return kept.select(rows_in_score_order)


class Matched(NamedTuple):
    scores: Tensor
    found: Tensor
    image_ids: Tensor
    gt_per_image: Tensor
    n_images: int

    @property
    def n_gt(self) -> int:
        return int(self.gt_per_image.sum())

    def above(self, threshold: float) -> int:
        return int((self.scores >= threshold).sum())


def match(predictions: Boxes, truth: Boxes, n_images: int, class_aware: bool = True) -> Matched:
    return Matched(
        scores=predictions.scores,
        found=hits(overlaps(predictions, truth, class_aware), len(predictions.boxes), MATCH_IOU),
        image_ids=predictions.image_ids.to(torch.int64),
        gt_per_image=torch.bincount(truth.image_ids.to(torch.int64), minlength=n_images),
        n_images=n_images,
    )


def operating_point(matched: Matched, threshold: float, beta: float = BETA) -> Point:
    n_above = matched.above(threshold)
    n_tp = int(matched.found[:n_above].sum())
    recall = n_tp / matched.n_gt if matched.n_gt else None
    precision = n_tp / n_above if n_above else None
    return Point(
        threshold=threshold,
        recall=recall,
        precision=precision,
        f_beta=f_beta(precision, recall, beta),
        fp_per_hour=false_positives_per_hour(n_above - n_tp, matched.n_images),
        boxes_per_tp=n_above / n_tp if n_tp else None,
        n_above=n_above,
        n_tp=n_tp,
    )


def select(matched: Matched, criterion: Criterion, beta: float = BETA) -> Point | None:
    points = [operating_point(matched, round(float(t), 2), beta) for t in THRESHOLD_GRID]
    if criterion.kind == "f_beta":
        return max(points, key=lambda point: point.f_beta or -1.0)
    feasible = [point for point in points if criterion.satisfied_by(point)]
    if not feasible:
        return None
    return max(feasible, key=lambda point: (point.recall or 0.0, point.precision or 0.0))


def bootstrap(
    matched: Matched,
    recordings: Tensor,
    threshold: float,
    beta: float = BETA,
    n_boot: int = N_BOOTSTRAP,
) -> dict[str, list[float]]:
    n_above = matched.above(threshold)
    above = matched.image_ids[:n_above]
    tp_per_window = torch.bincount(
        above, weights=matched.found[:n_above], minlength=matched.n_images
    )
    predicted_per_window = torch.bincount(above, minlength=matched.n_images)
    ones_per_window = torch.ones(matched.n_images)

    n_recordings = int(recordings.max()) + 1
    per_recording = torch.stack(
        [
            torch.bincount(recordings, weights=series.double(), minlength=n_recordings)
            for series in (
                tp_per_window,
                predicted_per_window,
                matched.gt_per_image,
                ones_per_window,
            )
        ]
    )

    generator = torch.Generator().manual_seed(SEED)
    draws = torch.randint(0, n_recordings, (n_boot, n_recordings), generator=generator)
    tp, predicted, n_gt, windows = per_recording[:, draws].sum(dim=-1)

    recall = tp / n_gt.clamp(min=1)
    precision = tp / predicted.clamp(min=1)
    samples = {
        "recall": recall,
        "precision": precision,
        "f_beta": (1 + beta**2)
        * precision
        * recall
        / (beta**2 * precision + recall).clamp(min=1e-9),
        "fp_per_hour": (predicted - tp) / (windows * P.clip_len_s / 3600).clamp(min=1e-9),
    }
    low, high = (1 - CONFIDENCE) / 2, (1 + CONFIDENCE) / 2
    return {
        name: [float(value.quantile(low)), float(value.quantile(high))]
        for name, value in samples.items()
    }


def max_recall_at_precision(matched: Matched, target: float = TARGET_PRECISION) -> float | None:
    if not len(matched.found) or not matched.n_gt:
        return None
    tp = matched.found.cumsum(0)
    feasible = tp / torch.arange(1, len(tp) + 1) >= target
    return float((tp[feasible] / matched.n_gt).max()) if feasible.any() else 0.0


def saturated_classes(truth: Boxes, n_classes: int) -> list[int]:
    widths = [truth.boxes[truth.labels == class_id][:, 2] for class_id in range(n_classes)]
    return [
        class_id
        for class_id, width in enumerate(widths)
        if len(width) and float(width.median()) >= WINDOW_CLASS_WIDTH
    ]


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
    class_names: list[str],
    classes: list[int],
) -> list[ClassRow]:
    n_above = matched.above(threshold)
    predicted_labels = predictions.labels[:n_above]
    found = matched.found[:n_above]
    rows = []
    for class_id in classes:
        n_gt = int((truth.labels == class_id).sum())
        tp = float(found[predicted_labels == class_id].sum())
        rows.append(
            ClassRow(
                name=class_names[class_id],
                n_gt=n_gt,
                recall=tp / n_gt if n_gt else None,
                ap=ap[class_id],
            )
        )
    return rows


def describe(dump: RawPredictions) -> Split:
    return Split(dump.n_images, len(dump.truth.boxes), len(dump.recording_names))


def check_comparable(
    models: list[tuple[RawPredictions, RawPredictions]],
    reference_val: RawPredictions,
    reference_test: RawPredictions,
) -> None:
    for val, test in models:
        for dump, reference in ((val, reference_val), (test, reference_test)):
            if dump.labels != reference.labels or not torch.equal(
                dump.recordings, reference.recordings
            ):
                raise ValueError(
                    f"{dump.model} y {reference.model} no evaluaron el mismo {dump.split}: "
                    f"{dump.n_images} ventanas y {len(dump.labels)} clases contra "
                    f"{reference.n_images} y {len(reference.labels)}."
                )


def paired_points(
    matched_val: Matched,
    matched_test: Matched,
    agnostic: Matched,
    recordings: Tensor,
    criteria: tuple[Criterion, ...],
    beta: float,
    n_boot: int,
) -> list[Paired]:
    paired = []
    for criterion in criteria:
        chosen = select(matched_val, criterion, beta)
        if chosen is None:
            paired.append(Paired(str(criterion), None, None, None, None, {}))
            continue
        paired.append(
            Paired(
                criterion=str(criterion),
                threshold=chosen.threshold,
                val=chosen,
                test=operating_point(matched_test, chosen.threshold, beta),
                agnostic_recall=operating_point(agnostic, chosen.threshold, beta).recall,
                interval=bootstrap(matched_test, recordings, chosen.threshold, beta, n_boot),
            )
        )
    return paired


def compare(
    models: list[tuple[RawPredictions, RawPredictions]],
    criteria: tuple[Criterion, ...] = DEFAULT_CRITERIA,
    max_det: int = EQUALIZED_MAX_DET,
    beta: float = BETA,
    n_boot: int = N_BOOTSTRAP,
) -> Comparison:
    reference_val, reference_test = models[0]
    check_comparable(models, reference_val, reference_test)

    class_names = reference_val.labels
    saturated = saturated_classes(reference_test.truth, len(class_names))
    detection = [c for c in range(len(class_names)) if c not in saturated]
    recordings = reference_test.recordings.to(torch.int64)
    reference_criterion = next((c for c in criteria if c.kind == "f_beta"), criteria[0])

    compared: list[ModelComparison] = []
    for val_dump, test_dump in models:
        val = equalize(val_dump.predictions, max_det)
        test = equalize(test_dump.predictions, max_det)
        matched_val = match(val, val_dump.truth, val_dump.n_images)
        matched_test = match(test, test_dump.truth, test_dump.n_images)
        agnostic = match(test, test_dump.truth, test_dump.n_images, class_aware=False)

        paired = paired_points(
            matched_val, matched_test, agnostic, recordings, criteria, beta, n_boot
        )
        reference_threshold = next(
            (p.threshold for p in paired if p.criterion == str(reference_criterion)), None
        )
        ap = average_precision_per_class(test, test_dump.truth, len(class_names))

        if reference_threshold is None:
            per_class: list[ClassRow] = []
            window_rows: list[WindowLevel] = []
        else:
            per_class = class_rows(
                matched_test,
                test,
                test_dump.truth,
                reference_threshold,
                ap[MATCH_IOU],
                class_names,
                detection,
            )
            window_rows = [
                window_level(
                    test, test_dump.truth, class_id, class_names[class_id], reference_threshold
                )
                for class_id in saturated
            ]

        compared.append(
            ModelComparison(
                model=test_dump.model,
                architecture=test_dump.architecture,
                detections_per_window=test_dump.detections_per_window,
                map_30=mean_average_precision(ap[MATCH_IOU], detection),
                map_50=mean_average_precision(ap[0.5], detection),
                map_50_95=mean_average_precision_over(ap, COCO_THRESHOLDS, detection),
                map_all=mean_average_precision(ap[MATCH_IOU]),
                max_recall=max_recall_at_precision(matched_test),
                paired=paired,
                decomposition=decompose(test, test_dump.truth, class_names, detection),
                per_class=per_class,
                window_level=window_rows,
            )
        )

    return Comparison(
        protocol=Protocol(
            iou=MATCH_IOU,
            max_det=max_det,
            score_floor=SCORE_FLOOR,
            beta=beta,
            n_bootstrap=n_boot,
            confidence=CONFIDENCE,
            reference_criterion=str(reference_criterion),
            seed=SEED,
        ),
        val=describe(reference_val),
        test=describe(reference_test),
        detection_classes=[class_names[c] for c in detection],
        window_classes=[class_names[c] for c in saturated],
        models=compared,
    )
