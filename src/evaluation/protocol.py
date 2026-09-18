import json
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from core.config import MAX_DETECTIONS, SCORE_FLOOR, SCORE_STEP, SEED, score_grid
from data.species import split_label
from evaluation.evaluator import RawPredictions
from evaluation.metrics import (
    MATCH_IOU,
    Boxes,
    assignments,
    average_precision_per_class,
    hits,
    mean_average_precision,
    overlaps,
    window_classes,
)

# El punto de operación de cada modelo, junto a su `best.pt`: es el umbral con el que arranca
# el visor. Lo escribe `compare_models.py` y lo lee `models.registry.load_checkpoint`;
# el lector vive en `inference.catalog` para que el visor lo consulte sin torch.
from inference.catalog import OPERATING_POINT, read_operating_point  # noqa: E402, F401

THRESHOLD_GRID = score_grid(SCORE_STEP, 1.0 - SCORE_STEP)
# Precisión mínima en val del punto de operación; la segunda sólo la usan los cuadernos
MIN_PRECISIONS = (0.70, 0.50)
# IoU al que se mide el encuadre, además del de acierto
STRICT_IOU = 0.5
N_BOOTSTRAP = 1000
CONFIDENCE = 0.95


class Point(NamedTuple):
    threshold: float
    recall: float | None
    precision: float | None
    n_above: int
    n_tp: int


# Umbral elegido en val a una precisión mínima, medido en test con IC de grabaciones. Lo usan
# los cuadernos (`pair`); la comparación imprime sólo el primer umbral, sin intervalo.
class Paired(NamedTuple):
    min_precision: float
    threshold: float | None
    val: Point | None
    test: Point | None
    interval: dict[str, list[float]]


# Una clase de ventana por ventana: se acierta la ventana, no la caja. Sólo para los cuadernos.
class WindowLevel(NamedTuple):
    name: str
    windows_gt: int
    windows_predicted: int
    recall: float | None
    precision: float | None


# Por especie/llamada, al umbral del modelo y sobre toda la curva
class ClassRow(NamedTuple):
    name: str
    n_gt: int
    recall: float | None
    precision: float | None
    ap: float | None  # a MATCH_IOU
    ap_strict: float | None  # a STRICT_IOU
    window_class: bool  # la caja ocupa la ventana; no entra en la mAP


# Detección, encuadre y clasificación por separado, al umbral del modelo
class Axes(NamedTuple):
    recall: float | None  # emparejando sin clase
    precision: float | None
    median_iou: float | None  # de los pares sin clase
    recall_strict: float | None  # con clase, a STRICT_IOU
    class_correct: float | None  # de los pares sin clase, misma clase
    species_correct: float | None


class ModelComparison(NamedTuple):
    model: str
    threshold: float | None  # elegido en val; None si ningún umbral llega a la precisión mínima
    val: Point | None
    test: Point | None
    map_30: float | None
    map_50: float | None
    axes: Axes | None
    per_class: list[ClassRow]


class Protocol(NamedTuple):
    iou: float
    strict_iou: float
    max_det: int
    score_floor: float
    min_precision: float
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


def match(
    predictions: Boxes,
    truth: Boxes,
    n_images: int,
    iou: float = MATCH_IOU,
    class_aware: bool = True,
) -> Matched:
    return Matched(
        scores=predictions.scores,
        found=hits(overlaps(predictions, truth, class_aware), len(predictions.boxes), iou),
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
    matched: Matched, recordings: Tensor, threshold: float, n_boot: int = N_BOOTSTRAP
) -> dict[str, list[float]]:
    # IC por remuestreo de grabaciones: las ventanas de una misma grabación no son independientes.
    # Lo usan los cuadernos; la comparación no lo imprime.
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


def pair(
    matched_val: Matched, matched_test: Matched, recordings: Tensor, min_precision: float
) -> Paired:
    chosen = select(matched_val, min_precision)
    if chosen is None:
        return Paired(min_precision, None, None, None, {})
    return Paired(
        min_precision,
        chosen.threshold,
        chosen,
        operating_point(matched_test, chosen.threshold),
        bootstrap(matched_test, recordings.to(torch.int64), chosen.threshold),
    )


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


def measure_axes(predictions: Boxes, truth: Boxes, names: list[str], threshold: float) -> Axes:
    # Emparejando sin clase se separa encontrar la llamada de nombrarla; el encuadre es el IoU
    # de esos pares y el recall estricto con clase.
    above = predictions.select(torch.arange(int((predictions.scores >= threshold).sum())))
    pairs = assignments(overlaps(above, truth, class_aware=False), MATCH_IOU)
    n_gt, n_above = len(truth.boxes), len(above.boxes)
    if not pairs:
        return Axes(0.0 if n_gt else None, 0.0 if n_above else None, None, None, None, None)
    p_rows = torch.tensor([p for p, _ in pairs])
    t_rows = torch.tensor([t for _, t in pairs])
    iou = box_iou(
        box_convert(above.boxes[p_rows], "cxcywh", "xyxy"),
        box_convert(truth.boxes[t_rows], "cxcywh", "xyxy"),
    ).diagonal()
    predicted = [names[int(c)] for c in above.labels[p_rows]]
    annotated = [names[int(c)] for c in truth.labels[t_rows]]
    same_class = [p == a for p, a in zip(predicted, annotated, strict=True)]
    same_species = [
        split_label(p)[0] == split_label(a)[0] for p, a in zip(predicted, annotated, strict=True)
    ]
    strict = hits(overlaps(above, truth, class_aware=True), n_above, STRICT_IOU)
    return Axes(
        recall=len(pairs) / n_gt,
        precision=len(pairs) / n_above,
        median_iou=float(iou.median()),
        recall_strict=float(strict.sum()) / n_gt,
        class_correct=sum(same_class) / len(pairs),
        species_correct=sum(same_species) / len(pairs),
    )


def class_rows(
    matched: Matched,
    predictions: Boxes,
    truth: Boxes,
    threshold: float,
    names: list[str],
    saturated: list[int],
) -> list[ClassRow]:
    ap = average_precision_per_class(predictions, truth, len(names))
    ap_strict = average_precision_per_class(predictions, truth, len(names), STRICT_IOU)
    n_above = matched.above(threshold)
    predicted_labels = predictions.labels[:n_above]
    found = matched.found[:n_above]
    rows = []
    for class_id, name in enumerate(names):
        n_gt = int((truth.labels == class_id).sum())
        of_class = predicted_labels == class_id
        tp = float(found[of_class].sum())
        rows.append(
            ClassRow(
                name,
                n_gt,
                tp / n_gt if n_gt else None,
                tp / int(of_class.sum()) if of_class.any() else None,
                ap[class_id],
                ap_strict[class_id],
                class_id in saturated,
            )
        )
    return rows


def write_operating_point(
    directory: Path, model: str, compared: ModelComparison, protocol: Protocol
) -> Path:
    path = directory / OPERATING_POINT
    record = {
        "model": model,
        "threshold": compared.threshold,
        "val": as_json(compared.val),
        "test": as_json(compared.test),
        "protocol": as_json(protocol),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


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
    min_precision: float = MIN_PRECISIONS[0],
    max_det: int = MAX_DETECTIONS,
) -> Comparison:
    # Umbral de cada modelo elegido en val; todo lo demás medido en test con ese umbral.
    check_comparable(models)
    reference_val, reference_test = models[0]
    names = reference_val.labels
    saturated = window_classes(reference_test.truth, len(names))
    detection = [c for c in range(len(names)) if c not in saturated]

    compared: list[ModelComparison] = []
    for val_dump, test_dump in models:
        val = equalize(val_dump.predictions, max_det)
        test = equalize(test_dump.predictions, max_det)
        truth = test_dump.truth
        matched_test = match(test, truth, test_dump.n_images)
        chosen = select(match(val, val_dump.truth, val_dump.n_images), min_precision)
        per_class = []
        at_test = axes_at = None
        if chosen is not None:
            at_test = operating_point(matched_test, chosen.threshold)
            axes_at = measure_axes(test, truth, names, chosen.threshold)
            per_class = class_rows(matched_test, test, truth, chosen.threshold, names, saturated)
        compared.append(
            ModelComparison(
                model=test_dump.model,
                threshold=None if chosen is None else chosen.threshold,
                val=chosen,
                test=at_test,
                map_30=mean_average_precision(
                    average_precision_per_class(test, truth, len(names)), detection
                ),
                map_50=mean_average_precision(
                    average_precision_per_class(test, truth, len(names), STRICT_IOU), detection
                ),
                axes=axes_at,
                per_class=per_class,
            )
        )

    return Comparison(
        protocol=Protocol(MATCH_IOU, STRICT_IOU, max_det, SCORE_FLOOR, min_precision, SEED),
        val=describe(reference_val),
        test=describe(reference_test),
        detection_classes=[names[c] for c in detection],
        window_classes=[names[c] for c in saturated],
        models=compared,
    )
