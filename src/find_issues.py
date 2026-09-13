"""CLOD sobre un volcado fuera de muestra: anotaciones que parecen faltar, sobrar, estar mal
etiquetadas o mal ubicadas (Chachuła et al., 2023). Escribe una tabla de hallazgos, un hallazgo
por fila con su `Calidad` (peor primero) y la grabación donde está, para revisarla en Raven o
en un notebook.

    python src/find_issues.py runs/detr_kfold5/detr_kfold5_train_predictions.pt

Predicciones y anotaciones de una ventana se ligan por IoU. Una predicción sin anotación
ligada es `missing` si supera el umbral de confianza de su clase; una anotación sin predicción
confiada de su clase es `spurious`, o `label` si otra clase sí la ve; con predicción de su clase
pero mal superpuesta, `location`. El umbral por clase es el de confident learning: el score
medio que el modelo da a las anotaciones de esa clase.
"""

import argparse
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import torch
from torch import Tensor
from torchvision.ops import box_convert, box_iou

from core.config import P
from core.runtime import setup_logging
from data import cache
from data.raven import CLEANED_BOX_COLUMNS
from data.species import LabelSet
from evaluation.evaluator import RawPredictions
from evaluation.metrics import MATCH_IOU, Boxes, rows_by_image
from evaluation.protocol import equalize
from utils.audio import y_to_hz

logger = logging.getLogger("find_issues")

# Columnas de la tabla de hallazgos: cajas con las columnas de `cleaned/`, el tipo de hallazgo,
# su calidad y la ruta de la grabación.
ISSUE, QUALITY, RECORDING = "Hallazgo", "Calidad", "recording"
SPURIOUS, MISSING, LOCATION, LABEL = "spurious", "missing", "location", "label"

# Liga predicción y anotación
LINK_IOU = MATCH_IOU
# Por debajo, una pareja de la misma clase es `location`
LOCATION_IOU = 0.6
# Predicciones que entran al análisis
SCORE_FLOOR = 0.05
# Redondeo con el que se funden los hallazgos repetidos entre ventanas solapadas
ROUND_S, ROUND_HZ = 0.05, 50.0
SPECIES, SUGGESTION, SCORE, WINDOW = "species", "sugerencia", "score", "ventana"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dump", type=Path, help="*_predictions.pt fuera de muestra")
    parser.add_argument(
        "--output", type=Path, help="por defecto hallazgos_<split>.csv junto al volcado"
    )
    parser.add_argument("--score-floor", type=float, default=SCORE_FLOOR)
    return parser.parse_args()


class Finding(NamedTuple):
    window: int
    issue: str
    label: int
    box: Tensor  # cxcywh en la ventana
    quality: float
    score: float
    suggestion: int | None


class WindowBoxes(NamedTuple):
    predictions: Tensor  # filas en `predictions`
    truth: Tensor  # filas en `truth`
    iou: Tensor  # [predicciones, anotaciones]


def windows(predictions: Boxes, truth: Boxes) -> Iterator[tuple[int, WindowBoxes]]:
    predicted_xyxy = box_convert(predictions.boxes, "cxcywh", "xyxy")
    truth_xyxy = box_convert(truth.boxes, "cxcywh", "xyxy")
    prediction_rows = rows_by_image(predictions.image_ids)
    truth_rows = rows_by_image(truth.image_ids)
    for image_id in sorted(prediction_rows.keys() | truth_rows.keys()):
        pred = torch.tensor(prediction_rows.get(image_id, []), dtype=torch.long)
        gt = torch.tensor(truth_rows.get(image_id, []), dtype=torch.long)
        yield image_id, WindowBoxes(pred, gt, box_iou(predicted_xyxy[pred], truth_xyxy[gt]))


def class_thresholds(predictions: Boxes, truth: Boxes, n_classes: int) -> Tensor:
    # Score medio que reciben las anotaciones de cada clase de una predicción ligada de su clase.
    best = torch.zeros(len(truth.boxes))
    for _, window in windows(predictions, truth):
        if not len(window.truth):
            continue
        same = predictions.labels[window.predictions][:, None] == truth.labels[window.truth][None]
        linked = (window.iou >= LINK_IOU) & same
        scores = predictions.scores[window.predictions][:, None].expand_as(linked)
        best[window.truth] = scores.masked_fill(~linked, 0.0).amax(dim=0)
    thresholds = torch.zeros(n_classes)
    for class_id in range(n_classes):
        of_class = best[truth.labels == class_id]
        thresholds[class_id] = of_class.mean() if len(of_class) else 1.0
    return thresholds


def find(predictions: Boxes, truth: Boxes, thresholds: Tensor) -> list[Finding]:
    findings: list[Finding] = []
    for image_id, window in windows(predictions, truth):
        p_labels = predictions.labels[window.predictions].long()
        p_scores = predictions.scores[window.predictions]
        linked = window.iou >= LINK_IOU

        for j, row in enumerate(window.predictions.tolist()):
            label, score = int(p_labels[j]), float(p_scores[j])
            if not linked[j].any() and score >= float(thresholds[label]):
                findings.append(
                    Finding(
                        image_id, MISSING, label, predictions.boxes[row], 1 - score, score, None
                    )
                )

        for i, row in enumerate(window.truth.tolist()):
            label = int(truth.labels[row])
            same = linked[:, i] & (p_labels == label)
            other = linked[:, i] & (p_labels != label)
            own_score = float(p_scores.masked_fill(~same, 0.0).max()) if len(same) else 0.0
            box = truth.boxes[row]
            if own_score >= float(thresholds[label]):
                iou = float(window.iou[:, i].masked_fill(~same, 0.0).max())
                if iou < LOCATION_IOU:
                    findings.append(Finding(image_id, LOCATION, label, box, iou, own_score, None))
                continue
            if other.any():
                best = int(p_scores.masked_fill(~other, 0.0).argmax())
                other_label, other_score = int(p_labels[best]), float(p_scores[best])
                if other_score >= float(thresholds[other_label]):
                    findings.append(
                        Finding(image_id, LABEL, label, box, own_score, other_score, other_label)
                    )
                    continue
            findings.append(Finding(image_id, SPURIOUS, label, box, own_score, own_score, None))
    return findings


def to_table(findings: list[Finding], dump: RawPredictions, labels: LabelSet) -> pd.DataFrame:
    sources = cache.sources(dump.split, dump.n_images)
    boxes = torch.stack([f.box for f in findings])
    starts = torch.tensor([sources.clip_start_s[f.window] for f in findings])
    x0, x1 = (
        starts + (boxes[:, 0] - boxes[:, 2] / 2) * P.clip_len_s,
        starts + (boxes[:, 0] + boxes[:, 2] / 2) * P.clip_len_s,
    )
    y0, y1 = (
        (boxes[:, 1] - boxes[:, 3] / 2).clamp(0, 1),
        (boxes[:, 1] + boxes[:, 3] / 2).clamp(0, 1),
    )
    begin, end, low, high = CLEANED_BOX_COLUMNS
    table = pd.DataFrame(
        {
            RECORDING: [
                sources.recordings[sources.recording_of_window[f.window]] for f in findings
            ],
            SPECIES: [labels.name(f.label) for f in findings],
            ISSUE: [f.issue for f in findings],
            QUALITY: [round(f.quality, 4) for f in findings],
            begin: x0.tolist(),
            end: x1.tolist(),
            low: y_to_hz(y0.numpy(), P).tolist(),
            high: y_to_hz(y1.numpy(), P).tolist(),
            SCORE: [round(f.score, 4) for f in findings],
            SUGGESTION: [
                None if f.suggestion is None else labels.name(f.suggestion) for f in findings
            ],
            WINDOW: [f.window for f in findings],
        }
    )
    # La misma caja aparece en dos ventanas solapadas: queda el hallazgo de peor calidad.
    key = [RECORDING, ISSUE, SPECIES]
    rounded = table.assign(
        _b=(table[begin] / ROUND_S).round(),
        _e=(table[end] / ROUND_S).round(),
        _l=(table[low] / ROUND_HZ).round(),
        _h=(table[high] / ROUND_HZ).round(),
    )
    keep = rounded.sort_values(QUALITY).drop_duplicates([*key, "_b", "_e", "_l", "_h"]).index
    return table.loc[sorted(keep)].sort_values(QUALITY).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    setup_logging()
    dump = RawPredictions.load(args.dump)
    labels = LabelSet(dump.labels)
    predictions = equalize(dump.predictions, score_floor=args.score_floor)
    logger.info(
        "%s %s: %d ventanas, %d anotaciones, %d predicciones >= %.2f",
        dump.model,
        dump.split,
        dump.n_images,
        len(dump.truth.boxes),
        len(predictions.boxes),
        args.score_floor,
    )

    thresholds = class_thresholds(predictions, dump.truth, len(labels))
    for class_id, threshold in enumerate(thresholds.tolist()):
        logger.info("umbral %-10s %.2f", labels.name(class_id), threshold)

    table = to_table(find(predictions, dump.truth, thresholds), dump, labels)
    output = args.output or args.dump.with_name(f"hallazgos_{dump.split}.csv")
    table.to_csv(output, index=False)
    counts = table.groupby([ISSUE, SPECIES]).size().unstack(ISSUE, fill_value=0)
    logger.info("%d hallazgos -> %s\n%s", len(table), output, counts.to_string())


if __name__ == "__main__":
    main()
