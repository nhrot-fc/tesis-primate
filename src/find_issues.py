"""CLOD sobre un volcado fuera de muestra: anotaciones que parecen faltar, sobrar, estar mal
etiquetadas o mal ubicadas (Chachuła et al., 2023). Escribe una tabla de hallazgos, un hallazgo
por fila con su `Calidad` (peor es menor) y la grabación donde está, ordenada por grabación y
tipo, para revisarla en Raven o en un notebook.

    python src/find_issues.py runs/detr_kfold5/detr_kfold5_train_predictions.pt

Predicciones y anotaciones de una ventana se ligan por IoU. Una predicción sin anotación
ligada es `missing` si supera el umbral de confianza de su clase; una anotación sin predicción
confiada de su clase es `label` si otra clase sí la ve con confianza, o `spurious` si ninguna
predicción de ninguna clase la toca (el modelo ve fondo); con predicción confiada de su clase
pero mal superpuesta, `location`. Una anotación que sólo tiene predicciones poco confiadas no
es hallazgo: como en confident learning, lo que no es confiado en ninguna clase no cuenta. El
umbral por clase es el de confident learning: el score medio que el modelo da a las anotaciones
de esa clase.

Cada hallazgo se contrasta con la tabla cruda de Raven de su grabación en `data/unified/` (la
unión de todas las copias, sin limpiar): qué fila hay debajo (`raw_*`) y qué hizo la limpieza con
ella (`raw_estado`: limpia, en revisión, sin clase, descartada o sin fila). Un `missing` sobre
una fila descartada (ruido) o en revisión no es una anotación que falte; uno sin fila, sí.
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

from core.config import CLEANED_DIR, NMS_IOU, UNIFIED_DIR, P
from core.runtime import setup_logging
from data import cache
from data.annotations import MAX_FREQ_HZ
from data.raven import BEGIN, BOX_COLUMNS, CALL, CLEANED_BOX_COLUMNS, END, HIGH, LOW, SPECIES
from data.species import LABEL_SEPARATOR, LabelSet
from evaluation.evaluator import RawPredictions
from evaluation.metrics import MATCH_IOU, Boxes, rows_by_image
from evaluation.protocol import equalize
from utils.audio import hz_to_y, y_to_hz
from utils.boxes import suppress_nested

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
LABEL_COLUMN, SUGGESTION, SCORE, WINDOW = "species", "sugerencia", "score", "ventana"
WINDOW_START = "ventana_inicio_s"

# Contraste con `unified/`: la fila cruda debajo del hallazgo y qué hizo la limpieza con ella
RAW_IOU, RAW_SPECIES, RAW_CALL, RAW_FOLDER, RAW_RATING, RAW_LABEL, RAW_STATE = (
    "raw_iou",
    "raw_species",
    "raw_call",
    "raw_folder",
    "raw_rating",
    "raw_clase",  # especie/llamada con que quedó en cleaned/, ya unidas las sinónimas
    "raw_estado",
)
# Estados: nada anotado; la limpieza la quitó (ruido, caja degenerada); `requires_review`;
# quedó pero su clase no entró al conjunto (`excluded_labels`, < 100); es una anotación del conjunto.
NO_ROW, DROPPED, REVIEW, EXCLUDED, CLEAN = (
    "sin fila",
    "descartada",
    "en revisión",
    "sin clase",
    "limpia",
)
STATES = [NO_ROW, DROPPED, REVIEW, EXCLUDED, CLEAN]
# Columnas de la tabla cruda que no están en `data.raven`
FOLDER_SPECIES, RATING = "Folder Species", "Rating"
# Una fila cruda está en `cleaned/` si su caja aparece con esta tolerancia (s y Hz)
BOX_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dump", type=Path, help="*_predictions.pt fuera de muestra")
    parser.add_argument(
        "--output", type=Path, help="por defecto hallazgos_<split>.csv junto al volcado"
    )
    parser.add_argument("--score-floor", type=float, default=SCORE_FLOOR)
    parser.add_argument(
        "--unified", type=Path, default=UNIFIED_DIR, help="tablas crudas; sin ella no se contrasta"
    )
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
        if not len(window.truth) or not len(window.predictions):
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
            if not linked[:, i].any():
                # Nada la toca: la calidad es lo más que se le acerca cualquier predicción
                nearest = float(window.iou[:, i].max()) if len(window.predictions) else 0.0
                findings.append(Finding(image_id, SPURIOUS, label, box, nearest, 0.0, None))
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
            LABEL_COLUMN: [labels.name(f.label) for f in findings],
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
            WINDOW_START: starts.tolist(),
        }
    )
    return merge_duplicates(table).sort_values([RECORDING, ISSUE, begin]).reset_index(drop=True)


def merge_duplicates(table: pd.DataFrame) -> pd.DataFrame:
    # La misma caja sale en dos ventanas solapadas, con bordes algo distintos o recortada por una
    # de ellas: por grabación, tipo y clase se funden como el predictor funde las detecciones
    # entre ventanas (`suppress_nested` a NMS_IOU, y la anidada en otra); queda la de peor calidad.
    begin, end, low, high = CLEANED_BOX_COLUMNS
    kept: list[int] = []
    for _, group in table.groupby([RECORDING, ISSUE]):
        xyxy = to_window_space(torch.tensor(group[[begin, low, end, high]].to_numpy(dtype=float)))
        priority = torch.tensor(1.0 - group[QUALITY].to_numpy(dtype=float))
        classes = torch.tensor(pd.factorize(group[LABEL_COLUMN])[0])
        keep = suppress_nested(xyxy.float(), priority.float(), classes, NMS_IOU)
        kept.extend(group.index[keep.tolist()])
    return table.loc[sorted(kept)]


def read_raw(recording: str, unified: Path) -> pd.DataFrame | None:
    # La tabla de `unified/` se llama como el wav; todo texto salvo la caja.
    path = unified / f"{Path(recording).stem}.txt"
    if not path.is_file():
        return None
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    for column in BOX_COLUMNS:
        if column not in table:
            return None
        table[column] = pd.to_numeric(table[column], errors="coerce")
    valid = table.dropna(subset=BOX_COLUMNS)
    valid = valid[(valid[END] > valid[BEGIN]) & (valid[HIGH] > valid[LOW])]
    return valid.reset_index(drop=True)


def to_window_space(xyxy: Tensor) -> Tensor:
    # Tiempo en segundos, frecuencia en la fila del mel en [0, 1]: el IoU sale igual que el del
    # modelo, que mira la ventana y no los hercios.
    y = torch.tensor(hz_to_y(xyxy[:, [1, 3]].numpy(), P)).clamp(0.0, 1.0)
    return torch.stack([xyxy[:, 0], y[:, 0], xyxy[:, 2], y[:, 1]], dim=1)


def clipped_iou(findings_xyxy: Tensor, starts: Tensor, raw_xyxy: Tensor) -> Tensor:
    # [hallazgos, filas crudas]: IoU con la fila cruda recortada a la ventana del hallazgo, que
    # es lo que el modelo y la anotación limpia vieron; una llamada más larga que la ventana no
    # liga con su propia fila si se compara entera.
    f = to_window_space(findings_xyxy)[:, None, :]
    r = to_window_space(raw_xyxy)[None, :, :].expand(len(findings_xyxy), -1, -1).clone()
    r[..., 0] = torch.maximum(r[..., 0], starts[:, None])
    r[..., 2] = torch.minimum(r[..., 2], starts[:, None] + P.clip_len_s)
    width = (torch.minimum(f[..., 2], r[..., 2]) - torch.maximum(f[..., 0], r[..., 0])).clamp(min=0)
    height = (torch.minimum(f[..., 3], r[..., 3]) - torch.maximum(f[..., 1], r[..., 1])).clamp(
        min=0
    )
    intersection = width * height
    area_f = (f[..., 2] - f[..., 0]) * (f[..., 3] - f[..., 1])
    area_r = (r[..., 2] - r[..., 0]).clamp(min=0) * (r[..., 3] - r[..., 1]).clamp(min=0)
    return intersection / (area_f + area_r - intersection).clamp(min=1e-9)


def read_cleaned(recording: str) -> pd.DataFrame:
    # Todas las copias de la grabación en `cleaned/`, una por carpeta de especie.
    frames = [
        pd.read_csv(path, sep="\t", keep_default_na=False)
        for path in sorted(CLEANED_DIR.rglob(f"{Path(recording).stem}.txt"))
    ]
    columns = ["species", "call_type", *CLEANED_BOX_COLUMNS, "requires_review"]
    return (
        pd.concat(frames, ignore_index=True)[columns] if frames else pd.DataFrame(columns=columns)
    )


def cleaning_state(
    raw: pd.DataFrame, cleaned: pd.DataFrame, names: list[str]
) -> list[tuple[str, str | None]]:
    # Una fila cruda sigue en `cleaned/` si su caja aparece ahí (la frecuencia alta se recorta al
    # tope del mel); si alguna copia la lleva con `requires_review`, el manifest descarta sus ventanas.
    raw_boxes = torch.tensor(raw[BOX_COLUMNS].to_numpy(dtype=float))
    raw_boxes[:, 3] = raw_boxes[:, 3].clamp(max=MAX_FREQ_HZ)
    if not len(cleaned):
        return [(DROPPED, None)] * len(raw)
    joined = cache.meta().get("joined_labels", {})
    clean_boxes = torch.tensor(cleaned[CLEANED_BOX_COLUMNS].to_numpy(dtype=float))
    under_review = torch.tensor(cleaned["requires_review"].to_numpy(dtype=bool))
    labels = [
        joined.get(label, label)
        for label in cleaned["species"] + LABEL_SEPARATOR + cleaned["call_type"]
    ]
    same = (raw_boxes[:, None, :] - clean_boxes[None, :, :]).abs().amax(dim=2) < BOX_TOLERANCE
    states: list[tuple[str, str | None]] = []
    for row in same:
        if not row.any():
            states.append((DROPPED, None))
            continue
        label = labels[int(row.to(torch.uint8).argmax())]
        if bool(row[under_review].any()):
            states.append((REVIEW, label))
        else:
            states.append((CLEAN if label in names else EXCLUDED, label))
    return states


def contrast(table: pd.DataFrame, labels: LabelSet, unified: Path = UNIFIED_DIR) -> pd.DataFrame:
    # Por grabación: la fila cruda de mayor IoU bajo cada hallazgo, si liga. La tabla trae la
    # ventana de cada hallazgo (`WINDOW_START`): las filas se recortan a ella.
    begin, end, low, high = CLEANED_BOX_COLUMNS
    columns = [RAW_IOU, RAW_SPECIES, RAW_CALL, RAW_FOLDER, RAW_RATING, RAW_LABEL, RAW_STATE]
    raw_columns = pd.DataFrame(index=table.index, columns=columns, dtype=object)
    for recording, group in table.groupby(RECORDING):
        raw = read_raw(str(recording), unified)
        if raw is None or not len(raw):
            continue
        states = cleaning_state(raw, read_cleaned(str(recording)), labels.names)
        findings_xyxy = torch.tensor(group[[begin, low, end, high]].to_numpy(dtype=float))
        starts = torch.tensor(group[WINDOW_START].to_numpy(dtype=float))
        raw_xyxy = torch.tensor(raw[[BEGIN, LOW, END, HIGH]].to_numpy(dtype=float))
        iou, best = clipped_iou(findings_xyxy, starts, raw_xyxy).max(dim=1)
        for index, value, j in zip(group.index, iou.tolist(), best.tolist(), strict=True):
            if value < LINK_IOU:
                continue
            row = raw.iloc[j]
            raw_columns.loc[index] = [
                round(value, 3),
                row.get(SPECIES, ""),
                row.get(CALL, ""),
                row.get(FOLDER_SPECIES, ""),
                row.get(RATING, ""),
                *reversed(states[j]),
            ]
    raw_columns[RAW_STATE] = raw_columns[RAW_STATE].fillna(NO_ROW)
    return pd.concat([table, raw_columns], axis=1)


def summarize_contrast(table: pd.DataFrame) -> str:
    crossed = (
        table.groupby([ISSUE, RAW_STATE])
        .size()
        .unstack(RAW_STATE, fill_value=0)
        .reindex(columns=STATES, fill_value=0)
    )
    lines = [crossed.to_string()]
    missing = table[table[ISSUE] == MISSING]
    dropped = missing[missing[RAW_STATE] == DROPPED]
    if len(dropped):
        by_call = dropped[RAW_CALL].str.strip().str.lower().replace("", "(vacío)").value_counts()
        lines.append(
            f"`{MISSING}` sobre fila descartada, por `Call type` crudo: "
            + ", ".join(f"{k} {v}" for k, v in by_call.head(8).items())
        )
    review = missing[missing[RAW_STATE] == REVIEW]
    if len(review):
        lines.append(
            f"`{MISSING}` sobre fila en revisión: "
            + ", ".join(f"{k} {v}" for k, v in review[RAW_LABEL].value_counts().head(8).items())
        )
    excluded = missing[missing[RAW_STATE] == EXCLUDED]
    if len(excluded):
        lines.append(
            f"`{MISSING}` sobre fila sin clase: "
            + ", ".join(f"{k} {v}" for k, v in excluded[RAW_LABEL].value_counts().items())
        )
    clean = missing[missing[RAW_STATE] == CLEAN]
    if len(clean):
        same = int((clean[RAW_LABEL] == clean[LABEL_COLUMN]).sum())
        lines.append(
            f"`{MISSING}` sobre anotación del conjunto: {same} de su clase (la caja quedó fuera "
            f"de la ventana o con IoU < {LINK_IOU:g}), {len(clean) - same} de otra clase"
        )
    annotated = table[table[RAW_STATE].isin([CLEAN, REVIEW, EXCLUDED])]
    written = annotated[RAW_SPECIES].str.strip().str.lower()
    folder = annotated[RAW_FOLDER].str.strip().str.lower()
    discordant = annotated[(written != "") & (written != folder)]
    if len(discordant):
        lines.append(
            f"{len(discordant)} hallazgos sobre filas con `Species` escrita distinta de la "
            "carpeta: " + ", ".join(f"{k} {v}" for k, v in discordant.groupby(ISSUE).size().items())
        )
    return "\n".join(lines)


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
    counts = table.groupby([ISSUE, LABEL_COLUMN]).size().unstack(ISSUE, fill_value=0)
    logger.info("%d hallazgos\n%s", len(table), counts.to_string())
    if args.unified.is_dir():
        table = contrast(table, labels, args.unified)
        logger.info("contraste con %s\n%s", args.unified, summarize_contrast(table))
    else:
        logger.warning("sin %s: no se contrasta con las tablas crudas", args.unified)
    output = args.output or args.dump.with_name(f"hallazgos_{dump.split}.csv")
    table.to_csv(output, index=False)
    logger.info("tabla de hallazgos -> %s", output)


if __name__ == "__main__":
    main()
