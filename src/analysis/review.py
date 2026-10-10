from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_area, box_convert, box_iou

from core.config import RAW_DIR, P
from data.annotations import SPECIES_CODES, clean_annotations, list_files, species_of, unify_copies
from data.raven import BEGIN, CALL, END, HIGH, LOW, SCORE, SPECIES
from evaluation.metrics import MATCH_IOU, Boxes, assignments, overlaps, rows_by_image
from models.registry import LoadedModel
from prepare_annotations import COLUMNS, find_recording
from utils.audio import hz_to_y

# Una detección "cubre" una anotación si tapa al menos esta fracción de su área (RE3.3)
COVER = 0.5
# Las categorías de RE3.3: acierto, caja distinta, otra llamada, nada anotado
MATCH, BOX, LABEL, NONE = "match", "box", "label", "none"
# Las cajas de las anotaciones limpias, en el orden de las de Raven
CLEANED = ["begin_time_s", "end_time_s", "low_freq_hz", "high_freq_hz"]


def relative(path: str | Path) -> str:
    # Una grabación por su ruta bajo raw/, que es la misma en el servidor, en local y en el equipo
    return Path(str(path).split("data/raw/", 1)[-1]).as_posix()


def load_annotations(joined: dict[str, str]) -> pd.DataFrame:
    # Las tablas crudas de raw/, limpiadas en memoria igual que `prepare_annotations.py`, con las
    # copias unidas y las etiquetas unidas de la corrida. Incluye las filas en revisión.
    frames: list[pd.DataFrame] = []
    for table in list_files(RAW_DIR, ".txt"):
        audio = find_recording(table) if species_of(table) in SPECIES_CODES else None
        if audio is None:
            continue
        try:
            frame = clean_annotations(pd.read_csv(table, sep="\t"), species_of(audio))
        except Exception:  # una tabla ilegible no detiene nada, igual que en el pipeline
            continue
        if len(frame):
            frames.append(frame[COLUMNS].assign(audio_path=str(audio)))
    annotations = unify_copies(pd.concat(frames, ignore_index=True))
    label = annotations["species"] + "/" + annotations["call_type"].fillna("")
    return annotations.assign(
        requires_review=annotations["requires_review"].astype(bool),
        label=label.replace(joined),
        recording=annotations["audio_path"].map(relative),
    )


def tool_tables(
    loaded: LoadedModel, recordings: list[str], threshold: float, device: str = "cpu"
) -> pd.DataFrame:
    # La tabla de Raven que `detect.py` escribe para cada grabación, todas juntas
    from inference.predictor import predict

    tables = [
        predict(loaded, RAW_DIR / recording, device, threshold).assign(recording=recording)
        for recording in recordings
    ]
    table = pd.concat(tables, ignore_index=True)
    return table.assign(label=(table[SPECIES] + "/" + table[CALL]).str.lower())


def xyxy(frame: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    # Cajas en segundos × fila del mel en [0, 1]: la misma escala de frecuencia que ve el modelo.
    # El IoU no cambia si cada eje se estira por separado, así que segundos sirven igual que
    # fracciones de ventana.
    begin, end, low, high = (frame[c].to_numpy(dtype=float) for c in columns)
    y0, y1 = (np.clip(hz_to_y(f, P), 0.0, 1.0) for f in (low, high))
    return torch.tensor(np.stack([begin, y0, end, y1], axis=1), dtype=torch.float32).reshape(-1, 4)


def to_boxes(
    frame: pd.DataFrame, columns: list[str], ids: list[int], recordings: list[int]
) -> Boxes:
    return Boxes(
        box_convert(xyxy(frame, columns), "xyxy", "cxcywh"),
        torch.tensor(recordings, dtype=torch.long),
        torch.tensor(ids, dtype=torch.long),
        torch.ones(len(frame)),
    )


def miss(
    p: torch.Tensor, label: int, t: torch.Tensor, t_labels: torch.Tensor, t_claimed: torch.Tensor
) -> tuple[str, int | None]:
    # La regla de RE3.3 para una detección que no acertó, frente a las anotaciones de su
    # grabación (cajas xyxy): su categoría y la anotación a que se refiere
    if not len(t):
        return NONE, None
    iou = box_iou(p[None], t)[0]
    same = t_labels == label
    inter = (torch.min(p[2:], t[:, 2:]) - torch.max(p[:2], t[:, :2])).clamp(min=0).prod(dim=1)
    covered = same & (inter / box_area(t).clamp(min=1e-9) >= COVER)
    best_same = int((iou * same).argmax()) if same.any() else None
    best_other = int((iou * ~same).argmax()) if (~same).any() else None
    if best_same is not None and iou[best_same] >= MATCH_IOU and t_claimed[best_same]:
        return BOX, best_same
    if best_other is not None and iou[best_other] >= MATCH_IOU:
        return LABEL, best_other
    if int(covered.sum()) >= 2 or (best_same is not None and iou[best_same] > 0):
        return BOX, best_same
    return NONE, None


def categorize(det: pd.DataFrame, reference: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    # Cada detección (una fila de las tablas de la herramienta) con su categoría de RE3.3 frente a
    # `reference`, y la fila de `reference` a que se refiere. Acierta si toma una anotación de su
    # clase con IoU ≥ 0,3, en orden de score; si no, va por la regla de `miss`.
    ids = {name: i for i, name in enumerate(names)}
    recordings = {r: i for i, r in enumerate(sorted({*det.recording, *reference.recording}))}
    det = det.sort_values(SCORE, ascending=False, kind="stable")
    pred = to_boxes(
        det,
        [BEGIN, END, LOW, HIGH],
        [ids[n] for n in det.label],
        [recordings[r] for r in det.recording],
    )
    truth = to_boxes(
        reference,
        CLEANED,
        [ids[n] for n in reference.label],
        [recordings[r] for r in reference.recording],
    )
    pairs = assignments(overlaps(pred, truth, class_aware=True), MATCH_IOU)
    found = {p for p, _ in pairs}
    claimed = torch.zeros(len(truth.boxes), dtype=torch.bool)
    claimed[[t for _, t in pairs]] = True
    p_xyxy = box_convert(pred.boxes, "cxcywh", "xyxy")
    t_xyxy = box_convert(truth.boxes, "cxcywh", "xyxy")
    by_recording = rows_by_image(truth.image_ids)
    matched_row = dict(pairs)
    category, row_of = [], []
    for row, (label, recording) in enumerate(
        zip(pred.labels.tolist(), pred.image_ids.tolist(), strict=True)
    ):
        t = torch.tensor(by_recording.get(recording, []), dtype=torch.long)
        if row in found:
            category.append(MATCH)
            row_of.append(matched_row[row])
            continue
        c, k = miss(p_xyxy[row], label, t_xyxy[t], truth.labels[t], claimed[t])
        category.append(c)
        row_of.append(None if k is None else int(t[k]))
    index = reference.index.to_numpy()
    annotation = pd.array([None if k is None else index[k] for k in row_of], dtype="Int64")
    return det.assign(category=category, annotation=annotation).sort_index()


def touches(det: pd.DataFrame, others: pd.DataFrame) -> pd.Series:
    # Si cada detección se superpone, aunque sea un poco, con alguna fila de `others` de su
    # grabación (sin mirar la clase)
    recordings = {r: i for i, r in enumerate(sorted({*det.recording, *others.recording}))}
    pred = to_boxes(
        det, [BEGIN, END, LOW, HIGH], [0] * len(det), [recordings[r] for r in det.recording]
    )
    rest = to_boxes(others, CLEANED, [0] * len(others), [recordings[r] for r in others.recording])
    hit = np.zeros(len(det), dtype=bool)
    for overlap in overlaps(pred, rest, class_aware=False):
        hit[overlap.prediction_rows] = (overlap.iou > 0).any(dim=1).numpy()
    return pd.Series(hit, index=det.index)
