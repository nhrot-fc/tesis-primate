from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal

from clod.cluster import BOX_COLUMNS as CLOD_BOXES
from clod.issues import ISSUE, MISSING, QUALITY, SPURIOUS
from viewer.spectrogram import Waveform

ANNOTATIONS = "Anotaciones"
DETECTIONS = "Modelo"
FINDINGS = "Hallazgos"
SOURCES = (ANNOTATIONS, DETECTIONS, FINDINGS)
COLORS = {ANNOTATIONS: "#00d8ff", DETECTIONS: "#8cff3d", FINDINGS: "#ff4d6d"}

BEGIN, END, LOW, HIGH = "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"
BOX_COLUMNS = [BEGIN, END, LOW, HIGH]
SPECIES, CALL = "Species", "Call type"

RECORDING, VERDICT = "recording", "Veredicto"
ACCEPTED, REJECTED = "aceptado", "rechazado"
# La caja de CLOD y la del Raven fuente sólo difieren en el redondeo del archivo.
TOLERANCE_S = 1e-3


# Una tabla de CLOD se reconoce por su columna de hallazgo; el resto se lee como Raven.
def read_table(path: Path) -> tuple[str, pd.DataFrame]:
    table = pd.read_csv(path, sep=None, engine="python")
    if ISSUE in table.columns:
        absent = [name for name in [*CLOD_BOXES, QUALITY, RECORDING] if name not in table.columns]
        if absent:
            raise ValueError(f"'{path.name}' no trae las columnas: {', '.join(absent)}")
        table = table.rename(columns=dict(zip(CLOD_BOXES, BOX_COLUMNS, strict=True)))
        # CLOD pega especie y llamada en 'sm/fs'; el Raven fuente las trae aparte y en mayúscula.
        table[[SPECIES, CALL]] = table["species"].str.upper().str.split("/", n=1, expand=True)
        table[VERDICT] = ""
        return FINDINGS, table.sort_values(QUALITY, kind="mergesort").reset_index(drop=True)

    absent = [name for name in BOX_COLUMNS if name not in table.columns]
    if absent:
        raise ValueError(f"'{path.name}' no tiene las columnas: {', '.join(absent)}")
    return ANNOTATIONS, table


def label(row: pd.Series) -> str:
    name = "/".join(str(row[c]) for c in (SPECIES, CALL) if c in row.index)
    return f"{row[ISSUE]} {name}" if ISSUE in row.index else name


@dataclass(frozen=True)
class Row:
    source: str
    index: Hashable  # la etiqueta de la fila en su DataFrame
    begin: float
    end: float
    low: float
    high: float
    label: str
    score: float  # NaN cuando la tabla no trae 'Score'


# Estado compartido entre el espectrograma y la tabla de revisión: quien mira las cajas
# las lee de aquí y quien las edita emite `changed`.
class Session(QObject):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.audio_path: Path | None = None
        self.model_path: Path | None = None
        self.waveform: Waveform | None = None
        self.sr = 1
        self.score = 0.5
        self.tables: dict[str, pd.DataFrame | None] = dict.fromkeys(SOURCES)

    @property
    def duration(self) -> float:
        return 0.0 if self.waveform is None else self.waveform.size / self.sr

    def set_audio(self, path: Path, waveform: Waveform, sr: int) -> None:
        self.audio_path, self.waveform, self.sr = path, waveform, sr
        self.tables[DETECTIONS] = None  # son de otro audio
        self.changed.emit()

    def set_table(self, source: str, table: pd.DataFrame) -> None:
        self.tables[source] = table
        # Al recargar el Raven de una grabación ya revisada se rehacen sus veredictos.
        judged = self.visible(FINDINGS) if source == ANNOTATIONS else None
        if judged is not None:
            for _, row in judged.iterrows():
                self.apply(row)
        self.changed.emit()

    def set_score(self, score: float) -> None:
        self.score = score
        self.changed.emit()

    def visible(self, source: str) -> pd.DataFrame | None:
        table = self.tables[source]
        if table is None:
            return None
        # Los hallazgos son de todo el dataset; sobre el audio abierto van sólo los suyos.
        if RECORDING in table.columns:
            table = table.loc[table[RECORDING] == str(self.audio_path)]
        if "Score" not in table.columns:
            return table
        return table.loc[table["Score"] >= self.score]

    def remove(self, targets: list[tuple[str, Hashable]]) -> None:
        for source in SOURCES:
            indices = [index for target, index in targets if target == source]
            table = self.tables[source]
            if indices and table is not None:
                self.tables[source] = table.drop(index=indices)
        self.changed.emit()

    def judge(self, index: Hashable, verdict: str) -> str:
        findings = self.tables[FINDINGS]
        if findings is None:
            return ""
        findings.loc[index, VERDICT] = verdict
        done = self.apply(findings.loc[index])
        self.changed.emit()
        return done

    # Lleva un veredicto aceptado a las anotaciones y cuenta qué hizo. `location` y
    # `label` piden mover la caja, y el visor no la edita: sólo quedan registrados.
    def apply(self, row: pd.Series) -> str:
        annotations = self.tables[ANNOTATIONS]
        if row[VERDICT] != ACCEPTED or annotations is None:
            return row[VERDICT] or "sin veredicto"
        if row[ISSUE] == MISSING:
            self.tables[ANNOTATIONS] = pd.concat(
                [annotations, row[[*BOX_COLUMNS, SPECIES, CALL]].to_frame().T], ignore_index=True
            )
            return "anotación añadida"
        if row[ISSUE] == SPURIOUS:
            gap = (annotations[BEGIN] - row[BEGIN]).abs() + (annotations[END] - row[END]).abs()
            if len(gap) and gap.min() < TOLERANCE_S:
                self.tables[ANNOTATIONS] = annotations.drop(index=gap.idxmin())
                return "anotación borrada"
            return "aceptado, pero esa anotación no está en la tabla"
        return "aceptado; la caja hay que moverla a mano"

    # Las cajas de un origen ordenadas por tiempo, que es como se revisan.
    def rows(self, source: str) -> list[Row]:
        table = self.visible(source)
        if table is None:
            return []
        boxes = table[BOX_COLUMNS].to_numpy(dtype=float)
        scores = (
            table["Score"].to_numpy(dtype=float)
            if "Score" in table.columns
            else np.full(len(table), float("nan"))
        )
        rows = [
            Row(source, index, begin, end, low, high, label(row), score)
            for (index, row), (begin, end, low, high), score in zip(
                table.iterrows(), boxes, scores, strict=True
            )
        ]
        return sorted(rows, key=lambda row: row.begin)
