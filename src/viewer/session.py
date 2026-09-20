from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, Qt, pyqtSignal

from core.config import SCORE_THRESHOLD
from data.raven import BOX_COLUMNS, CALL, SCORE, SPECIES
from viewer.spectrogram import Waveform

ANNOTATIONS = "Annotations"
DETECTIONS = "Detections"
SOURCES = (ANNOTATIONS, DETECTIONS)
# Sobre el espectrograma magma (negro, púrpura, amarillo) verde y celeste se separan de la
# imagen y entre sí; el trazo lo confirma para quien no distingue colores: la verdad va
# entera, el modelo a rayas.
COLORS = {ANNOTATIONS: "#3ddc84", DETECTIONS: "#4fc3f7"}
STYLES = {ANNOTATIONS: Qt.PenStyle.SolidLine, DETECTIONS: Qt.PenStyle.DashLine}
WIDTHS = {ANNOTATIONS: 2, DETECTIONS: 2}


# Una tabla de Raven con columna `Score` la escribió un modelo y va a la capa de detecciones,
# donde el slider la filtra; sin ella es una anotación.
def read_table(path: Path) -> tuple[str, pd.DataFrame]:
    table = pd.read_csv(path, sep=None, engine="python")
    absent = [name for name in BOX_COLUMNS if name not in table.columns]
    if absent:
        raise ValueError(f"'{path.name}' is missing the columns: {', '.join(absent)}")
    return (DETECTIONS if SCORE in table.columns else ANNOTATIONS), table


# `especie/llamada`; una celda vacía (NaN en la tabla) no escribe "nan".
def label(row: pd.Series) -> str:
    parts = [row[c] for c in (SPECIES, CALL) if c in row.index]
    return "/".join(str(p) for p in parts if not pd.isna(p) and str(p).strip())


@dataclass(frozen=True)
class Row:
    source: str
    index: Hashable  # la etiqueta de la fila en su DataFrame
    begin: float
    end: float
    low: float
    high: float
    label: str
    score: float  # NaN cuando la tabla no trae `SCORE`


# Estado compartido entre el espectrograma y la tabla de cajas: quien mira las cajas las lee
# de aquí y quien las edita emite `changed`. El audio va primero: cargar otro vacía las capas.
class Session(QObject):
    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.audio_path: Path | None = None
        self.model_path: Path | None = None
        self.waveform: Waveform | None = None
        self.sr = 1
        self.score = SCORE_THRESHOLD
        self.tables: dict[str, pd.DataFrame | None] = dict.fromkeys(SOURCES)

    @property
    def duration(self) -> float:
        return 0.0 if self.waveform is None else self.waveform.size / self.sr

    def set_audio(self, path: Path, waveform: Waveform, sr: int) -> None:
        self.audio_path, self.waveform, self.sr = path, waveform, sr
        self.tables = dict.fromkeys(SOURCES)  # eran de otro audio
        self.changed.emit()

    def set_table(self, source: str, table: pd.DataFrame | None) -> None:
        self.tables[source] = table
        self.changed.emit()

    def set_score(self, score: float) -> None:
        self.score = score
        self.changed.emit()

    def visible(self, source: str) -> pd.DataFrame | None:
        table = self.tables[source]
        if table is None or SCORE not in table.columns:
            return table
        return table.loc[table[SCORE] >= self.score]

    # Una caja nueva al final de la tabla de `source`; si no había tabla, la crea con las
    # columnas de la caja y la clase. La etiqueta de fila sigue a la mayor que hubiera, así las
    # de las demás filas no cambian.
    def add(self, source: str, values: dict[str, object]) -> None:
        table = self.tables[source]
        if table is None:
            table = pd.DataFrame(columns=[*BOX_COLUMNS, SPECIES, CALL])
        next_label = int(table.index.max()) + 1 if len(table) else 0
        row = pd.DataFrame([values], index=[next_label])
        self.tables[source] = pd.concat([table, row])
        self.changed.emit()

    def remove(self, targets: list[tuple[str, Hashable]]) -> None:
        for source in SOURCES:
            indices = [index for target, index in targets if target == source]
            table = self.tables[source]
            if indices and table is not None:
                self.tables[source] = table.drop(index=indices)
        self.changed.emit()

    # Las cajas de un origen ordenadas por tiempo, que es como se revisan.
    def rows(self, source: str) -> list[Row]:
        table = self.visible(source)
        if table is None:
            return []
        boxes = table[BOX_COLUMNS].to_numpy(dtype=float)
        scores = (
            table[SCORE].to_numpy(dtype=float)
            if SCORE in table.columns
            else np.full(len(table), float("nan"))
        )
        rows = [
            Row(source, index, begin, end, low, high, label(row), score)
            for (index, row), (begin, end, low, high), score in zip(
                table.iterrows(), boxes, scores, strict=True
            )
        ]
        return sorted(rows, key=lambda row: row.begin)
