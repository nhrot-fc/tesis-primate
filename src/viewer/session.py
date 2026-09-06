from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal

from viewer.spectrogram import Waveform

ANNOTATIONS = "Anotaciones"
DETECTIONS = "Modelo"
SOURCES = (ANNOTATIONS, DETECTIONS)
COLORS = {ANNOTATIONS: "#00d8ff", DETECTIONS: "#8cff3d"}

BEGIN, END, LOW, HIGH = "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"
BOX_COLUMNS = [BEGIN, END, LOW, HIGH]


def read_boxes(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    missing = [column for column in BOX_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"'{path.name}' no tiene las columnas: {', '.join(missing)}")
    return table


def label(row: pd.Series) -> str:
    return "/".join(str(row[c]) for c in ("Species", "Call type") if c in row.index)


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
        self.changed.emit()

    def set_score(self, score: float) -> None:
        self.score = score
        self.changed.emit()

    def visible(self, source: str) -> pd.DataFrame | None:
        table = self.tables[source]
        if table is None or "Score" not in table.columns:
            return table
        return table.loc[table["Score"] >= self.score]

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
