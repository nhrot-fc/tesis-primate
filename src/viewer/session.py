import csv
from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, Qt, pyqtSignal

from core.config import SCORE_THRESHOLD
from data.raven import BOX_COLUMNS, CALL, SCORE, SPECIES, VIEW
from viewer.spectrogram import Waveform
from viewer.tasks import UnreadableError

ANNOTATIONS = "Annotations"
DETECTIONS = "Detections"
SOURCES = (ANNOTATIONS, DETECTIONS)
# Sobre el espectrograma magma (negro, púrpura, amarillo) verde y celeste se separan de la
# imagen y entre sí; el trazo lo confirma para quien no distingue colores: la verdad va
# entera, el modelo a rayas.
COLORS = {ANNOTATIONS: "#3ddc84", DETECTIONS: "#4fc3f7"}
STYLES = {ANNOTATIONS: Qt.PenStyle.SolidLine, DETECTIONS: Qt.PenStyle.DashLine}
WIDTHS = {ANNOTATIONS: 2, DETECTIONS: 2}
# Raven escribe en UTF-8 o, en Windows, en la codificación del sistema; Excel le pone BOM.
ENCODINGS = ("utf-8-sig", "cp1252")
NOT_A_TABLE = (
    "'{name}' is not a Raven selection table: a tab-separated text file with the columns "
    "Begin Time (s), End Time (s), Low Freq (Hz) and High Freq (Hz)."
)


# Una tabla de Raven con columna `Score` la escribió un modelo y va a la capa de detecciones,
# donde el slider la filtra; sin ella es una anotación. Lo que no se puede dibujar se dice con
# el nombre del archivo y qué tiene mal: lo arregla quien la hizo, no quien programó esto.
def read_table(path: Path) -> tuple[str, pd.DataFrame]:
    table = parse(path)
    absent = [name for name in BOX_COLUMNS if name not in table.columns]
    if absent:
        raise UnreadableError(
            NOT_A_TABLE.format(name=path.name) + f" It has no {', '.join(absent)}."
        )
    for column in [*BOX_COLUMNS, *([SCORE] if SCORE in table.columns else [])]:
        table[column] = numeric(table[column], column, path)
    # Con el oscilograma abierto, Raven escribe cada selección dos veces, una por vista: la del
    # espectrograma es la que lleva la banda.
    if VIEW in table.columns:
        spectrogram = ~table[VIEW].astype(str).str.startswith("Waveform")
        if spectrogram.any():
            table = table.loc[spectrogram]
    # Una caja sin tiempo o sin banda no se puede dibujar: queda fuera y se avisa cuántas.
    complete = np.isfinite(table[BOX_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    table = table.loc[complete].reset_index(drop=True)
    table.attrs["skipped"] = int((~complete).sum())
    return (DETECTIONS if SCORE in table.columns else ANNOTATIONS), table


def parse(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        raise UnreadableError(f"'{path.name}' is empty.")
    for encoding in ENCODINGS:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
        except (ValueError, csv.Error) as exc:
            raise UnreadableError(NOT_A_TABLE.format(name=path.name)) from exc
    raise UnreadableError(NOT_A_TABLE.format(name=path.name) + " It is not even text.")


# Excel en castellano guarda "1,5": la coma decimal se acepta. Lo demás que no sea un número se
# señala con su fila, que es lo que hay que buscar para corregirlo.
def numeric(values: pd.Series, column: str, path: Path) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    text = values.astype(str).str.strip().str.replace(",", ".", regex=False)
    converted = pd.to_numeric(text.where(values.notna()), errors="coerce")
    wrong = (converted.isna() & values.notna() & (text != "")).to_numpy()
    if wrong.any():
        row = int(wrong.argmax())
        raise UnreadableError(
            f"'{path.name}': {column} has '{values.iloc[row]}' in row {row + 1}, "
            "which is not a number."
        )
    return converted.astype(float)


# `especie/llamada`; una celda vacía (NaN en la tabla) no escribe "nan".
def label(row: pd.Series) -> str:
    parts = [row[c] for c in (SPECIES, CALL) if c in row.index]
    return "/".join(str(p) for p in parts if not pd.isna(p) and str(p).strip())


# `label` de todas las filas de una vez: armar una Series por fila tarda con miles de cajas.
def labels(table: pd.DataFrame) -> list[str]:
    columns = [table[c] for c in (SPECIES, CALL) if c in table.columns]
    if not columns:
        return [""] * len(table)
    return [
        "/".join(str(p) for p in parts if not pd.isna(p) and str(p).strip())
        for parts in zip(*columns, strict=True)
    ]


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

    # Una fila sin score (agregada a mano en Raven) se ve con cualquier umbral, como cuenta Batch.
    def visible(self, source: str) -> pd.DataFrame | None:
        table = self.tables[source]
        if table is None or SCORE not in table.columns:
            return table
        return table.loc[table[SCORE].isna() | (table[SCORE] >= self.score)]

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
            Row(source, index, begin, end, low, high, text, score)
            for index, (begin, end, low, high), text, score in zip(
                table.index, boxes, labels(table), scores, strict=True
            )
        ]
        return sorted(rows, key=lambda row: row.begin)
