import logging
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd
import soundfile as sf
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from core.config import SCORE_THRESHOLD, score_grid
from data.raven import CALL, SCORE, SPECIES
from inference.catalog import DETECTIONS_SUFFIX, collect_audio, output_for
from viewer.controls import ModelPicker, Slider, headline_font
from viewer.inference import DETECT_THRESHOLD
from viewer.tasks import Worker

logger = logging.getLogger("viewer")

PENDING, EXISTS, RUNNING, DONE, SKIPPED, FAILED, STOPPED = (
    "pending",
    "exists",
    "running",
    "done",
    "skipped",
    "failed",
    "stopped",
)
# Desdobladas, después de estas va una columna por clase del modelo (`ESPECIE/LLAMADA`)
HEADERS = ("File", "Duration", "Detections")
FILE_COLUMN, DURATION_COLUMN, TOTAL_COLUMN = 0, 1, 2
# Una caja sin especie ni llamada (una tabla editada a mano)
UNLABELED = "(no label)"
# El CSV lleva la duración en segundos, que es lo que se suma en una hoja de cálculo
DURATION_SECONDS = "Duration (s)"
ROW_HEIGHT = 22
# Margen de una celda a cada lado del texto, para los anchos fijos
CELL_PADDING = 16
RIGHT = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
# La barra global avanza también dentro del archivo en curso, en centésimas
PROGRESS_STEPS = 100
RUN_WIDTH = 96
SCORE_WIDTH = 240
# Segundos de corrida antes de fiarse de la velocidad medida para el tiempo restante
ETA_AFTER_S = 5.0
ROOT = QModelIndex()

HEADLINE = "Drop a folder here"
PLACEHOLDER = (
    "Every recording in it gets a Raven table next to it, ready for Raven Pro or the "
    "Spectrogram view."
)
HINT = "Double-click a recording to open it in the Spectrogram view."


@dataclass
class Entry:
    path: Path
    name: str  # relativo a la carpeta elegida
    duration: float  # NaN si no se pudo leer la cabecera
    status: str = PENDING
    fraction: float = 0.0  # avance del archivo en curso
    # Los scores de su tabla por etiqueta, de menor a mayor; None si no tiene tabla
    found: dict[str, np.ndarray] | None = None
    message: str = ""


def clock(seconds: float) -> str:
    if math.isnan(seconds):
        return "?"
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def duration_of(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return float("nan")


# En mayúscula, como las escribe el modelo: `lw` y `LW` de una tabla retocada cuentan juntas.
def text_column(table: pd.DataFrame, name: str) -> list[str]:
    if name not in table.columns:
        return [""] * len(table)
    return table[name].fillna("").astype(str).str.strip().str.upper().tolist()


# Los scores de una tabla ya escrita por `ESPECIE/LLAMADA`, ordenados: contar lo que pasa un
# umbral es una búsqueda binaria por etiqueta. Sin score (una tabla hecha a mano, o una fila
# que se agregó en Raven) la caja cuenta siempre.
def boxes_in(table: Path) -> dict[str, np.ndarray] | None:
    try:
        frame = pd.read_csv(table, sep="\t")
        scores = (
            frame[SCORE].astype(float).fillna(math.inf).to_numpy()
            if SCORE in frame.columns
            else np.full(len(frame), math.inf)
        )
    except (OSError, ValueError) as exc:
        # La grabación queda sin conteo, como si no tuviera tabla; el log dice por qué.
        logger.warning("Could not read %s: %s", table, exc)
        return None
    labels = [
        "/".join(part for part in parts if part) or UNLABELED
        for parts in zip(text_column(frame, SPECIES), text_column(frame, CALL), strict=True)
    ]
    by_label = pd.Series(scores, index=labels, dtype=float)
    return {str(label): np.sort(group.to_numpy()) for label, group in by_label.groupby(level=0)}


def count(found: dict[str, np.ndarray], score: float) -> Counter[str]:
    return Counter(
        {
            label: scores.size - int(np.searchsorted(scores, score))
            for label, scores in found.items()
        }
    )


# Lista los audios de la carpeta con su duración y lo que ya tienen; corre en un hilo porque
# en una carpeta de red leer mil cabeceras tarda.
def scan(folder: Path, recursive: bool) -> list[Entry]:
    entries = []
    for path in collect_audio([folder], recursive):
        table = output_for(path)
        entry = Entry(path, str(path.relative_to(folder)), duration_of(path))
        if table.is_file():
            entry.status, entry.found = EXISTS, boxes_in(table)
        entries.append(entry)
    return entries


# Una fila por grabación con cuántas detecciones de su tabla pasan el score; desdoblada, una
# columna más por cada clase del modelo (y por cualquier otra etiqueta que traigan las
# tablas), en orden de especie. El encabezado de cada conteo lleva el total de la carpeta. Sin
# tabla las celdas quedan vacías; los ceros de las clases también, para que resalte lo hallado.
class FileModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self.entries: list[Entry] = []
        self.score = SCORE_THRESHOLD
        self.classes: list[str] = []  # las del modelo elegido, como `LW/CS`
        self.expanded = False
        self.labels: list[str] = []  # las columnas desdobladas; el CSV las lleva siempre
        self.counts: list[Counter[str] | None] = []
        self.totals: Counter[str] = Counter()
        self.ceilings: Counter[str] = Counter()  # todas las cajas, sin umbral: el ancho máximo

    def tally(self) -> tuple[list[Counter[str] | None], Counter[str]]:
        counts = [
            None if entry.found is None else count(entry.found, self.score)
            for entry in self.entries
        ]
        totals: Counter[str] = Counter()
        for counted in counts:
            totals.update(counted or {})
        return counts, totals

    def reset(self, entries: list[Entry] | None = None) -> None:
        self.beginResetModel()
        if entries is not None:
            self.entries = entries
        found = {label for entry in self.entries for label in entry.found or {}}
        self.labels = sorted(found | set(self.classes))
        self.counts, self.totals = self.tally()
        self.ceilings = Counter()
        for entry in self.entries:
            self.ceilings.update(count(entry.found or {}, -math.inf))
        self.endResetModel()

    # Mover el score no cambia las columnas, sólo los números.
    def recount(self) -> None:
        self.counts, self.totals = self.tally()
        last = self.columnCount() - 1
        if self.entries:
            self.dataChanged.emit(
                self.index(0, TOTAL_COLUMN), self.index(len(self.entries) - 1, last)
            )
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, TOTAL_COLUMN, last)

    # Un archivo terminó con tabla nueva: si trae una etiqueta que no es columna, se rehace.
    def update(self, index: int) -> None:
        found = self.entries[index].found or {}
        if set(found) - set(self.labels):
            self.reset()
        else:
            self.recount()

    def set_score(self, score: float) -> None:
        self.score = score
        self.recount()

    def set_classes(self, classes: list[str]) -> None:
        self.classes = [name.upper() for name in classes]
        self.reset()

    def set_expanded(self, expanded: bool) -> None:
        self.expanded = expanded
        self.reset()

    def shown(self) -> list[str]:
        return self.labels if self.expanded else []

    # Una fila por grabación al score elegido, con todas las clases aunque la vista esté
    # plegada; sin tabla los conteos quedan vacíos (no 0).
    def table(self) -> pd.DataFrame:
        columns = [HEADERS[FILE_COLUMN], DURATION_SECONDS, HEADERS[TOTAL_COLUMN], *self.labels]
        rows = [
            [
                entry.name,
                round(entry.duration, 2),
                None if counted is None else counted.total(),
                *(None if counted is None else counted[label] for label in self.labels),
            ]
            for entry, counted in zip(self.entries, self.counts, strict=True)
        ]
        table = pd.DataFrame(rows, columns=columns)
        return table.astype(dict.fromkeys(columns[2:], "Int64"))

    @override
    def rowCount(self, parent=ROOT) -> int:
        return 0 if parent.isValid() else len(self.entries)

    @override
    def columnCount(self, parent=ROOT) -> int:
        return 0 if parent.isValid() else len(HEADERS) + len(self.shown())

    @override
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        counted = section >= TOTAL_COLUMN
        label = self.labels[section - len(HEADERS)] if section >= len(HEADERS) else None
        total = self.totals.total() if label is None else self.totals[label]
        if role == Qt.ItemDataRole.DisplayRole:
            if not counted or not any(c is not None for c in self.counts):
                return HEADERS[section]
            return f"{label or HEADERS[section]}\n{total:,}"
        if role == Qt.ItemDataRole.ToolTipRole and counted:
            files = sum(1 for c in self.counts if c and (c[label] if label else c.total()))
            return (
                f"{label or HEADERS[section]} at Score ≥ {self.score:.2f}: {total:,} "
                f"in {files} of {len(self.entries)} recordings"
            )
        if role == Qt.ItemDataRole.TextAlignmentRole and section != FILE_COLUMN:
            return RIGHT
        return None

    @override
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        entry = self.entries[index.row()]
        column = index.column()
        # Sin columna de estado: un archivo que falló, o cuya tabla no se pudo leer, lo dice
        # al pasar el ratón.
        if role == Qt.ItemDataRole.ToolTipRole:
            if entry.status == FAILED:
                return f"{entry.path}\nFailed: {entry.message}"
            if entry.status in (EXISTS, SKIPPED) and entry.found is None:
                return f"{entry.path}\nIts {DETECTIONS_SUFFIX} could not be read"
            return str(entry.path)
        if role == Qt.ItemDataRole.TextAlignmentRole and column != FILE_COLUMN:
            return RIGHT
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if column == FILE_COLUMN:
            return entry.name
        if column == DURATION_COLUMN:
            return clock(entry.duration)
        counted = self.counts[index.row()]
        if counted is None:
            return ""
        if column == TOTAL_COLUMN:
            return f"{counted.total():,}"
        found = counted[self.labels[column - len(HEADERS)]]
        return f"{found:,}" if found else ""


# Anchos fijos, calculados con la fuente cuando cambian las columnas: con `ResizeToContents`
# Qt vuelve a medir las filas en cada aviso de progreso y la ventana se traba durante la
# corrida. Cada conteo se mide con lo más que podría llegar a decir (todas sus cajas). El
# nombre se estira si todo cabe; si no, queda a su ancho y la tabla se desplaza de lado.
class FileTable(QTableView):
    def __init__(self, files: FileModel) -> None:
        super().__init__()
        self.files = files
        self.setModel(files)
        self.name_width = 0
        self.widths: list[int] = []  # de Duration en adelante
        files.modelReset.connect(self.measure)
        self.measure()

    def measure(self) -> None:
        header = self.horizontalHeader()
        if header is None:
            return
        cell, head = self.fontMetrics(), header.fontMetrics()

        def width(texts: list[str], heading: str) -> int:
            widest = max((cell.horizontalAdvance(text) for text in texts), default=0)
            return max(widest, head.horizontalAdvance(heading)) + CELL_PADDING

        files = self.files
        self.name_width = width([e.name for e in files.entries], HEADERS[FILE_COLUMN])
        self.widths = [
            width(["0:00:00"], HEADERS[DURATION_COLUMN]),
            width([f"{files.ceilings.total():,}"], HEADERS[TOTAL_COLUMN]),
            *(width([f"{files.ceilings[label]:,}"], label) for label in files.shown()),
        ]
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column, size in enumerate(self.widths, start=DURATION_COLUMN):
            header.resizeSection(column, size)
        self.fit()

    def fit(self) -> None:
        header, viewport = self.horizontalHeader(), self.viewport()
        if header is None or viewport is None:
            return
        if self.name_width + sum(self.widths) <= viewport.width():
            header.setSectionResizeMode(FILE_COLUMN, QHeaderView.ResizeMode.Stretch)
        else:
            header.setSectionResizeMode(FILE_COLUMN, QHeaderView.ResizeMode.Interactive)
            header.resizeSection(FILE_COLUMN, self.name_width)

    @override
    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self.fit()


# Un hilo para toda la lista: carga el modelo (o lo toma del caché del visor) y recorre los
# archivos con `run_batch`. Parar es una bandera que el recorrido consulta entre lotes.
class BatchWorker(QThread):
    started_file = pyqtSignal(int)
    progressed = pyqtSignal(int, int, int)
    finished_file = pyqtSignal(object)  # inference.batch.Outcome
    failed = pyqtSignal(str)  # el modelo no cargó: no se procesó nada

    def __init__(self, files: list[Path], checkpoint: Path, threshold: float, overwrite: bool):
        super().__init__()
        self.files = files
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.overwrite = overwrite
        self.stopping = threading.Event()

    def stop(self) -> None:
        self.stopping.set()

    @override
    def run(self) -> None:
        from inference.batch import run_batch
        from viewer.inference import load

        try:
            loaded, device = load(self.checkpoint)
        except Exception as exc:
            logger.exception("Could not load the model %s", self.checkpoint)
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        current = -1

        def report(index: int, done: int, total: int) -> None:
            nonlocal current
            if index != current:
                current = index
                self.started_file.emit(index)
            self.progressed.emit(index, done, total)

        for outcome in run_batch(
            loaded,
            self.files,
            device,
            self.threshold,
            self.overwrite,
            on_progress=report,
            should_stop=self.stopping.is_set,
        ):
            self.finished_file.emit(outcome)


class BatchView(QWidget):
    open_requested = pyqtSignal(object)  # Path del audio
    finished_file = pyqtSignal(object)  # Path con tabla nueva
    state_changed = pyqtSignal()  # empezó o terminó una corrida
    said = pyqtSignal(str)  # para la barra de estado

    def __init__(self) -> None:
        super().__init__()
        self.folder_path: Path | None = None
        self.model_path: Path | None = None
        self.engine_ready: bool | None = None  # None mientras carga; False si no cargó
        self.model_error = ""  # el modelo de la última corrida no cargó
        self.blocked = False  # el espectrograma está detectando: un modelo a la vez
        self.worker: BatchWorker | None = None
        self.scanner: Worker | None = None
        self.reader: Worker | None = None  # lee las clases del modelo elegido
        self.started_at = 0.0
        self.done_audio_s = 0.0

        self.folder = QLineEdit()
        self.folder.setPlaceholderText("Folder with recordings (WAV, FLAC, MP3)")
        self.folder.returnPressed.connect(lambda: self.set_folder(Path(self.folder.text())))
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self.browse)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(True)
        self.recursive.toggled.connect(lambda _: self.rescan())

        # El modelo es el mismo que el del espectrograma: la ventana mantiene los dos
        # selectores iguales. Acá hay uno para no tener que cambiar de vista para elegirlo.
        self.picker = ModelPicker()

        # El mismo slider que en el espectrograma: Run escribe todo lo que pasa DETECT_THRESHOLD
        # y el score sólo filtra lo que cuenta la tabla, así que moverlo no vuelve a detectar.
        # Arranca en el punto de operación del modelo (el que eligió la comparación en val).
        thresholds = score_grid(DETECT_THRESHOLD, 1.0)
        self.score = Slider(
            "Score ≥", thresholds, thresholds.index(SCORE_THRESHOLD), "{:.2f}", label_width=52
        )
        self.score.setMaximumWidth(SCORE_WIDTH)
        self.score.setToolTip(
            "Counts only the detections at or above this score. The tables keep every "
            "detection, so moving it does not run the model again."
        )
        self.overwrite = QCheckBox("Overwrite existing tables")
        self.run_button = QPushButton("Run")
        self.run_button.setMinimumWidth(RUN_WIDTH)
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumWidth(RUN_WIDTH)
        self.stop_button.clicked.connect(self.stop)
        for button in (self.browse_button, self.run_button, self.stop_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.files = FileModel()
        self.files.set_score(self.score.value())
        self.score.changed.connect(lambda: self.files.set_score(self.score.value()))
        self.table = FileTable(self.files)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.doubleClicked.connect(
            lambda index: self.open_requested.emit(self.files.entries[index.row()].path)
        )
        vertical = self.table.verticalHeader()
        if vertical is not None:
            vertical.setVisible(False)
            vertical.setDefaultSectionSize(ROW_HEIGHT)

        # Sin carpeta: una frase, una aclaración y el botón, como la bienvenida del espectrograma.
        headline = QLabel(HEADLINE)
        headline.setFont(headline_font(headline))
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder = QLabel(PLACEHOLDER)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setWordWrap(True)
        self.choose_button = QPushButton("Choose a folder…")
        self.choose_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.choose_button.clicked.connect(self.browse)
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.setContentsMargins(40, 0, 40, 0)
        empty_layout.addStretch(1)
        empty_layout.addWidget(headline)
        empty_layout.addSpacing(6)
        empty_layout.addWidget(placeholder)
        empty_layout.addSpacing(22)
        empty_layout.addWidget(self.choose_button, 0, Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch(1)
        self.pages = QStackedWidget()
        self.pages.addWidget(empty)
        self.pages.addWidget(self.table)

        # Plegada, la tabla sólo cuenta detecciones; desdoblada, una columna por clase.
        self.expand = QCheckBox("Count per species and call")
        self.expand.setToolTip("One column per class of the model, ordered by species")
        self.expand.toggled.connect(self.files.set_expanded)
        self.export_button = QPushButton("Export CSV…")
        self.export_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.export_button.setToolTip(
            "One row per recording: its detections at this score, in total and per class"
        )
        self.export_button.clicked.connect(self.export)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.summary = QLabel("")
        self.hint = QLabel(HINT)

        source = QHBoxLayout()
        source.setSpacing(8)
        source.addWidget(QLabel("Folder"))
        source.addWidget(self.folder, 1)
        source.addWidget(self.browse_button)
        source.addSpacing(8)
        source.addWidget(self.recursive)

        settings = QHBoxLayout()
        settings.setSpacing(8)
        settings.addWidget(QLabel("Model"))
        settings.addWidget(self.picker)
        settings.addSpacing(8)
        settings.addWidget(self.score, 1)
        settings.addSpacing(8)
        settings.addWidget(self.overwrite)
        settings.addStretch(1)
        settings.addWidget(self.run_button)
        settings.addWidget(self.stop_button)

        counts = QHBoxLayout()
        counts.setSpacing(8)
        counts.addWidget(self.expand)
        counts.addStretch(1)
        counts.addWidget(self.export_button)

        footer = QHBoxLayout()
        footer.setSpacing(12)
        footer.addWidget(self.progress, 1)
        footer.addWidget(self.summary)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        layout.addLayout(source)
        layout.addLayout(settings)
        layout.addWidget(self.pages, 1)
        layout.addLayout(counts)
        layout.addLayout(footer)
        layout.addWidget(self.hint)
        self.sync()

    # --- Estado -----------------------------------------------------------------

    def running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def scanning(self) -> bool:
        return self.scanner is not None and self.scanner.isRunning()

    def listed(self) -> bool:
        return bool(self.files.entries)

    def set_model(self, path: Path | None, operating_point: float | None) -> None:
        self.model_path = path
        self.score.set_value(SCORE_THRESHOLD if operating_point is None else operating_point)
        self.read_classes()
        self.sync()

    def set_engine(self, ready: bool | None) -> None:
        self.engine_ready = ready
        self.read_classes()
        self.sync()

    # Las columnas desdobladas son las clases del modelo, aunque ninguna tabla las tenga
    # todavía. Leerlas pide torch, así que se espera al motor y se hace en un hilo.
    def read_classes(self) -> None:
        path = self.model_path
        if path is None:
            self.files.set_classes([])
            return
        if not self.engine_ready:
            return
        from viewer.inference import classes

        self.reader = Worker(lambda: classes(path), what=f"Reading the classes of {path}")
        self.reader.ok.connect(
            lambda names: self.files.set_classes(names) if path == self.model_path else None
        )
        self.reader.error.connect(
            lambda message: self.said.emit(f"Could not read the model's classes: {message}")
        )
        self.reader.start()

    def set_blocked(self, blocked: bool) -> None:
        self.blocked = blocked
        self.sync()

    # Si Run puede correr ahora, y por qué no: la ventana lo usa para su propio menú, así que
    # la razón se escribe una vez y se lee desde los dos sitios.
    def can_run(self) -> bool:
        return (
            not self.running()
            and not self.scanning()
            and not self.blocked
            and self.listed()
            and self.model_path is not None
            and self.engine_ready is True
        )

    def why(self) -> str:
        if self.running():
            return "Batch is already running"
        if not self.listed():
            return "Open a folder of recordings first"
        if self.model_path is None:
            return "Choose a model first"
        if self.engine_ready is None:
            return "Waiting for the detection engine"
        if not self.engine_ready:
            return "The detection engine did not load: see viewer.log"
        if self.blocked:
            return "Wait for the current detection to finish"
        return "Run the model over every recording in the list"

    def sync(self) -> None:
        running = self.running()
        listed = self.listed()
        for widget in (self.folder, self.browse_button, self.recursive, self.overwrite):
            widget.setEnabled(not running)
        self.export_button.setEnabled(listed and not running)
        self.picker.setEnabled(not running and not self.blocked)
        self.run_button.setEnabled(self.can_run())
        self.run_button.setToolTip(self.why())
        # Un solo botón a la vista: Run, que mientras corre es Stop.
        self.run_button.setVisible(not running)
        self.stop_button.setVisible(running)
        self.stop_button.setEnabled(running)
        self.pages.setCurrentWidget(self.table if listed else self.pages.widget(0))
        self.progress.setVisible(running)
        self.summary.setVisible(bool(self.summary.text()))
        self.hint.setVisible(listed)

    # --- Carpeta ----------------------------------------------------------------

    def browse(self) -> None:
        start = str(self.folder_path) if self.folder_path else ""
        chosen = QFileDialog.getExistingDirectory(self, "Choose a folder with recordings", start)
        if chosen:
            self.set_folder(Path(chosen))

    def set_folder(self, folder: Path) -> None:
        if self.running():
            return
        if not folder.is_dir():
            self.said.emit(f"{folder} is not a folder.")
            return
        self.folder_path = folder
        self.folder.setText(str(folder))
        self.rescan()

    def rescan(self) -> None:
        if self.folder_path is None or self.running() or self.scanning():
            return
        folder, recursive = self.folder_path, self.recursive.isChecked()
        self.said.emit(f"Scanning {folder}…")
        self.scanner = Worker(lambda: scan(folder, recursive), what=f"Scanning {folder}")
        self.scanner.ok.connect(self.scanned)
        self.scanner.error.connect(lambda message: self.said.emit(f"Could not scan: {message}"))
        self.scanner.finished.connect(self.sync)
        self.scanner.start()
        self.sync()

    def scanned(self, entries: list[Entry]) -> None:
        self.files.reset(entries)
        self.summary.setText("")
        existing = sum(e.status == EXISTS for e in entries)
        total_s = sum(e.duration for e in entries if not math.isnan(e.duration))
        self.said.emit(
            f"{len(entries)} recordings · {clock(total_s)} of audio"
            + (f" · {existing} already have a table" if existing else "")
            if entries
            else "No recordings in that folder."
        )
        self.sync()
        self.state_changed.emit()

    # --- Exportar ---------------------------------------------------------------

    # La tabla con todas las clases, al score de ahora, que va en el nombre: el CSV no lo lleva.
    def export(self) -> None:
        folder = self.folder_path
        if folder is None or not self.listed() or self.running():
            return
        name = f"{folder.name}_counts_score{self.score.value():.2f}.csv"
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Export counts", str(folder / name), "CSV (*.csv)"
        )
        if not chosen:
            return
        path = Path(chosen)
        table = self.files.table()
        try:
            table.to_csv(path, index=False)
        except Exception as exc:
            logger.exception("Could not save the counts %s", path)
            QMessageBox.critical(
                self, "Error", f"Could not save the counts:\n{type(exc).__name__}: {exc}"
            )
        else:
            self.said.emit(f"{len(table)} recordings saved to {path.name}")

    # --- Corrida ----------------------------------------------------------------

    def run(self) -> None:
        if not self.can_run() or self.model_path is None:
            self.said.emit(self.why())
            return
        for entry in self.files.entries:
            entry.status = EXISTS if output_for(entry.path).is_file() else PENDING
            entry.fraction, entry.message = 0.0, ""
        self.files.reset(self.files.entries)
        self.started_at = time.perf_counter()
        self.done_audio_s = 0.0
        self.progress.setRange(0, len(self.files.entries) * PROGRESS_STEPS)
        self.progress.setValue(0)
        self.worker = BatchWorker(
            [e.path for e in self.files.entries],
            self.model_path,
            DETECT_THRESHOLD,
            self.overwrite.isChecked(),
        )
        self.worker.started_file.connect(self.on_started)
        self.worker.progressed.connect(self.on_progress)
        self.worker.finished_file.connect(self.on_outcome)
        self.model_error = ""
        self.worker.failed.connect(self.on_model_failed)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.said.emit(
            f"Running {self.model_path.parent.name} over {len(self.files.entries)} recordings…"
        )
        self.sync()
        self.state_changed.emit()

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.said.emit("Stopping after the current batch of windows…")

    def on_started(self, index: int) -> None:
        entry = self.files.entries[index]
        entry.status, entry.fraction = RUNNING, 0.0

    def on_progress(self, index: int, done: int, total: int) -> None:
        entry = self.files.entries[index]
        entry.fraction = done / max(total, 1)
        self.progress.setValue(int((index + entry.fraction) * PROGRESS_STEPS))
        self.summary.setText(self.eta(index))
        self.summary.show()

    def on_outcome(self, outcome) -> None:
        entry = self.files.entries[outcome.index]
        entry.status, entry.fraction, entry.message = outcome.status, 1.0, outcome.message
        if outcome.status == DONE:
            entry.found = boxes_in(output_for(entry.path))
            if not math.isnan(entry.duration):
                self.done_audio_s += entry.duration
            self.files.update(outcome.index)
            self.finished_file.emit(entry.path)
        self.progress.setValue((outcome.index + 1) * PROGRESS_STEPS)

    # Sin modelo no se procesó nada: se avisa con un diálogo y el cierre de la corrida no lo
    # tapa con un "Done".
    def on_model_failed(self, message: str) -> None:
        self.model_error = message
        name = self.model_path.parent.name if self.model_path is not None else "the model"
        QMessageBox.critical(self, "Model failed", f"Could not load {name}:\n{message}")

    def on_finished(self) -> None:
        if self.model_error:
            self.summary.setText("")
            self.said.emit(f"Model failed: {self.model_error}")
            self.sync()
            self.state_changed.emit()
            return
        entries = self.files.entries
        done = [e for e in entries if e.status == DONE]
        failed = sum(e.status == FAILED for e in entries)
        skipped = sum(e.status == SKIPPED for e in entries)
        stopped = any(e.status == STOPPED for e in entries)
        score = self.score.value()
        detections = sum(count(e.found, score).total() for e in done if e.found is not None)
        parts = [f"{len(done)} processed", f"{detections:,} detections at score ≥ {score:.2f}"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed (hover a file for the error; details in viewer.log)")
        elapsed = time.perf_counter() - self.started_at
        self.summary.setText(f"{'Stopped' if stopped else 'Done'} in {clock(elapsed)}")
        self.said.emit(("Stopped: " if stopped else "Done: ") + " · ".join(parts))
        self.sync()
        self.state_changed.emit()

    # Velocidad medida en segundos de audio por segundo de reloj y lo que falta a ese ritmo.
    def eta(self, current: int) -> str:
        entries = self.files.entries
        elapsed = time.perf_counter() - self.started_at
        durations = [0.0 if math.isnan(e.duration) else e.duration for e in entries]
        processed = self.done_audio_s + entries[current].fraction * durations[current]
        remaining = (1.0 - entries[current].fraction) * durations[current] + sum(
            durations[i] for i in range(current + 1, len(entries)) if self.queued(i)
        )
        head = f"{current + 1} / {len(entries)} · {entries[current].name}"
        if elapsed < ETA_AFTER_S or processed <= 0.0:
            return head
        rate = processed / elapsed
        return f"{head} · {rate:.1f}× realtime · ~{clock(remaining / rate)} left"

    def queued(self, index: int) -> bool:
        # Lo que aún va a procesarse: lo pendiente y, con Overwrite, lo que ya tiene tabla.
        status = self.files.entries[index].status
        return status == PENDING or (status == EXISTS and self.overwrite.isChecked())
