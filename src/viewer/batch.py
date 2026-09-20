"""Vista Batch: una carpeta de grabaciones, el modelo (la misma lista que en el espectrograma)
y un umbral; deja la tabla de Raven de cada audio junto a él (`<audio>.detections.txt`), igual
que `detect.exe`. Es una página aparte de la del espectrograma, con sus propios mandos: se ve
la lista entera, el estado de cada archivo y cuánto falta. El doble clic abre la grabación en
el espectrograma."""

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import override

import soundfile as sf
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from core.config import SCORE_THRESHOLD
from inference.catalog import collect_audio, output_for
from viewer.controls import ModelPicker, headline_font
from viewer.tasks import Worker

PENDING, EXISTS, RUNNING, DONE, SKIPPED, FAILED, STOPPED = (
    "pending",
    "exists",
    "running",
    "done",
    "skipped",
    "failed",
    "stopped",
)
STATUS_TEXT = {
    PENDING: "Pending",
    EXISTS: "Table exists",
    DONE: "Done",
    SKIPPED: "Skipped (table exists)",
    STOPPED: "Stopped",
}
HEADERS = ("File", "Duration", "Status", "Detections")
FILE_COLUMN, STATUS_COLUMN = 0, 2
ROW_HEIGHT = 22
# La barra global avanza también dentro del archivo en curso, en centésimas
PROGRESS_STEPS = 100
SCORE_RANGE, SCORE_STEP = (0.05, 1.0), 0.05
RUN_WIDTH = 96
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
    detections: int | None = None  # cajas de su tabla, si la tiene
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


# Cuántas cajas tiene una tabla ya escrita: filas menos la cabecera.
def rows_in(table: Path) -> int | None:
    try:
        with table.open(encoding="utf-8", errors="replace") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return None


# Lista los audios de la carpeta con su duración y lo que ya tienen; corre en un hilo porque
# en una carpeta de red leer mil cabeceras tarda.
def scan(folder: Path, recursive: bool) -> list[Entry]:
    entries = []
    for path in collect_audio([folder], recursive):
        table = output_for(path)
        entry = Entry(path, str(path.relative_to(folder)), duration_of(path))
        if table.is_file():
            entry.status, entry.detections = EXISTS, rows_in(table)
        entries.append(entry)
    return entries


class FileModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self.entries: list[Entry] = []

    def reset(self, entries: list[Entry]) -> None:
        self.beginResetModel()
        self.entries = entries
        self.endResetModel()

    def touch(self, index: int) -> None:
        self.dataChanged.emit(self.index(index, 0), self.index(index, len(HEADERS) - 1))

    @override
    def rowCount(self, parent=ROOT) -> int:
        return 0 if parent.isValid() else len(self.entries)

    @override
    def columnCount(self, parent=ROOT) -> int:
        return 0 if parent.isValid() else len(HEADERS)

    @override
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return HEADERS[section]
        if role == Qt.ItemDataRole.TextAlignmentRole and section not in (
            FILE_COLUMN,
            STATUS_COLUMN,
        ):
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    @override
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        entry = self.entries[index.row()]
        column = index.column()
        if role == Qt.ItemDataRole.ToolTipRole:
            return entry.message if column == STATUS_COLUMN and entry.message else str(entry.path)
        numeric = column not in (FILE_COLUMN, STATUS_COLUMN)
        if role == Qt.ItemDataRole.TextAlignmentRole and numeric:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if column == FILE_COLUMN:
            return entry.name
        if column == 1:
            return clock(entry.duration)
        if column == STATUS_COLUMN:
            if entry.status == RUNNING:
                return f"Running {entry.fraction:.0%}"
            if entry.status == FAILED:
                return f"Error: {entry.message}"
            return STATUS_TEXT[entry.status]
        return "" if entry.detections is None else f"{entry.detections:,}"


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
        self.engine_ready = False
        self.blocked = False  # el espectrograma está detectando: un modelo a la vez
        self.worker: BatchWorker | None = None
        self.scanner: Worker | None = None
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

        # Se escribe con este score, que arranca en el punto de operación del modelo (el que
        # eligió la comparación en val); es lo mismo que `detect.exe --score`.
        self.score = QDoubleSpinBox()
        self.score.setRange(*SCORE_RANGE)
        self.score.setSingleStep(SCORE_STEP)
        self.score.setDecimals(2)
        self.score.setValue(SCORE_THRESHOLD)
        self.score.setToolTip(
            "Only detections at or above this score are written. "
            "Starts at the model's operating point."
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
        self.table = QTableView()
        self.table.setModel(self.files)
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
        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(False)
            for column in range(1, len(HEADERS)):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(FILE_COLUMN, QHeaderView.ResizeMode.Stretch)

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
        settings.addWidget(QLabel("Score ≥"))
        settings.addWidget(self.score)
        settings.addSpacing(8)
        settings.addWidget(self.overwrite)
        settings.addStretch(1)
        settings.addWidget(self.run_button)
        settings.addWidget(self.stop_button)

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
        self.score.setValue(SCORE_THRESHOLD if operating_point is None else operating_point)
        self.sync()

    def set_engine(self, ready: bool) -> None:
        self.engine_ready = ready
        self.sync()

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
            and self.engine_ready
        )

    def why(self) -> str:
        if self.running():
            return "Batch is already running"
        if not self.listed():
            return "Open a folder of recordings first"
        if self.model_path is None:
            return "Choose a model first"
        if not self.engine_ready:
            return "Waiting for the detection engine"
        if self.blocked:
            return "Wait for the current detection to finish"
        return "Run the model over every recording in the list"

    def sync(self) -> None:
        running = self.running()
        listed = self.listed()
        for widget in (self.folder, self.browse_button, self.recursive, self.overwrite):
            widget.setEnabled(not running)
        self.score.setEnabled(not running)
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
        self.scanner = Worker(lambda: scan(folder, recursive))
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
            self.score.value(),
            self.overwrite.isChecked(),
        )
        self.worker.started_file.connect(self.on_started)
        self.worker.progressed.connect(self.on_progress)
        self.worker.finished_file.connect(self.on_outcome)
        self.worker.failed.connect(lambda message: self.said.emit(f"Model failed: {message}"))
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
        self.files.touch(index)
        self.table.scrollTo(self.files.index(index, 0))

    def on_progress(self, index: int, done: int, total: int) -> None:
        entry = self.files.entries[index]
        entry.fraction = done / max(total, 1)
        self.files.touch(index)
        self.progress.setValue(int((index + entry.fraction) * PROGRESS_STEPS))
        self.summary.setText(self.eta(index))
        self.summary.show()

    def on_outcome(self, outcome) -> None:
        entry = self.files.entries[outcome.index]
        entry.status, entry.fraction, entry.message = outcome.status, 1.0, outcome.message
        if outcome.status == DONE:
            entry.detections = outcome.detections
            if not math.isnan(entry.duration):
                self.done_audio_s += entry.duration
            self.finished_file.emit(entry.path)
        self.files.touch(outcome.index)
        self.progress.setValue((outcome.index + 1) * PROGRESS_STEPS)

    def on_finished(self) -> None:
        entries = self.files.entries
        done = [e for e in entries if e.status == DONE]
        failed = sum(e.status == FAILED for e in entries)
        skipped = sum(e.status == SKIPPED for e in entries)
        stopped = any(e.status == STOPPED for e in entries)
        detections = sum(e.detections or 0 for e in done)
        parts = [f"{len(done)} processed", f"{detections:,} detections"]
        if skipped:
            parts.append(f"{skipped} skipped")
        if failed:
            parts.append(f"{failed} failed")
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
        head = f"{current + 1} / {len(entries)}"
        if elapsed < ETA_AFTER_S or processed <= 0.0:
            return head
        rate = processed / elapsed
        return f"{head} · {rate:.1f}× realtime · ~{clock(remaining / rate)} left"

    def queued(self, index: int) -> bool:
        # Lo que aún va a procesarse: lo pendiente y, con Overwrite, lo que ya tiene tabla.
        status = self.files.entries[index].status
        return status == PENDING or (status == EXISTS and self.overwrite.isChecked())
