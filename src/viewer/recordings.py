"""Panel lateral con las grabaciones de una carpeta: se elige una y se abre, y `Detect all`
corre el modelo sobre toda la lista dejando `<audio>.detections.txt` junto a cada una. Es la
vista Batch de antes convertida en navegador, como la lista de mensajes de un cliente de correo:
no hay que cambiar de vista para pasar de una grabación a la siguiente."""

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
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from core.config import SCORE_THRESHOLD
from inference.catalog import collect_audio, output_for
from viewer.controls import emphasize
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
HEADERS = ("Recording", "Length", "Boxes")
FILE_COLUMN, BOXES_COLUMN = 0, 2
ROW_HEIGHT = 22
PANEL_WIDTH = 300
# La barra global avanza también dentro del archivo en curso, en centésimas
PROGRESS_STEPS = 100
SCORE_RANGE, SCORE_STEP = (0.05, 1.0), 0.05
# Segundos de corrida antes de fiarse de la velocidad medida para el tiempo restante
ETA_AFTER_S = 5.0
ROOT = QModelIndex()
# Teclas que la lista se queda: moverse por las grabaciones. Las demás van a la ventana.
LIST_KEYS = {
    Qt.Key.Key_Up,
    Qt.Key.Key_Down,
    Qt.Key.Key_PageUp,
    Qt.Key.Key_PageDown,
    Qt.Key.Key_Home,
    Qt.Key.Key_End,
}

PLACEHOLDER = (
    "<div style='font-size:12pt'>Drop a folder here</div>"
    "<div style='margin-top:6px'>Every recording in it will be listed; "
    "<b>Detect all</b> leaves a <tt>.detections.txt</tt> next to each one.</div>"
)


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
def scan(folder: Path) -> list[Entry]:
    entries = []
    for path in collect_audio([folder]):
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
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return HEADERS[section]
        return None

    @override
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        entry = self.entries[index.row()]
        column = index.column()
        if role == Qt.ItemDataRole.ToolTipRole:
            if column == BOXES_COLUMN and entry.message:
                return entry.message
            return str(entry.path)
        if role == Qt.ItemDataRole.TextAlignmentRole and column != FILE_COLUMN:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if column == FILE_COLUMN:
            return entry.name
        if column == 1:
            return clock(entry.duration)
        # La última columna cuenta el estado: cuántas cajas tiene, o qué le pasa.
        if entry.status == RUNNING:
            return f"{entry.fraction:.0%}"
        if entry.status == FAILED:
            return "error"
        if entry.status == STOPPED:
            return "stopped"
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


# La lista: ↑ ↓ cambian de grabación; el resto de las teclas (Espacio, A, R, N…) siguen
# siendo de la ventana aunque la lista tenga el foco.
class RecordingList(QTableView):
    @override
    def keyPressEvent(self, e) -> None:
        if e is not None and e.key() not in LIST_KEYS:
            e.ignore()
            return
        super().keyPressEvent(e)


class RecordingsPanel(QDockWidget):
    opened = pyqtSignal(object)  # Path de la grabación elegida
    finished_file = pyqtSignal(object)  # Path con tabla nueva
    state_changed = pyqtSignal()  # empezó o terminó una corrida
    said = pyqtSignal(str)  # para la barra de estado

    def __init__(self) -> None:
        super().__init__("Recordings")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.folder_path: Path | None = None
        self.model_path: Path | None = None
        self.engine_ready = False
        self.blocked = False  # el espectrograma está detectando: un modelo a la vez
        self.leading = True  # sin grabación abierta, Detect all es el siguiente paso
        self.worker: BatchWorker | None = None
        self.scanner: Worker | None = None
        self.started_at = 0.0
        self.done_audio_s = 0.0

        self.title = QLabel("")
        self.title.setObjectName("hint")
        self.title.setTextFormat(Qt.TextFormat.PlainText)

        self.files = FileModel()
        self.table = RecordingList()
        self.table.setModel(self.files)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
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
        selection = self.table.selectionModel()
        if selection is not None:
            selection.currentRowChanged.connect(self.on_current)

        self.placeholder = QLabel(PLACEHOLDER)
        self.placeholder.setObjectName("placeholder")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.choose_button = QPushButton("Choose a folder…")
        self.choose_button.clicked.connect(self.choose)
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.addStretch(1)
        empty_layout.addWidget(self.placeholder)
        empty_layout.addWidget(self.choose_button, 0, Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch(1)
        self.pages = QStackedWidget()
        self.pages.addWidget(empty)
        self.pages.addWidget(self.table)

        self.score = QDoubleSpinBox()
        self.score.setRange(*SCORE_RANGE)
        self.score.setSingleStep(SCORE_STEP)
        self.score.setDecimals(2)
        self.score.setValue(SCORE_THRESHOLD)
        self.score.setToolTip(
            "Only detections at or above this score are written. "
            "Starts at the model's operating point."
        )
        self.overwrite = QCheckBox("Redo existing")
        self.overwrite.setToolTip("Run again over recordings that already have a table")
        self.run_button = QPushButton("Detect all")
        self.run_button.clicked.connect(self.run)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        for button in (self.choose_button, self.run_button, self.stop_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.summary = QLabel("")
        self.summary.setObjectName("hint")

        settings = QHBoxLayout()
        settings.setSpacing(8)
        settings.addWidget(QLabel("Score ≥"))
        settings.addWidget(self.score)
        settings.addWidget(self.overwrite)
        settings.addStretch(1)
        settings.addWidget(self.run_button)
        settings.addWidget(self.stop_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self.title)
        layout.addWidget(self.pages, 1)
        layout.addLayout(settings)
        layout.addWidget(self.progress)
        layout.addWidget(self.summary)
        body = QWidget()
        body.setLayout(layout)
        body.setMinimumWidth(PANEL_WIDTH)
        self.setWidget(body)
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

    # El acento va a Detect all sólo cuando no hay una grabación abierta que lo reclame.
    def set_leading(self, leading: bool) -> None:
        self.leading = leading
        self.sync()

    def sync(self) -> None:
        running = self.running()
        listed = bool(self.files.entries)
        self.title.setVisible(self.folder_path is not None)
        for widget in (self.score, self.overwrite):
            widget.setEnabled(not running)
        self.run_button.setEnabled(
            not running
            and not self.scanning()
            and not self.blocked
            and listed
            and self.model_path is not None
            and self.engine_ready
        )
        self.run_button.setToolTip(
            "Choose a model in the toolbar first"
            if self.model_path is None
            else "Waiting for the detection engine"
            if not self.engine_ready
            else "Wait for the current detection to finish"
            if self.blocked
            else "Run the model over every recording in the list"
        )
        emphasize(self.run_button, self.leading and self.run_button.isEnabled())
        # Un solo botón a la vista: Detect all, que mientras corre es Stop.
        self.run_button.setVisible(not running)
        self.stop_button.setVisible(running)
        self.stop_button.setEnabled(running)
        self.pages.setCurrentWidget(self.table if listed else self.pages.widget(0))
        self.progress.setVisible(running)
        self.summary.setVisible(bool(self.summary.text()))

    # --- Carpeta ----------------------------------------------------------------

    def choose(self) -> None:
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
        self.title.setText(f"{folder.name} · scanning…")
        self.title.setToolTip(str(folder))
        self.rescan()

    def rescan(self) -> None:
        if self.folder_path is None or self.running() or self.scanning():
            return
        folder = self.folder_path
        self.scanner = Worker(lambda: scan(folder))
        self.scanner.ok.connect(self.scanned)
        self.scanner.error.connect(lambda message: self.said.emit(f"Could not scan: {message}"))
        self.scanner.finished.connect(self.sync)
        self.scanner.start()
        self.sync()

    def scanned(self, entries: list[Entry]) -> None:
        self.files.reset(entries)
        self.summary.setText("")
        folder = self.folder_path.name if self.folder_path else ""
        existing = sum(e.status == EXISTS for e in entries)
        total_s = sum(e.duration for e in entries if not math.isnan(e.duration))
        self.title.setText(
            f"{folder} · {len(entries)} recordings · {clock(total_s)}"
            + (f" · {existing} with a table" if existing else "")
            if entries
            else f"{folder} · no recordings"
        )
        self.sync()
        self.state_changed.emit()

    # La fila de la grabación abierta, sin volver a abrirla.
    def select(self, path: Path | None) -> None:
        selection = self.table.selectionModel()
        if selection is None:
            return
        row = next((i for i, e in enumerate(self.files.entries) if e.path == path), -1)
        selection.blockSignals(True)
        if row < 0:
            selection.clearSelection()
        else:
            self.table.selectRow(row)
            self.table.scrollTo(self.files.index(row, 0))
        selection.blockSignals(False)

    def on_current(self, current, _previous) -> None:
        if current.isValid():
            self.opened.emit(self.files.entries[current.row()].path)

    # --- Corrida ----------------------------------------------------------------

    def run(self) -> None:
        if self.running() or self.model_path is None or not self.files.entries:
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
        # Lo que aún va a procesarse: lo pendiente y, con Redo, lo que ya tiene tabla.
        status = self.files.entries[index].status
        return status == PENDING or (status == EXISTS and self.overwrite.isChecked())
