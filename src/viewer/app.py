import logging
from pathlib import Path
from typing import override

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDesktopServices,
    QFontDatabase,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from core.config import PROJECT_DIR, SCORE_THRESHOLD, P, score_grid
from core.runtime import log_location
from data.raven import BEGIN, END, selection_table
from inference.catalog import (
    AUDIO_SUFFIXES,
    CHECKPOINT_SUFFIXES,
    MODELS_DIR,
    is_checkpoint,
    output_for,
    read_operating_point,
)
from viewer.batch import BatchView
from viewer.controls import Band, Layers, ModelPicker, SettingsPanel, Slider, headline_font
from viewer.inference import DETECT_THRESHOLD, detect, preload
from viewer.plot import Layer, SpectrogramView
from viewer.review import ACCEPTED, REJECTED, ReviewBar, Reviewer
from viewer.session import (
    ANNOTATIONS,
    COLORS,
    DETECTIONS,
    SOURCES,
    STYLES,
    WIDTHS,
    Row,
    Session,
    read_table,
)
from viewer.spectrogram import load_audio
from viewer.table import BoxTable
from viewer.tasks import Worker
from viewer.transport import Transport

logger = logging.getLogger("viewer")

BASE_TITLE = "Primate Vocalization Detector"
SPECTROGRAM, BATCH = "Spectrogram", "Batch"
SCORE_WIDTH = 200
HELP_SIZE = (640, 620)
EMBLEM_SIZE = 96
EMBLEM_BACKGROUND, EMBLEM_STROKE = "#1c1b22", 5
REVIEW_BAND_MARGIN = 2.0  # la banda mide al menos el doble de la caja

AUDIO = "*.wav *.flac *.mp3 *.WAV *.FLAC *.MP3"
TABLES = "*.txt *.csv"
MODELS = "*.pth *.pt"
TABLE_SUFFIXES = {".txt", ".csv"}
MANUAL_PATHS = (
    PROJECT_DIR / "Manual.pdf",
    PROJECT_DIR / "docs" / "manual" / "build" / "manual.pdf",
)
WELCOME = (
    "Drop a recording here",
    "WAV, FLAC or MP3. Drop a folder to process every recording in it (Batch).",
)
HELP = {
    "Open": [
        ("Ctrl+O", "a recording (WAV, FLAC, MP3); dropping it on the window does the same"),
        ("Ctrl+T", "a Raven table over the open recording: with a Score column it goes to "
                   "Detections, otherwise to Annotations; the <tt>.detections.txt</tt> next "
                   "to a recording opens with it"),
        ("Ctrl+Shift+O", "a folder: the Batch view lists its recordings; double-click one to "
                         "open it here"),
    ],
    "Detect": [
        ("Ctrl+R", "run the model (the Model list under the spectrogram) over the open "
                   "recording"),
        ("Ctrl+Shift+R", "the Batch view: Run leaves a <tt>.detections.txt</tt> next to every "
                         "recording of the folder, with every detection the model finds"),
        ("Score ≥", "in the Spectrogram view only hides the weaker detections; Save keeps what "
                    "is visible. It starts at the model's operating point"),
        ("Batch table", "detections per recording at its Score ≥, with the folder total on "
                        "top; moving Score only recounts. Count per species and call shows one "
                        "column per class; Export CSV saves it"),
        ("Ctrl+L", "clear the detections"),
    ],
    "Look": [
        ("Wheel", "scroll through the audio"),
        ("Ctrl+1 / Ctrl+3", "zoom in and out in time (Ctrl + wheel does the same): "
                            "Window is how much time fits on screen, 0.25 to 30 s"),
        ("Ctrl+↑ / Ctrl+↓", "zoom in and out in frequency (Shift + wheel, around the pointer): "
                            "Band is how much of the range fits"),
        ("↑ ↓", "move the band; F shows it all"),
        ("View > Settings", "brightness, contrast, volume and the audio output"),
        ("Time bar", "the marks are the boxes: click near one to go there"),
        ("Ctrl+E", "show or hide the Boxes panel"),
        ("Ctrl+S", "save the visible stretch as an image"),
    ],
    "Listen": [
        ("Space", "play or pause"),
        ("Click", "move the playhead there"),
        ("Volume", "0 dB is the recording normalized to its peak; raise it for distant calls"),
    ],
    "Move": [
        ("← →", "step forward and back"),
        ("PgUp / PgDn", "a whole window"),
        ("Home / End", "start and end of the audio"),
        ("N / P", "next and previous box: it is framed and its row is selected in Boxes"),
    ],
    "Boxes": [
        ("1 / 2", "show or hide the Annotations and Detections layers"),
        ("Ctrl+E", "table of boxes; click a row to frame it, N / P walk through the tab "
                   "that is showing"),
        ("Del", "in the table, remove the selected boxes"),
        ("File > Save", "keeps the boxes that are visible"),
    ],
    "Review": [
        ("Review", "go through the visible detections one by one, in time order; each box is "
                   "framed, gets handles to drag it or its corners, and the playhead waits at "
                   "its start"),
        ("A / Enter", "accept: the box, as framed and labelled, moves to Annotations"),
        ("R / Del", "reject: the box is dropped"),
        ("N / P", "next or previous box without deciding"),
        ("Esc", "leave the review; every decision is kept, and opening the same recording "
                "again picks up where you left off"),
    ],
}  # fmt: skip


class ControlsRow(QWidget):
    CLOCK_FROM = 520

    def __init__(self, transport: Transport, band: Band) -> None:
        super().__init__()
        self.clock = transport.player.clock

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(transport, 1)
        row.addSpacing(4)
        row.addWidget(transport.spans_label)
        row.addWidget(transport.spans)
        row.addSpacing(4)
        row.addWidget(band.heights_label)
        row.addWidget(band.heights)

    @override
    def resizeEvent(self, a0) -> None:
        super().resizeEvent(a0)
        self.clock.setVisible(self.width() >= self.CLOCK_FROM)


class Welcome(QWidget):
    def __init__(self, open_audio, open_folder) -> None:
        super().__init__()
        emblem = QLabel()
        emblem.setPixmap(emblem_pixmap(EMBLEM_SIZE))
        emblem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        headline = QLabel(WELCOME[0])
        headline.setFont(headline_font(headline))
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail = QLabel(WELCOME[1])
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)

        open_button = QPushButton("Open audio…")
        open_button.clicked.connect(lambda: open_audio())
        folder_button = QPushButton("Open folder…")
        folder_button.clicked.connect(lambda: open_folder())
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)
        buttons.addWidget(open_button)
        buttons.addWidget(folder_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(emblem)
        layout.addSpacing(18)
        layout.addWidget(headline)
        layout.addSpacing(6)
        layout.addWidget(detail)
        layout.addSpacing(22)
        layout.addLayout(buttons)
        layout.addStretch(1)


def emblem_pixmap(size: int) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(EMBLEM_BACKGROUND))
    radius = size * 0.2
    painter.drawRoundedRect(0, 0, size, size, radius, radius)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for source, box in (
        (ANNOTATIONS, (0.17, 0.28, 0.41, 0.44)),
        (DETECTIONS, (0.39, 0.47, 0.44, 0.36)),
    ):
        pen = QPen(QColor(COLORS[source]), EMBLEM_STROKE)
        pen.setStyle(STYLES[source])
        painter.setPen(pen)
        x, y, w, h = (v * size for v in box)
        painter.drawRect(int(x), int(y), int(w), int(h))
    painter.end()
    return pixmap


class Viewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(BASE_TITLE)
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self.last_folder = ""

        self.session = Session()
        self.worker: Worker | None = None
        self.engine: bool | None = None  # None mientras carga
        self.pending_table: Path | None = None  # tabla que se abre en cuanto cargue su audio
        self.queued: list[Path] = []  # lo que se pidió abrir mientras otra cosa cargaba
        self.dock_shown = False  # si el panel de cajas estaba abierto al pasar a Batch
        self.help_dialog: QDialog | None = None

        self.plot = SpectrogramView()
        self.plot.moved.connect(self.track)

        self.transport = Transport()
        self.transport.changed.connect(self.refresh)
        self.plot.scrolled.connect(self.transport.step)
        self.transport.playhead.connect(self.plot.set_playhead)
        self.transport.failed.connect(self.say)
        self.plot.clicked.connect(self.transport.seek)
        self.plot.zoomed.connect(self.transport.zoom)

        self.settings = SettingsPanel()
        self.settings.brightness.changed.connect(self.draw_spectrogram)
        self.settings.contrast.changed.connect(self.draw_spectrogram)
        self.settings.volume.changed.connect(
            lambda: self.transport.set_gain(self.settings.volume.value())
        )
        self.settings.device_changed.connect(self.transport.player.set_device)
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("Settings")
        self.settings_dialog.setWindowFlag(Qt.WindowType.Tool)
        QVBoxLayout(self.settings_dialog).addWidget(self.settings)

        self.band = Band()
        self.band.changed.connect(self.plot.set_band)
        self.plot.banded.connect(self.band.set_values)
        self.plot.band_zoomed.connect(self.band.zoom)

        self.table = BoxTable(self.session)
        self.table.picked.connect(self.focus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.table)
        self.table.hide()

        self.review_bar = ReviewBar()
        self.reviewer = Reviewer(self.session, self.plot, self.review_bar, self.present)
        self.reviewer.changed.connect(self.sync_review)

        self.batch = BatchView()
        self.batch.open_requested.connect(self.open_recording)
        self.batch.finished_file.connect(self.table_written)
        self.batch.said.connect(self.say)
        self.batch.state_changed.connect(self.sync_controls)

        self.picker = ModelPicker()
        self.pickers = (self.picker, self.batch.picker)
        for picker in self.pickers:
            picker.chosen.connect(self.model_chosen)
            picker.browse.connect(self.browse_model)

        self.build_actions()
        self.build_menus()
        self.pages = QStackedWidget()
        self.page = self.build_page()
        self.pages.addWidget(self.page)
        self.pages.addWidget(self.batch)
        self.setCentralWidget(self.pages)
        self.build_status_bar()

        self.session.changed.connect(self.on_changed)
        self.set_mode(SPECTROGRAM)
        self.first_model()
        self.start_engine()

    def build_actions(self) -> None:
        self.modes = QActionGroup(self)
        self.mode_actions: dict[str, QAction] = {}
        for mode, tip in ((SPECTROGRAM, "One recording at a time"), (BATCH, "A whole folder")):
            action = QAction(mode, self.modes)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(lambda _, name=mode: self.set_mode(name))
            self.mode_actions[mode] = action

        self.open_audio_action = self.action(
            "Open audio…", "Ctrl+O", self.open_audio, "WAV, FLAC or MP3"
        )
        self.open_table_action = self.action(
            "Open annotations…",
            "Ctrl+T",
            self.open_table,
            "A Raven selection table over the open recording. With a Score column it goes "
            "to Detections, otherwise to Annotations",
        )
        self.open_folder_action = self.action(
            "Open folder…",
            "Ctrl+Shift+O",
            self.open_folder,
            "A folder of recordings, listed in the Batch view",
        )
        self.quit_action = self.action("Quit", "Ctrl+Q", self.close)

        self.image_action = self.action("Save image of this stretch…", "Ctrl+S", self.export_image)
        self.save_actions = {
            source: self.action(
                f"Save {source.lower()} table…", "", lambda name=source: self.export_table(name)
            )
            for source in SOURCES
        }
        self.counts_action = self.action(
            "Export batch counts…", "", self.batch.export,
            "The Batch table as CSV: one row per recording, its detections at the Batch score, "
            "in total and per class",
        )  # fmt: skip

        self.zoom_in_action = self.action(
            "Zoom in", ["Ctrl+1", "Ctrl++", "Ctrl+="], lambda: self.transport.zoom(-1),
            "Less time on screen (Ctrl + wheel)",
        )  # fmt: skip
        self.zoom_out_action = self.action(
            "Zoom out", ["Ctrl+3", "Ctrl+-"], lambda: self.transport.zoom(1),
            "More time on screen (Ctrl + wheel)",
        )  # fmt: skip
        self.band_in_action = self.action(
            "Narrower band", "Ctrl+Up", lambda: self.band.zoom(-1),
            "Less frequency range on screen (Shift + wheel)",
        )  # fmt: skip
        self.band_out_action = self.action(
            "Wider band", "Ctrl+Down", lambda: self.band.zoom(1),
            "More frequency range on screen (Shift + wheel)",
        )  # fmt: skip
        # "\tF" enseña el atajo sin registrarlo: F va por keyPressEvent
        self.full_band_action = self.action("Full band\tF", "", self.band.full)
        self.boxes_action = self.table.toggleViewAction()
        if self.boxes_action is not None:
            self.boxes_action.setText("Boxes")
            self.boxes_action.setToolTip("Table of annotations and detections (Ctrl+E)")
            self.boxes_action.setShortcut(QKeySequence("Ctrl+E"))
        self.settings_action = self.action("Settings…", "", self.show_settings)

        self.run_action = self.action("Detect", "Ctrl+R", self.run_model)
        self.batch_action = self.action(
            "Run batch on a folder…", "Ctrl+Shift+R", self.run_batch,
            "Detect over every recording of the folder in the Batch view",
        )  # fmt: skip
        self.clear_action = self.action(
            "Clear detections", "Ctrl+L", lambda: self.session.set_table(DETECTIONS, None)
        )
        self.browse_model_action = self.action("Browse for a model…", "Ctrl+M", self.browse_model)

        self.help_action = self.action("Controls", "F1", self.show_help)
        self.manual_action = self.action("User manual (PDF)", "", self.open_manual)
        self.manual_action.setEnabled(self.manual() is not None)

    def action(self, text: str, keys: str | list[str], slot, tip: str = "") -> QAction:
        action = QAction(text, self)
        if isinstance(keys, list):
            action.setShortcuts([QKeySequence(k) for k in keys])
        elif keys:
            action.setShortcut(QKeySequence(keys))
        if tip:
            action.setToolTip(tip)
        action.triggered.connect(lambda _: slot())
        # Los atajos sin botón sólo llegan si sus acciones cuelgan de la ventana.
        self.addAction(action)
        return action

    def build_menus(self) -> None:
        bar = self.menuBar()
        if bar is None:
            return
        file_menu = bar.addMenu("&File")
        if file_menu is not None:
            file_menu.addAction(self.open_audio_action)
            file_menu.addAction(self.open_table_action)
            file_menu.addAction(self.open_folder_action)
            file_menu.addSeparator()
            for action in self.save_actions.values():
                file_menu.addAction(action)
            file_menu.addAction(self.image_action)
            file_menu.addAction(self.counts_action)
            file_menu.addSeparator()
            file_menu.addAction(self.quit_action)
        view_menu = bar.addMenu("&View")
        if view_menu is not None:
            for action in self.mode_actions.values():
                view_menu.addAction(action)
            view_menu.addSeparator()
            view_menu.addAction(self.zoom_in_action)
            view_menu.addAction(self.zoom_out_action)
            view_menu.addAction(self.band_in_action)
            view_menu.addAction(self.band_out_action)
            view_menu.addAction(self.full_band_action)
            view_menu.addSeparator()
            if self.boxes_action is not None:
                view_menu.addAction(self.boxes_action)
            view_menu.addAction(self.settings_action)
        detect_menu = bar.addMenu("&Detect")
        if detect_menu is not None:
            detect_menu.addAction(self.run_action)
            detect_menu.addAction(self.batch_action)
            detect_menu.addAction(self.clear_action)
            detect_menu.addSeparator()
            detect_menu.addAction(self.browse_model_action)
        help_menu = bar.addMenu("&Help")
        if help_menu is not None:
            help_menu.addAction(self.help_action)
            help_menu.addAction(self.manual_action)

    def build_page(self) -> QWidget:
        self.welcome = Welcome(self.open_audio, self.open_folder)
        self.spectrogram = QWidget()
        with_scroll = QHBoxLayout(self.spectrogram)
        with_scroll.setContentsMargins(0, 0, 0, 0)
        with_scroll.setSpacing(2)
        with_scroll.addWidget(self.plot, 1)
        with_scroll.addWidget(self.band.bar)
        self.canvas = QStackedWidget()
        self.canvas.addWidget(self.welcome)
        self.canvas.addWidget(self.spectrogram)

        self.controls = ControlsRow(self.transport, self.band)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.controls)
        layout.addWidget(self.build_context())
        page = QWidget()
        page.setLayout(layout)
        return page

    def build_context(self) -> QWidget:
        picker_label = QLabel("Model")
        picker_label.setToolTip(self.picker.toolTip())
        self.run_button = QPushButton("Detect")
        self.run_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.run_button.clicked.connect(self.run_action.trigger)

        self.layers = Layers([(s, COLORS[s], STYLES[s], WIDTHS[s]) for s in SOURCES])
        self.layers.changed.connect(self.draw_boxes)

        # Sólo esconde cajas: nada de lo que se escribe depende de esto
        thresholds = score_grid(DETECT_THRESHOLD, 1.0)
        self.score = Slider(
            "Score ≥", thresholds, thresholds.index(SCORE_THRESHOLD), "{:.2f}", label_width=52
        )
        self.score.setMaximumWidth(SCORE_WIDTH)
        self.score.setToolTip("Hide detections below this score")
        self.score.changed.connect(lambda: self.session.set_score(self.score.value()))
        self.review_button = QPushButton("Review")
        self.review_button.setToolTip("Go through the visible detections one by one")
        self.review_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.review_button.clicked.connect(self.reviewer.start)

        self.model_tools = QWidget()
        row = QHBoxLayout(self.model_tools)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(picker_label)
        row.addWidget(self.picker)
        row.addWidget(self.run_button)
        row.addSpacing(16)
        row.addWidget(self.score)
        row.addSpacing(8)
        row.addWidget(self.layers)
        row.addStretch(1)
        row.addWidget(self.review_button)

        self.context = QStackedWidget()
        self.context.addWidget(self.model_tools)
        self.context.addWidget(self.review_bar)
        return self.context

    def show_settings(self) -> None:
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def manual(self) -> Path | None:
        return next((path for path in MANUAL_PATHS if path.is_file()), None)

    def open_manual(self) -> None:
        path = self.manual()
        if path is None:
            self.say("The manual (Manual.pdf) is not next to the program.")
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def show_help(self) -> None:
        if self.help_dialog is None:
            rows = "".join(
                f"<tr><td colspan='2' style='padding-top:14px'><b>{section}</b></td></tr>"
                + "".join(
                    f"<tr><td style='padding-right:20px; white-space:nowrap'><tt>{keys}</tt></td>"
                    f"<td>{what}</td></tr>"
                    for keys, what in entries
                )
                for section, entries in HELP.items()
            )
            text = QTextBrowser()
            text.setOpenExternalLinks(False)
            text.setFrameStyle(0)
            text.setHtml(f"<table cellspacing='0' cellpadding='2'>{rows}</table>")
            close = QPushButton("Close")
            dialog = QDialog(self)
            dialog.setWindowTitle("Controls")
            dialog.setWindowFlag(Qt.WindowType.Tool)
            dialog.resize(*HELP_SIZE)
            close.clicked.connect(dialog.hide)
            buttons = QHBoxLayout()
            buttons.addStretch(1)
            buttons.addWidget(close)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.addWidget(text, 1)
            layout.addLayout(buttons)
            self.help_dialog = dialog
        self.help_dialog.show()
        self.help_dialog.raise_()
        self.help_dialog.activateWindow()

    def build_status_bar(self) -> None:
        self.task = QLabel("")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setTextVisible(False)
        self.readout = QLabel("")
        self.readout.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.readout.setMinimumWidth(160)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.bar = QStatusBar()
        self.bar.addPermanentWidget(self.task)
        self.bar.addPermanentWidget(self.progress)
        self.bar.addPermanentWidget(self.readout)
        self.setStatusBar(self.bar)
        self.show_task(None)

    def show_task(self, text: str | None) -> None:
        self.task.setText(text or "")
        self.task.setVisible(text is not None)
        self.progress.setVisible(text is not None)

    def first_model(self) -> None:
        listed = self.picker.paths()
        if listed:
            self.picker.select(listed[0])

    def folder(self) -> str:
        if self.session.audio_path is not None:
            return str(self.session.audio_path.parent)
        return self.last_folder

    def start_engine(self) -> None:
        self.say("Loading the detection engine… you can open a recording meanwhile.")
        self.progress.setRange(0, 0)
        self.show_task("Loading detection engine")
        self.preloader = Worker(preload, what="Loading the detection engine")
        self.preloader.ok.connect(self.engine_ready)
        self.preloader.error.connect(self.engine_failed)
        self.preloader.start()

    def engine_ready(self, _) -> None:
        self.engine = True
        self.batch.set_engine(True)
        self.settings.reload_outputs()
        self.finish()
        if self.mode() == SPECTROGRAM and self.session.waveform is None:
            self.say("Ready. Drop a recording to start, or press Ctrl+O.")
        else:
            self.say("Ready.")

    def engine_failed(self, message: str) -> None:
        self.engine = False
        self.batch.set_engine(False)
        self.settings.reload_outputs()
        self.finish()
        self.say("The detection engine did not load. Viewing still works.")
        QMessageBox.warning(
            self,
            "Detection unavailable",
            f"The detection engine did not load:\n{message}\n\n"
            "You can still open recordings and tables. "
            f"Details are in {log_location() or 'the console'}.",
        )

    def say(self, message: str) -> None:
        self.bar.showMessage(message)

    def fail(self, message: str) -> None:
        self.pending_table = None
        self.say("Error.")
        QMessageBox.critical(self, "Error", message)

    # Entre que el hilo entrega y termina, isRunning sigue en alto un instante
    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning() and not self.worker.done

    def mode(self) -> str:
        return BATCH if self.pages.currentWidget() is self.batch else SPECTROGRAM

    def set_mode(self, mode: str) -> None:
        self.mode_actions[mode].setChecked(True)
        leaving = mode == BATCH and self.mode() == SPECTROGRAM
        if leaving:
            self.dock_shown = self.table.isVisible()
            self.table.hide()
        self.pages.setCurrentWidget(self.batch if mode == BATCH else self.page)
        if mode == SPECTROGRAM and self.dock_shown:
            self.table.show()
            self.dock_shown = False
        self.sync_controls()

    def sync_controls(self) -> None:
        busy = self.busy()
        loaded = self.session.waveform is not None
        detections = self.session.tables[DETECTIONS] is not None
        model = self.session.model_path
        engine = bool(self.engine)
        reviewing = self.reviewer.active
        batch = self.batch.running()
        spectrogram = self.mode() == SPECTROGRAM
        self.open_audio_action.setEnabled(not busy)
        self.open_table_action.setEnabled(not busy and loaded)
        self.open_folder_action.setEnabled(not batch)
        self.picker.setEnabled(not busy and not batch)
        self.batch.set_blocked(busy)
        self.browse_model_action.setEnabled(self.picker.isEnabled())
        self.run_action.setEnabled(
            spectrogram and not busy and not batch and loaded and engine and model is not None
        )
        self.run_action.setToolTip(
            "Open a recording first (Ctrl+R)"
            if not loaded
            else "Choose a model first (Ctrl+R)"
            if model is None
            else "The detection engine did not load: see viewer.log (Ctrl+R)"
            if self.engine is False
            else "Waiting for the detection engine (Ctrl+R)"
            if not engine
            else "Batch is running (Ctrl+R)"
            if batch
            else f"Run {model.parent.name} over the open recording (Ctrl+R)"
        )
        self.run_button.setEnabled(self.run_action.isEnabled())
        self.run_button.setToolTip(self.run_action.toolTip())
        self.batch_action.setEnabled(not batch)
        self.clear_action.setEnabled(spectrogram and not busy and detections)
        self.image_action.setEnabled(spectrogram and not busy and loaded)
        for source, action in self.save_actions.items():
            action.setEnabled(spectrogram and not busy and self.session.tables[source] is not None)
        self.counts_action.setEnabled(not spectrogram and not batch and self.batch.listed())
        for action in (
            self.zoom_in_action,
            self.zoom_out_action,
            self.band_in_action,
            self.band_out_action,
            self.full_band_action,
        ):
            action.setEnabled(spectrogram and loaded)
        if self.boxes_action is not None:
            self.boxes_action.setEnabled(spectrogram)
        self.canvas.setCurrentWidget(self.spectrogram if loaded else self.welcome)
        self.controls.setVisible(loaded)
        self.context.setVisible(loaded)
        self.context.setCurrentIndex(1 if reviewing else 0)
        self.score.setEnabled(detections and not busy)
        self.review_button.setEnabled(detections and not busy)

    def on_changed(self) -> None:
        self.draw_boxes()
        self.sync_controls()

    def start(self, task, done, message: str, reports: bool = False) -> None:
        if self.busy():
            return
        if self.worker is not None:
            self.worker.wait()
        self.say(message)
        self.worker = Worker(task, reports, what=message.rstrip("…"))
        self.worker.ok.connect(done)
        self.worker.error.connect(self.fail)
        self.worker.progress.connect(self.show_progress)
        self.worker.finished.connect(self.finish)
        self.worker.start()
        self.sync_controls()
        self.progress.setRange(0, 0)  # indeterminado hasta el primer reporte
        self.show_task(message.rstrip("…"))

    def show_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def finish(self) -> None:
        if self.busy():
            pass
        elif self.engine is None:
            self.show_task("Loading detection engine")
        else:
            self.show_task(None)
        self.sync_controls()
        if self.queued and not self.busy():
            self.open_path(self.queued.pop(0))

    def open_audio(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Open audio", self.folder(), f"Recordings ({AUDIO})"
        )
        if chosen:
            self.last_folder = str(Path(chosen).parent)
            self.open_path(Path(chosen))

    def open_table(self) -> None:
        if self.session.audio_path is None:
            self.say("Open a recording first: tables are drawn over it.")
            return
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Open annotations", self.folder(), f"Raven tables ({TABLES})"
        )
        if chosen:
            self.open_path(Path(chosen))

    def open_folder(self) -> None:
        self.set_mode(BATCH)
        self.batch.browse()

    def browse_model(self) -> None:
        folder = str(MODELS_DIR if MODELS_DIR.is_dir() else PROJECT_DIR)
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose a model", folder, f"Checkpoint ({MODELS})"
        )
        if chosen:
            self.picker.select(Path(chosen))

    def open_path(self, path: Path) -> None:
        if self.busy():
            if path not in self.queued:
                self.queued.append(path)
            return
        suffix = path.suffix.lower()
        if path.is_dir():
            self.set_mode(BATCH)
            self.batch.set_folder(path)
        elif suffix in AUDIO_SUFFIXES:
            self.set_mode(SPECTROGRAM)
            self.load_audio(path)
        elif suffix in TABLE_SUFFIXES:
            self.set_mode(SPECTROGRAM)
            if self.session.audio_path is None:
                self.say("Open a recording first: tables are drawn over it.")
            else:
                self.load_table(path)
        elif is_checkpoint(path):
            self.picker.select(path)
        else:
            self.say(f"Cannot open '{path.name}'.")

    def load_audio(self, path: Path) -> None:
        table = output_for(path)
        self.pending_table = table if table.is_file() else None
        self.start(
            lambda: (path, load_audio(path, P.target_sr)),
            self.audio_loaded,
            f"Loading {path.name}…",
        )

    def load_table(self, path: Path) -> None:
        self.start(lambda: read_table(path), self.table_loaded, f"Loading {path.name}…")

    # select no emite si ya está elegido
    def model_chosen(self, path: Path | None) -> None:
        for picker in self.pickers:
            picker.select(path)
        if path == self.session.model_path:
            return
        self.session.model_path = path
        operating = None if path is None else read_operating_point(path.parent)
        self.batch.set_model(path, operating)
        if operating is not None:
            self.score.set_value(operating)
        if path is not None:
            self.say(
                f"Model: {path.parent.name}"
                + ("" if operating is None else f" · operating point {operating:.2f}")
            )
        self.sync_controls()

    def run_model(self) -> None:
        audio, model = self.session.audio_path, self.session.model_path
        if audio is None or model is None or not self.run_action.isEnabled():
            return
        self.start(
            lambda report: detect(audio, model, on_progress=report),
            self.detections_ready,
            f"Running {model.parent.name}…",
            reports=True,
        )

    def run_batch(self) -> None:
        self.set_mode(BATCH)
        if not self.batch.listed():
            self.batch.browse()
        elif self.batch.can_run():
            self.batch.run()
        else:
            self.say(self.batch.why())

    def open_recording(self, path: Path) -> None:
        self.set_mode(SPECTROGRAM)
        if path != self.session.audio_path:
            self.open_path(path)

    def table_written(self, path: Path) -> None:
        if path == self.session.audio_path and not self.reviewer.active:
            self.load_table(output_for(path))

    def audio_loaded(self, loaded: tuple) -> None:
        path, waveform = loaded
        self.session.set_audio(path, waveform, P.target_sr)
        self.setWindowTitle(f"{path.name} - {BASE_TITLE}")
        self.plot.set_waveform(waveform, P.target_sr)
        self.transport.set_audio(waveform, P.target_sr)
        self.band.set_limits(P.nyquist_hz)
        self.say(f"{path.name} · {self.session.duration:.1f} s")
        if self.pending_table is not None:
            table, self.pending_table = self.pending_table, None
            self.load_table(table)

    def table_loaded(self, loaded: tuple[str, pd.DataFrame]) -> None:
        source, table = loaded
        self.session.set_table(source, table)
        skipped = table.attrs.get("skipped", 0)
        self.say(
            f"{len(table)} {source.lower()} loaded."
            + (f" {skipped} rows without a time or a frequency were left out." if skipped else "")
        )

    def detections_ready(self, table: pd.DataFrame) -> None:
        # Sin operating_point.json el slider se queda donde estaba
        operating = table.attrs.get("operating_point")
        if operating is not None:
            self.score.set_value(operating)
        self.session.set_table(DETECTIONS, table)
        where = "the GPU" if str(table.attrs.get("device")).startswith("cuda") else "the CPU"
        self.say(
            f"{len(table)} detections from the model, on {where}."
            + ("" if operating is None else f"   Operating point: {operating:.2f}")
        )

    def skip(self, direction: int) -> None:
        if self.reviewer.active:
            self.reviewer.step(direction)
            return
        start, stop = self.transport.time_window()
        if not self.table.step(direction, start if direction > 0 else stop):
            self.say("No more boxes that way.")

    def focus(self, row) -> None:
        if row is None:
            self.plot.set_highlight(None)
            return
        self.frame(row)
        self.plot.set_highlight((row.begin, row.end, row.low, row.high))

    def frame(self, row: Row) -> None:
        start, stop = self.transport.time_window()
        if not (start <= row.begin and row.end <= stop):
            self.transport.center(0.5 * (row.begin + row.end))
        low, high = self.band.current
        if not (low <= row.low and row.high <= high):
            self.band.frame(row.low, row.high)

    def present(self, row: Row) -> None:
        self.transport.fit(row.begin, row.end)
        self.band.frame(row.low, row.high, margin=REVIEW_BAND_MARGIN)
        self.transport.seek(row.begin)

    def sync_review(self) -> None:
        counts = self.reviewer.counts()
        accepted, rejected = counts.get(ACCEPTED, 0), counts.get(REJECTED, 0)
        if self.reviewer.active:
            left = len(self.reviewer.rows())
            resumed = self.reviewer.resumed
            self.say(
                f"Review: {accepted} accepted, {rejected} rejected · {left} left"
                + (f"   ({resumed} picked up from an earlier review)" if resumed else "")
            )
        elif accepted or rejected:
            self.say(
                f"Review closed: {accepted} accepted, {rejected} rejected. "
                "Save the Annotations table to keep the accepted ones."
            )
        else:
            self.say("Review closed.")
        self.sync_controls()

    def track(self, seconds: float, hz: float) -> None:
        if self.session.waveform is not None:
            self.readout.setText(f"{seconds:.3f} s   {hz:,.0f} Hz")

    def refresh(self) -> None:
        self.draw_spectrogram()
        self.draw_boxes()

    def draw_spectrogram(self) -> None:
        start = self.transport.time_window()[0]
        self.plot.draw(
            start,
            self.transport.span(),
            self.settings.brightness.value(),
            self.settings.contrast.value(),
        )

    def draw_boxes(self) -> None:
        start, stop = self.transport.time_window()
        layers = [
            Layer(
                self.session.visible(source) if self.layers.enabled(source) else None,
                COLORS[source],
                STYLES[source],
                WIDTHS[source],
                source == ANNOTATIONS,
            )
            for source in SOURCES
        ]
        self.plot.draw_boxes(layers, start, stop)
        counts = {s: len(t) if (t := self.session.visible(s)) is not None else 0 for s in SOURCES}
        both = all(counts.values())
        for source in SOURCES:
            self.layers.set_state(source, counts[source] if both else 0)
        marks = [
            (float(begin), float(end), layer.color)
            for layer in layers
            if layer.table is not None
            for begin, end in zip(layer.table[BEGIN], layer.table[END], strict=True)
        ]
        self.transport.set_marks(marks)

    def save_path(self, title: str, suffix: str, file_filter: str) -> Path | None:
        audio = self.session.audio_path
        stem = audio.stem if audio is not None else "spectrogram"
        folder = audio.parent if audio is not None else Path.cwd()
        chosen, _ = QFileDialog.getSaveFileName(
            self, title, str(folder / f"{stem}{suffix}"), file_filter
        )
        return Path(chosen) if chosen else None

    def export_image(self) -> None:
        if self.session.waveform is None:
            return
        start, stop = self.transport.time_window()
        path = self.save_path("Save image", f"_{start:.2f}-{stop:.2f}s.png", "PNG (*.png)")
        if path is None:
            return
        try:
            self.plot.export_png(path)
        except Exception as exc:
            logger.exception("Could not save the image %s", path)
            self.fail(f"Could not save the image:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"Image saved as {path.name}")

    # Sólo las cajas visibles: las detecciones bajo el slider no van
    def export_table(self, source: str) -> None:
        table = self.session.visible(source)
        if table is None:
            return
        suffix = f".{'detections' if source == DETECTIONS else 'annotations'}.txt"
        path = self.save_path(f"Save {source.lower()}", suffix, "Raven (*.txt);;CSV (*.csv)")
        if path is None:
            return
        table = selection_table(table.copy())
        try:
            table.to_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", index=False)
        except Exception as exc:
            logger.exception("Could not save the table %s", path)
            self.fail(f"Could not save the table:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"{len(table)} rows saved to {path.name}")

    def dropped(self, event) -> Path | None:
        mime = event.mimeData() if event is not None else None
        urls = mime.urls() if mime is not None and mime.hasUrls() else []
        if not urls or self.busy():
            return None
        path = Path(urls[0].toLocalFile())
        known = {*AUDIO_SUFFIXES, *TABLE_SUFFIXES, *CHECKPOINT_SUFFIXES}
        if path.is_dir() or (path.is_file() and path.suffix.lower() in known):
            return path
        return None

    @override
    def dragEnterEvent(self, a0) -> None:
        if a0 is not None and self.dropped(a0) is not None:
            a0.acceptProposedAction()

    @override
    def dropEvent(self, a0) -> None:
        path = self.dropped(a0) if a0 is not None else None
        if a0 is not None and path is not None:
            a0.acceptProposedAction()
            self.open_path(path)

    @override
    def keyPressEvent(self, a0) -> None:
        if a0 is None or self.session.waveform is None or self.mode() != SPECTROGRAM:
            if a0 is not None:
                super().keyPressEvent(a0)
            return
        key = a0.key()
        # El Enter del teclado numérico llega con KeypadModifier: cuenta como una tecla sola.
        modifiers = a0.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
        reviewing = self.reviewer.active and not modifiers
        if reviewing and key in (Qt.Key.Key_A, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.reviewer.decide(ACCEPTED)
        elif reviewing and key in (Qt.Key.Key_R, Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.reviewer.decide(REJECTED)
        elif reviewing and key == Qt.Key.Key_Escape:
            self.reviewer.stop()
        elif key in (Qt.Key.Key_N, Qt.Key.Key_P):
            self.skip(1 if key == Qt.Key.Key_N else -1)
        elif key == Qt.Key.Key_Space:
            self.transport.toggle_play()
        elif key == Qt.Key.Key_Left:
            self.transport.step(-1)
        elif key == Qt.Key.Key_Right:
            self.transport.step(1)
        elif key == Qt.Key.Key_PageUp:
            self.transport.page(-1)
        elif key == Qt.Key.Key_PageDown:
            self.transport.page(1)
        elif key in (Qt.Key.Key_Home, Qt.Key.Key_End):
            self.transport.to_edge(key == Qt.Key.Key_End)
        elif key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            self.band.pan(1 if key == Qt.Key.Key_Up else -1)
        elif key == Qt.Key.Key_F:
            self.band.full()
        elif key in (Qt.Key.Key_1, Qt.Key.Key_2):
            self.layers.toggle(SOURCES[key - Qt.Key.Key_1])
        else:
            super().keyPressEvent(a0)

    @override
    def closeEvent(self, a0) -> None:
        if a0 is None:
            return
        if self.batch.running():
            answer = QMessageBox.question(
                self,
                "Batch is running",
                "Stop it and quit?\nTables already written are kept.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                a0.ignore()
                return
            self.batch.stop()
        if self.batch.worker is not None:
            self.batch.worker.wait()
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
        self.plot.close_renderer()
        super().closeEvent(a0)


def run(app: QApplication, paths: list[Path]) -> Viewer:
    app.setApplicationName(BASE_TITLE)
    app.setStyle("Fusion")
    style = app.style()
    if style is not None:
        app.setPalette(style.standardPalette())
    palette = app.palette()
    pg.setConfigOptions(
        imageAxisOrder="row-major",
        background=palette.base().color(),
        foreground=palette.text().color(),
    )
    viewer = Viewer()
    viewer.show()
    for path in paths:
        viewer.open_path(path)
    return viewer
