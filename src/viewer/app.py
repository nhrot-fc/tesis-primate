import zipfile
from pathlib import Path
from typing import override

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QFontDatabase, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.config import PROJECT_DIR, SCORE_THRESHOLD, P, score_grid
from data.raven import BEGIN, renumber
from inference.catalog import (
    AUDIO_SUFFIXES,
    CHECKPOINT_SUFFIXES,
    MODELS_DIR,
    is_checkpoint,
    output_for,
    read_operating_point,
)
from viewer.batch import BatchView
from viewer.controls import Band, Layers, ModelPicker, Popup, Slider, ViewPanel
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

BASE_TITLE = "Primate Vocalization Detector"
REVIEW, BATCH = "Spectrogram", "Batch"
SCORE_WIDTH = 220
HELP_WIDTH = 560
CARET = "▾"
# En orden de preferencia; si no hay ninguna se queda la del escritorio.
UI_FONTS = ("Inter", "Cantarell", "Noto Sans", "Ubuntu", "DejaVu Sans")
UI_POINT_SIZE = 10

AUDIO = "*.wav *.flac *.mp3 *.WAV *.FLAC *.MP3"
TABLES = "*.txt *.csv"
MODELS = "*.pth *.pt"
TABLE_SUFFIXES = {".txt", ".csv"}
MODEL_ZIP_SUFFIX = ".zip"
PLACEHOLDER = (
    "<div style='font-size:15pt'>Drag and drop an audio to start</div>"
    "<div style='margin-top:8px'>or press Ctrl+O · WAV, FLAC, MP3</div>"
)
HELP = {
    "Files": [
        ("Ctrl+O", "open an audio; drop one on the window does the same"),
        ("Ctrl+T", "open a Raven table over the audio: with a Score column it goes to "
                   "Detections, otherwise to Annotations"),
        ("Ctrl+M", "browse for a model checkpoint; the toolbar lists the ones in models\\"),
        ("Ctrl+R", "run the model over the open audio"),
        ("Ctrl+L", "clear the detections"),
        ("Ctrl+S", "save the visible stretch as an image"),
        ("Folder", "drop a folder on the window to process it in the Batch view"),
        ("Zip", "unzip a model zip into the program folder, next to Detector.exe; "
                "dropping it on the window does the same"),
    ],
    "View": [
        ("Wheel", "scroll through the audio"),
        ("Ctrl + wheel", "window width, 0.25 to 30 s; the list next to the time bar does the same"),
        ("Shift + wheel", "band height, 200 Hz to full, around the pointer; the list next to it "
                          "does the same"),
        ("↑ ↓", "move the band; the scroll on the right does the same"),
        ("F", "full band"),
        ("View", "brightness, contrast and volume"),
        ("", "the window width and the band only change when you change them"),
    ],
    "Listen": [
        ("Space", "play or pause"),
        ("Click", "move the playhead there"),
        ("Volume", "0 dB is the recording normalized to its peak; raise it for distant calls"),
    ],
    "Navigate": [
        ("← →", "step forward and back"),
        ("PgUp / PgDn", "a whole window"),
        ("Home / End", "start and end of the audio"),
        ("N / P", "next and previous detection"),
    ],
    "Boxes": [
        ("1 / 2", "show or hide the Annotations and Detections layers"),
        ("Ctrl+E", "table of boxes; click a row to frame it"),
        ("Del", "in the table, remove the selected boxes"),
        ("Score ≥", "hide detections below that score; Save keeps the visible ones"),
    ],
    "Review": [
        ("Review", "go through the visible detections one by one, in time order; each box is "
                   "framed and gets handles to drag it or its corners"),
        ("A", "accept: the box, as framed and labelled in the list, moves to Annotations"),
        ("R", "reject: the box is dropped"),
        ("S", "skip; N / P move to the next or previous box"),
        ("Journal", "every decision is written at once to a temporary file per recording "
                    "(Save → Reviewed boxes… turns the accepted ones into an annotations table)"),
    ],
}  # fmt: skip

# Espaciado y separadores salen de la paleta: la ventana se ve igual en claro y en oscuro.
QSS = """
QToolBar {{
    border: 0;
    border-bottom: 1px solid {line};
    padding: 5px 8px;
    spacing: 4px;
}}
QToolBar QToolButton, QToolBar QToolButton:menu-indicator {{
    padding: 5px 10px;
    border: 0;
    border-radius: 5px;
}}
QToolBar QToolButton:hover:enabled {{ background: {hover}; }}
QToolBar QToolButton:pressed:enabled {{ background: {press}; }}
QToolBar QToolButton:checked {{ background: {press}; font-weight: 600; }}
QToolBar QComboBox {{ padding: 3px 8px; min-width: 140px; }}
#hint, #placeholder {{ color: {muted}; }}
#readout, #clock {{ font-family: "{mono}"; }}
QStatusBar {{ border-top: 1px solid {line}; }}
QStatusBar::item {{ border: 0; }}
QToolTip {{ padding: 4px 6px; }}
"""


# Una tipografía y un espaciado para toda la ventana: sin esto cada control trae el suyo.
def apply_style(app: QApplication) -> None:
    families = set(QFontDatabase.families())
    font = app.font()
    for name in UI_FONTS:
        if name in families:
            font.setFamily(name)
            break
    font.setPointSize(UI_POINT_SIZE)
    app.setFont(font)

    window = app.palette().window().color()
    dark = window.lightness() < 128
    app.setStyleSheet(
        QSS.format(
            line=(window.lighter(140) if dark else window.darker(112)).name(),
            hover=(window.lighter(125) if dark else window.darker(107)).name(),
            press=(window.lighter(145) if dark else window.darker(115)).name(),
            muted=(window.lighter(220) if dark else window.darker(165)).name(),
            mono=QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family(),
        )
    )


# Un zip de modelo trae `models/<nombre>/…` y, si hace falta, `hf/…`; se vuelca sobre la raíz
# del paquete. Sólo entran rutas relativas dentro de esas dos carpetas.
def add_model_zip(archive: Path) -> str:
    with zipfile.ZipFile(archive) as z:
        members = [
            m
            for m in z.namelist()
            if not Path(m).is_absolute()
            and ".." not in Path(m).parts
            and Path(m).parts[:1] in (("models",), ("hf",))
        ]
        names = {
            Path(m).parts[1] for m in members if m.startswith("models/") and len(Path(m).parts) > 2
        }
        if not names:
            raise ValueError(f"{archive.name} has no models/<name>/ folder inside")
        z.extractall(PROJECT_DIR, members)
    return ", ".join(sorted(names))


class Viewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(BASE_TITLE)
        self.setMinimumSize(1000, 600)
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        self.session = Session()
        self.worker: Worker | None = None
        self.engine: bool | None = None  # None mientras carga
        self.pending_table: Path | None = None  # tabla que se abre en cuanto cargue su audio
        self.dock_shown = False  # si el panel de cajas estaba abierto al pasar a Batch

        self.plot = SpectrogramView()
        self.plot.moved.connect(self.track)

        self.transport = Transport()
        self.transport.changed.connect(self.refresh)
        self.plot.scrolled.connect(self.transport.step)
        self.transport.playhead.connect(self.plot.set_playhead)
        self.transport.failed.connect(self.say)
        self.plot.clicked.connect(self.transport.seek)
        self.plot.zoomed.connect(self.transport.zoom)

        self.view = ViewPanel()
        self.view.brightness.changed.connect(self.draw_spectrogram)
        self.view.contrast.changed.connect(self.draw_spectrogram)
        self.view.volume.changed.connect(lambda: self.transport.set_gain(self.view.volume.value()))

        # La banda: el control manda al espectrograma y este devuelve lo que pudo (recortado).
        self.band = Band()
        self.band.changed.connect(self.plot.set_band)
        self.plot.banded.connect(self.band.set_values)
        self.plot.band_zoomed.connect(self.band.zoom)

        self.table = BoxTable(self.session)
        self.table.picked.connect(self.focus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.table)
        self.table.hide()

        self.review_bar = ReviewBar()
        self.reviewer = Reviewer(self.session, self.plot, self.review_bar, self.frame)
        self.reviewer.changed.connect(self.sync_review)

        self.batch = BatchView()
        self.batch.open_requested.connect(self.open_result)
        self.batch.said.connect(self.say)
        self.batch.state_changed.connect(self.sync_controls)

        self.build_toolbar()
        self.pages = QStackedWidget()
        self.pages.addWidget(self.build_review())
        self.pages.addWidget(self.batch)
        self.setCentralWidget(self.pages)
        self.build_status_bar()

        self.session.changed.connect(self.on_changed)
        self.set_mode(REVIEW)
        self.start_engine()

    # --- Construccion -----------------------------------------------------------

    def build_toolbar(self) -> None:
        # Las dos vistas: un botón marcado por cada una.
        self.modes = QActionGroup(self)
        self.mode_actions = {}
        for mode, tip in ((REVIEW, "One recording at a time"), (BATCH, "A whole folder")):
            action = QAction(mode, self.modes)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(lambda _, name=mode: self.set_mode(name))
            self.mode_actions[mode] = action

        self.open_audio_action = QAction("Open audio…", self)
        self.open_audio_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_audio_action.setToolTip("WAV, FLAC or MP3 (Ctrl+O)")
        self.open_audio_action.triggered.connect(self.open_audio)
        self.open_table_action = QAction("Open table…", self)
        self.open_table_action.setShortcut(QKeySequence("Ctrl+T"))
        self.open_table_action.setToolTip(
            "Raven selection table over the open audio (Ctrl+T)\n"
            "With a Score column it goes to Detections, otherwise to Annotations"
        )
        self.open_table_action.triggered.connect(self.open_table)

        self.picker = ModelPicker()
        self.picker.chosen.connect(self.model_chosen)
        self.picker.browse.connect(self.browse_model)
        self.picker.add.connect(self.add_model)
        self.browse_model_action = QAction("Browse for a model…", self)
        self.browse_model_action.setShortcut(QKeySequence("Ctrl+M"))
        self.browse_model_action.triggered.connect(self.browse_model)

        self.run_action = QAction("Detect", self)
        self.run_action.setShortcut(QKeySequence("Ctrl+R"))
        self.run_action.triggered.connect(self.run_model)
        self.clear_action = QAction("Clear", self)
        self.clear_action.setShortcut(QKeySequence("Ctrl+L"))
        self.clear_action.setToolTip("Remove the detections (Ctrl+L)")
        self.clear_action.triggered.connect(lambda: self.session.set_table(DETECTIONS, None))

        self.viewer_button = Popup(f"View {CARET}", self.view, "Brightness, contrast, volume")

        self.image_action = QAction("Image of this stretch…", self)
        self.image_action.setShortcut(QKeySequence("Ctrl+S"))
        self.image_action.triggered.connect(self.export_image)
        self.save_actions = {source: QAction(f"{source} table…", self) for source in SOURCES}
        for source, action in self.save_actions.items():
            action.triggered.connect(lambda _, name=source: self.export_table(name))
        self.reviewed_action = QAction("Reviewed boxes…", self)
        self.reviewed_action.setToolTip("The boxes accepted in Review, as an annotations table")
        self.reviewed_action.triggered.connect(self.export_reviewed)
        export_menu = QMenu(self)
        export_menu.addAction(self.image_action)
        export_menu.addSeparator()
        for action in self.save_actions.values():
            export_menu.addAction(action)
        export_menu.addAction(self.reviewed_action)
        self.export_button = self.menu_button("Save", export_menu)

        self.help_action = QAction("Help", self)
        self.help_action.setShortcut(QKeySequence("F1"))
        self.help_action.setToolTip("What every control does (F1)")
        self.help_action.triggered.connect(self.show_help)

        self.boxes_action = self.table.toggleViewAction()
        if self.boxes_action is not None:
            self.boxes_action.setText("Boxes")
            self.boxes_action.setToolTip("Table of annotations and detections (Ctrl+E)")
            self.boxes_action.setShortcut(QKeySequence("Ctrl+E"))

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for action in self.mode_actions.values():
            toolbar.addAction(action)
        # Lo que sólo tiene sentido con un espectrograma delante se esconde en Batch.
        # (`addAction(QAction)` no devuelve nada; `addSeparator` y `addWidget` sí.)
        self.review_only: list[QAction] = []
        separator = toolbar.addSeparator()
        if separator is not None:
            self.review_only.append(separator)
        for action in (self.open_audio_action, self.open_table_action):
            toolbar.addAction(action)
            self.review_only.append(action)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Model"))
        toolbar.addWidget(self.picker)
        for action in (self.run_action, self.clear_action):
            toolbar.addAction(action)
            self.review_only.append(action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        for widget in (self.viewer_button, self.export_button):
            placed = toolbar.addWidget(widget)
            if placed is not None:
                self.review_only.append(placed)
        if self.boxes_action is not None:
            toolbar.addAction(self.boxes_action)
            self.review_only.append(self.boxes_action)
        toolbar.addAction(self.help_action)
        toolbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addToolBar(toolbar)
        # Los atajos del menú sólo llegan si sus acciones cuelgan de la ventana.
        for action in (
            self.browse_model_action,
            self.image_action,
            self.reviewed_action,
            *self.save_actions.values(),
        ):
            self.addAction(action)

    # La flecha va en el texto: la que dibuja el estilo se pierde en cuanto la hoja de
    # estilos toca el borde del boton, y sin ella nada dice que ahi hay un menu.
    def menu_button(self, text: str, menu: QMenu) -> QToolButton:
        button = QToolButton()
        button.setText(f"{text} {CARET}")
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return button

    # La vista de espectrograma: el mensaje de bienvenida hasta que haya audio, y con audio
    # el espectrograma, la barra de tiempo y la leyenda con el filtro de score.
    def build_review(self) -> QWidget:
        self.placeholder = QLabel(PLACEHOLDER)
        self.placeholder.setObjectName("placeholder")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # El espectrograma con el scroll de la banda pegado a su derecha.
        self.spectrogram = QWidget()
        with_scroll = QHBoxLayout(self.spectrogram)
        with_scroll.setContentsMargins(0, 0, 0, 0)
        with_scroll.setSpacing(4)
        with_scroll.addWidget(self.plot, 1)
        with_scroll.addWidget(self.band.bar)
        self.canvas = QStackedWidget()
        self.canvas.addWidget(self.placeholder)
        self.canvas.addWidget(self.spectrogram)

        # El ancho de ventana (en el transporte) y la altura de banda, lado a lado.
        time_and_band = QHBoxLayout()
        time_and_band.setContentsMargins(0, 0, 0, 0)
        time_and_band.setSpacing(12)
        time_and_band.addWidget(self.transport, 1)
        time_and_band.addWidget(self.band.heights)

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(time_and_band)
        layout.addWidget(self.build_legend())
        layout.addWidget(self.review_bar)
        self.review_bar.hide()
        page = QWidget()
        page.setLayout(layout)
        return page

    # Leyenda de las capas y, sólo cuando hay detecciones, su umbral y el salto entre ellas.
    def build_legend(self) -> QWidget:
        self.layers = Layers([(s, COLORS[s], STYLES[s], WIDTHS[s]) for s in SOURCES])
        self.layers.changed.connect(self.draw_boxes)

        # De lo que pide al modelo hasta 1, en el paso del protocolo; arranca en el umbral común.
        thresholds = score_grid(DETECT_THRESHOLD, 1.0)
        self.score = Slider(
            "Score ≥", thresholds, thresholds.index(SCORE_THRESHOLD), "{:.2f}", label_width=52
        )
        self.score.setFixedWidth(SCORE_WIDTH)
        self.score.changed.connect(lambda: self.session.set_score(self.score.value()))
        self.prev_button = self.chevron("◀", "Previous detection (P)", lambda: self.jump(-1))
        self.next_button = self.chevron("▶", "Next detection (N)", lambda: self.jump(1))
        self.review_button = QToolButton()
        self.review_button.setText("Review")
        self.review_button.setCheckable(True)
        self.review_button.setToolTip("Go through the visible detections one by one")
        self.review_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.review_button.toggled.connect(self.toggle_review)

        self.model_tools = QWidget()
        tools = QHBoxLayout(self.model_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(8)
        tools.addWidget(self.score)
        tools.addWidget(self.prev_button)
        tools.addWidget(self.next_button)
        tools.addWidget(self.review_button)

        self.legend = QWidget()
        row = QHBoxLayout(self.legend)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        row.addWidget(self.layers)
        row.addStretch(1)
        row.addWidget(self.model_tools)
        return self.legend

    def chevron(self, text: str, tip: str, action) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(lambda: action())
        return button

    def show_help(self) -> None:
        rows = "".join(
            f"<tr><td colspan='2' style='padding-top:12px'><b>{section}</b></td></tr>"
            + "".join(
                f"<tr><td style='padding-right:20px'><tt>{keys}</tt></td><td>{what}</td></tr>"
                for keys, what in entries
            )
            for section, entries in HELP.items()
        )
        box = QMessageBox(self)
        box.setWindowTitle("Controls")
        box.setTextFormat(Qt.TextFormat.RichText)
        # QMessageBox se ajusta al texto: sin un mínimo, las frases se parten en dos.
        box.setStyleSheet(f"QLabel {{ min-width: {HELP_WIDTH}px; }}")
        box.setText(f"<table>{rows}</table>")
        box.exec()

    def build_status_bar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.readout = QLabel("")
        self.readout.setObjectName("readout")
        self.readout.setMinimumWidth(160)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.bar = QStatusBar()
        self.bar.addPermanentWidget(self.progress)
        self.bar.addPermanentWidget(self.readout)
        self.setStatusBar(self.bar)

    # --- Motor ------------------------------------------------------------------

    def start_engine(self) -> None:
        self.say("Loading the detection engine… you can open an audio meanwhile.")
        self.progress.setRange(0, 0)
        self.progress.show()
        self.preloader = Worker(preload)
        self.preloader.ok.connect(self.engine_ready)
        self.preloader.error.connect(self.engine_failed)
        self.preloader.start()

    def engine_ready(self, _) -> None:
        self.engine = True
        self.batch.set_engine(True)
        self.finish()
        if self.mode() == REVIEW and self.session.waveform is None:
            self.say("Drag and drop an audio to start, or press Ctrl+O.")
        else:
            self.say("Detection engine ready.")

    def engine_failed(self, message: str) -> None:
        # Sin motor el visor sigue sirviendo para mirar y escuchar tablas ya hechas.
        self.engine = False
        self.finish()
        self.say(f"The detection engine did not load ({message}). Viewing still works.")

    # --- Estado -----------------------------------------------------------------

    def say(self, message: str) -> None:
        self.bar.showMessage(message)

    def fail(self, message: str) -> None:
        self.say("Error.")
        QMessageBox.critical(self, "Error", message)

    def mode(self) -> str:
        return BATCH if self.pages.currentWidget() is self.batch else REVIEW

    def set_mode(self, mode: str) -> None:
        self.mode_actions[mode].setChecked(True)
        leaving_review = mode == BATCH and self.mode() == REVIEW
        if leaving_review:
            self.dock_shown = self.table.isVisible()
            self.table.hide()
        self.pages.setCurrentWidget(self.batch if mode == BATCH else self.pages.widget(0))
        for action in self.review_only:
            action.setVisible(mode == REVIEW)
        if mode == REVIEW and self.dock_shown:
            self.table.show()
            self.dock_shown = False
        self.sync_controls()

    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    # Lo que no se puede usar todavía no se muestra apagado: se muestra cuando sirve.
    def sync_controls(self) -> None:
        busy = self.busy()
        loaded = self.session.waveform is not None
        detections = self.session.tables[DETECTIONS] is not None
        model = self.session.model_path
        engine = bool(self.engine)
        review = self.mode() == REVIEW
        self.open_audio_action.setEnabled(not busy)
        self.open_table_action.setEnabled(not busy and loaded)
        self.picker.setEnabled(not busy and not self.batch.running())
        self.batch.set_blocked(busy)
        self.browse_model_action.setEnabled(review and self.picker.isEnabled())
        self.run_action.setEnabled(
            review
            and not busy
            and not self.batch.running()
            and loaded
            and engine
            and model is not None
        )
        self.run_action.setToolTip(
            "Open an audio first (Ctrl+R)"
            if not loaded
            else "Choose a model first (Ctrl+R)"
            if model is None
            else "Waiting for the detection engine (Ctrl+R)"
            if not engine
            else f"Run {model.parent.name} over the open audio (Ctrl+R)"
        )
        self.clear_action.setEnabled(review and not busy and detections)
        self.image_action.setEnabled(review and not busy and loaded)
        for source, action in self.save_actions.items():
            action.setEnabled(review and not busy and self.session.tables[source] is not None)
        self.reviewed_action.setEnabled(
            review and not busy and self.reviewer.counts().get(ACCEPTED, 0) > 0
        )
        self.viewer_button.setEnabled(loaded)
        self.export_button.setEnabled(review and loaded)
        self.canvas.setCurrentWidget(self.spectrogram if loaded else self.placeholder)
        self.transport.setVisible(loaded)
        self.legend.setVisible(loaded)
        self.model_tools.setVisible(detections)

    def on_changed(self) -> None:
        self.draw_boxes()
        self.sync_controls()

    def start(self, task, done, message: str, reports: bool = False) -> None:
        if self.busy():
            return
        self.say(message)
        self.worker = Worker(task, reports)
        self.worker.ok.connect(done)
        self.worker.error.connect(self.fail)
        self.worker.progress.connect(self.show_progress)
        self.worker.finished.connect(self.finish)
        self.worker.start()
        self.sync_controls()
        self.progress.setRange(0, 0)  # indeterminado hasta el primer reporte
        self.progress.show()

    def show_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def finish(self) -> None:
        # La barra se queda mientras el motor o una tarea sigan cargando.
        if not self.busy() and self.engine is not None:
            self.progress.hide()
        self.sync_controls()

    # --- Apertura de archivos ---------------------------------------------------

    def open_audio(self) -> None:
        folder = str(self.session.audio_path.parent) if self.session.audio_path else ""
        chosen, _ = QFileDialog.getOpenFileName(self, "Open audio", folder, f"Audio ({AUDIO})")
        if chosen:
            self.load_audio(Path(chosen))

    def open_table(self) -> None:
        if self.session.audio_path is None:
            self.say("Open an audio first: tables are drawn over it.")
            return
        folder = str(self.session.audio_path.parent)
        chosen, _ = QFileDialog.getOpenFileName(self, "Open table", folder, f"Tables ({TABLES})")
        if chosen:
            self.load_table(Path(chosen))

    def browse_model(self) -> None:
        folder = str(MODELS_DIR if MODELS_DIR.is_dir() else PROJECT_DIR)
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Choose a model", folder, f"Checkpoint ({MODELS})"
        )
        if chosen:
            self.picker.select(Path(chosen))

    def add_model(self, archive: Path | None = None) -> None:
        if archive is None:
            chosen, _ = QFileDialog.getOpenFileName(
                self, "Add model from zip", str(PROJECT_DIR), "Model zip (*.zip)"
            )
            if not chosen:
                return
            archive = Path(chosen)
        self.start(lambda: add_model_zip(archive), self.model_added, f"Adding {archive.name}…")

    def model_added(self, names: str) -> None:
        self.picker.reload()
        self.say(f"Added {names}: choose it in the Model list.")

    # Lo que se suelta en la ventana se enruta por extensión; una carpeta va a Batch.
    def open_path(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if path.is_dir():
            self.set_mode(BATCH)
            self.batch.set_folder(path)
        elif suffix in AUDIO_SUFFIXES:
            self.set_mode(REVIEW)
            self.load_audio(path)
        elif suffix in TABLE_SUFFIXES:
            self.set_mode(REVIEW)
            if self.session.audio_path is None:
                self.say("Open an audio first: tables are drawn over it.")
            else:
                self.load_table(path)
        elif is_checkpoint(path):
            self.picker.select(path)
        elif suffix == MODEL_ZIP_SUFFIX:
            self.add_model(path)
        else:
            self.say(f"Cannot open '{path.name}'.")

    def load_audio(self, path: Path) -> None:
        self.start(
            lambda: (path, load_audio(path, P.target_sr)),
            self.audio_loaded,
            f"Loading {path.name}…",
        )

    def load_table(self, path: Path) -> None:
        self.start(lambda: read_table(path), self.table_loaded, f"Loading {path.name}…")

    def model_chosen(self, path: Path | None) -> None:
        self.session.model_path = path
        operating = None if path is None else read_operating_point(path.parent)
        self.batch.set_model(path, operating)
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

    # Desde Batch: la grabación con la tabla que el modelo le dejó, si la tiene.
    def open_result(self, path: Path) -> None:
        table = output_for(path)
        self.pending_table = table if table.is_file() else None
        self.set_mode(REVIEW)
        self.load_audio(path)

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
        if source == DETECTIONS:
            self.say(f"{len(table)} detections loaded (Score column found).")
        else:
            self.say(f"{len(table)} annotations loaded.")

    def detections_ready(self, table: pd.DataFrame) -> None:
        # Si la corrida pasó por `compare_models.py`, el slider arranca en el umbral que eligió
        # en val (su `operating_point.json`); si no, se queda donde estaba.
        operating = table.attrs.get("operating_point")
        if operating is not None:
            self.score.set_value(operating)
        self.session.set_table(DETECTIONS, table)
        self.say(
            f"{len(table)} detections from the model."
            + ("" if operating is None else f"   Operating point: {operating:.2f}")
        )

    # --- Recorrido --------------------------------------------------------------

    def jump(self, direction: int) -> None:
        # Centra la ventana en la primera detección que no esté en pantalla. La referencia
        # es el borde y no el centro: al principio y al final del audio la barra no puede
        # centrar la caja, y con el centro la misma detección volvía a salir elegida.
        table = self.session.visible(DETECTIONS)
        if table is None or table.empty:
            return
        start, stop = self.transport.time_window()
        times = table[BEGIN].to_numpy()
        candidates = times[times >= stop] if direction > 0 else times[times < start]
        if candidates.size == 0:
            self.say("No more detections that way.")
            return
        target = float(candidates[0] if direction > 0 else candidates[-1])
        self.transport.center(target)

    def focus(self, row) -> None:
        if row is None:
            self.plot.set_highlight(None)
            return
        self.frame(row)
        self.plot.set_highlight((row.begin, row.end, row.low, row.high))

    # Trae la caja a la vista sin mover nada que ya la muestre: el tiempo se centra si se sale
    # de la ventana y la banda se reencuadra si se sale de la visible.
    def frame(self, row: Row) -> None:
        start, stop = self.transport.time_window()
        if not (start <= row.begin and row.end <= stop):
            self.transport.center(0.5 * (row.begin + row.end))
        low, high = self.band.current
        if not (low <= row.low and row.high <= high):
            self.band.frame(row.low, row.high)

    def toggle_review(self, on: bool) -> None:
        if on:
            self.reviewer.start()
        else:
            self.reviewer.stop()

    def sync_review(self) -> None:
        active = self.reviewer.active
        self.review_bar.setVisible(active)
        if self.review_button.isChecked() != active:
            self.review_button.blockSignals(True)
            self.review_button.setChecked(active)
            self.review_button.blockSignals(False)
        if active:
            counts = self.reviewer.counts()
            journal = self.reviewer.journal
            self.say(
                f"Review: {counts.get(ACCEPTED, 0)} accepted, {counts.get(REJECTED, 0)} rejected"
                + ("" if journal is None else f"   Journal: {journal.path}")
            )
        self.sync_controls()

    def track(self, seconds: float, hz: float) -> None:
        if self.session.waveform is not None:
            self.readout.setText(f"{seconds:.3f} s   {hz:,.0f} Hz")

    # --- Dibujo -----------------------------------------------------------------

    def refresh(self) -> None:
        self.draw_spectrogram()
        self.draw_boxes()

    def draw_spectrogram(self) -> None:
        # El tramo visible ya lo dice el eje: la barra de estado se guarda para lo demás.
        start = self.transport.time_window()[0]
        self.plot.draw(
            start,
            self.transport.span(),
            self.view.brightness.value(),
            self.view.contrast.value(),
        )

    def draw_boxes(self) -> None:
        start, stop = self.transport.time_window()
        # Una capa apagada entra como None: ni se dibuja ni se cuenta. Las anotaciones se
        # rotulan en el borde de arriba y las detecciones en el de abajo.
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
        shown = self.plot.draw_boxes(layers, start, stop)
        for source, count in zip(SOURCES, shown, strict=True):
            table = self.session.visible(source)
            self.layers.set_state(source, count, 0 if table is None else len(table))

    # --- Exportacion ------------------------------------------------------------

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
            self.fail(f"Could not save the image:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"Image saved as {path.name}")

    # Se guardan las cajas visibles: las detecciones bajo el score del slider no van.
    def export_table(self, source: str) -> None:
        table = self.session.visible(source)
        if table is None:
            return
        suffix = f".{'detections' if source == DETECTIONS else 'annotations'}.txt"
        path = self.save_path(f"Save {source.lower()}", suffix, "Raven (*.txt);;CSV (*.csv)")
        if path is None:
            return
        table = renumber(table.copy())
        try:
            table.to_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", index=False)
        except Exception as exc:
            self.fail(f"Could not save the table:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"{len(table)} rows saved to {path.name}")

    def export_reviewed(self) -> None:
        journal = self.reviewer.journal
        if journal is None:
            return
        path = self.save_path("Save reviewed boxes", ".reviewed.txt", "Raven (*.txt);;CSV (*.csv)")
        if path is None:
            return
        try:
            table = journal.accepted()
            table.to_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", index=False)
        except Exception as exc:
            self.fail(f"Could not save the reviewed boxes:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"{len(table)} accepted boxes saved to {path.name}")

    # --- Eventos ----------------------------------------------------------------

    def dropped(self, event) -> Path | None:
        mime = event.mimeData() if event is not None else None
        urls = mime.urls() if mime is not None and mime.hasUrls() else []
        if not urls or self.busy():
            return None
        path = Path(urls[0].toLocalFile())
        known = {*AUDIO_SUFFIXES, *TABLE_SUFFIXES, *CHECKPOINT_SUFFIXES, MODEL_ZIP_SUFFIX}
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
        if a0 is None or self.mode() != REVIEW or self.session.waveform is None:
            if a0 is not None:
                super().keyPressEvent(a0)
            return
        key = a0.key()
        reviewing = self.reviewer.active and not a0.modifiers()
        if reviewing and key == Qt.Key.Key_A:
            self.reviewer.decide(ACCEPTED)
        elif reviewing and key == Qt.Key.Key_R:
            self.reviewer.decide(REJECTED)
        elif reviewing and key == Qt.Key.Key_S:
            self.reviewer.step(1)
        elif reviewing and key in (Qt.Key.Key_N, Qt.Key.Key_P):
            self.reviewer.step(1 if key == Qt.Key.Key_N else -1)
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
        elif key == Qt.Key.Key_N:
            self.jump(1)
        elif key == Qt.Key.Key_P:
            self.jump(-1)
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
                "Batch running",
                "A batch is still running. Stop it and quit?\nTables already written are kept.",
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


# El arranque (main.py) crea la QApplication y muestra el splash antes de importar este
# módulo; acá se termina de configurar y se abre la ventana con lo que venga por argumento.
def run(app: QApplication, paths: list[Path]) -> Viewer:
    app.setApplicationName(BASE_TITLE)
    app.setStyle("Fusion")
    style = app.style()
    if style is not None:
        app.setPalette(style.standardPalette())
    apply_style(app)
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
