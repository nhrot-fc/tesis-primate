import zipfile
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
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QTextBrowser,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.config import PROJECT_DIR, SCORE_THRESHOLD, P, score_grid
from data.raven import BEGIN, END, renumber
from inference.catalog import (
    AUDIO_SUFFIXES,
    CHECKPOINT_SUFFIXES,
    MODELS_DIR,
    is_checkpoint,
    output_for,
    read_operating_point,
)
from viewer.batch import BatchView
from viewer.controls import (
    Band,
    Layers,
    MenuButton,
    ModelPicker,
    Popup,
    SettingsPanel,
    Slider,
    TitleBlock,
    emphasize,
)
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
# Las dos vistas, como las dos ventanas de Raven: una grabación o una carpeta entera.
SPECTROGRAM, BATCH = "Spectrogram", "Batch"
SCORE_WIDTH = 200
PAGE_MIN_WIDTH = 460
# La ayuda es una ventana aparte con scroll: la lista es más alta que muchas pantallas.
HELP_SIZE = (640, 620)
# El emblema de la bienvenida: el mismo dibujo que el icono del .exe (deploy/build_windows.py).
EMBLEM_SIZE = 96
EMBLEM_BACKGROUND, EMBLEM_STROKE = "#1c1b22", 5
# En orden de preferencia; si no hay ninguna se queda la del escritorio.
UI_FONTS = ("Inter", "Cantarell", "Noto Sans", "Ubuntu", "DejaVu Sans")
UI_POINT_SIZE = 10
# Al encuadrar una caja en revisión: la banda mide al menos el doble que la caja.
REVIEW_BAND_MARGIN = 2.0

AUDIO = "*.wav *.flac *.mp3 *.WAV *.FLAC *.MP3"
TABLES = "*.txt *.csv"
MODELS = "*.pth *.pt"
TABLE_SUFFIXES = {".txt", ".csv"}
MODEL_ZIP_SUFFIX = ".zip"
# El manual, si está: junto al programa en el paquete, en docs/ en el repo.
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
        ("Zip", "drop a model zip on the window to add it next to the program"),
    ],
    "Detect": [
        ("Ctrl+R", "run the model over the open recording"),
        ("Ctrl+Shift+R", "the Batch view: Run leaves a <tt>.detections.txt</tt> next to every "
                         "recording of the folder, with the boxes at or above its Score"),
        ("Score ≥", "in the Spectrogram view only hides the weaker detections; Save keeps what "
                    "is visible. It starts at the model's operating point"),
        ("Ctrl+L", "clear the detections"),
    ],
    "Look": [
        ("Wheel", "scroll through the audio"),
        ("Ctrl+1 / Ctrl+3", "zoom in and out in time (Ctrl + wheel does the same): "
                            "Window is how much time fits on screen, 0.25 to 30 s"),
        ("Ctrl+↑ / Ctrl+↓", "zoom in and out in frequency (Shift + wheel, around the pointer): "
                            "Band is how much of the range fits"),
        ("↑ ↓", "move the band; F shows it all"),
        ("Settings", "brightness, contrast, volume and the audio output"),
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
        ("N / P", "next and previous detection"),
    ],
    "Boxes": [
        ("1 / 2", "show or hide the Annotations and Detections layers"),
        ("Ctrl+E", "table of boxes; click a row to frame it"),
        ("Del", "in the table, remove the selected boxes"),
        ("Save", "keeps the boxes that are visible"),
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

# Lo mínimo para que la ventana no sea la suma de los ajustes de fábrica de cada control:
# espaciado de la barra, el botón de acento y los tres papeles del texto. Todo lo demás lo
# dibuja el escritorio, que es lo que el usuario ya reconoce.
QSS = """
QToolBar {{
    border: 0;
    border-bottom: 1px solid {line};
    padding: 6px 10px;
    spacing: 4px;
}}
QToolBar QToolButton {{
    padding: 5px 11px;
    border: 0;
    border-radius: 5px;
}}
QToolBar QToolButton:hover:enabled {{ background: {hover}; }}
QToolBar QToolButton:pressed:enabled {{ background: {press}; }}
QToolBar QToolButton:checked {{ background: {press}; }}
/* La flecha va escrita en el texto del botón: la del estilo se dibuja suelta al lado. */
QToolBar QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolBar QComboBox {{ padding: 3px 8px; }}

QToolButton#primary, QPushButton#primary {{
    background: {accent};
    color: {accent_text};
    border: 0;
    border-radius: 5px;
    padding: 5px 14px;
    font-weight: 600;
}}
QToolButton#primary:hover:enabled, QPushButton#primary:hover:enabled {{
    background: {accent_hover};
}}
QToolButton#primary:disabled, QPushButton#primary:disabled {{
    background: {press};
    color: {muted};
    font-weight: 400;
}}

#title {{ font-weight: 600; }}
#subtitle, #hint, #placeholder, #dockName {{ color: {muted}; }}
#welcomeHeadline {{ font-size: {headline}pt; }}
#readout, #clock {{ font-family: "{mono}"; color: {muted}; }}
#dockTitle {{ border-bottom: 1px solid {line}; }}
QToolButton#dockClose {{ border: 0; padding: 2px; }}

QStatusBar {{ border-top: 1px solid {line}; }}
QStatusBar, QStatusBar QLabel {{ color: {muted}; }}
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

    palette = app.palette()
    window = palette.window().color()
    accent = palette.highlight().color()
    dark = window.lightness() < 128
    app.setStyleSheet(
        QSS.format(
            line=(window.lighter(140) if dark else window.darker(112)).name(),
            hover=(window.lighter(125) if dark else window.darker(107)).name(),
            press=(window.lighter(145) if dark else window.darker(115)).name(),
            muted=(window.lighter(220) if dark else window.darker(165)).name(),
            accent=accent.name(),
            accent_hover=accent.lighter(112).name(),
            accent_text=palette.highlightedText().color().name(),
            headline=UI_POINT_SIZE + 4,
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


# La fila del transporte, con los dos zooms al final.
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

    # Estrecha, la barra de tiempo se queda sin sitio y la fila deja de servir; antes de eso
    # se suelta el reloj, que el eje y el cabezal también dicen.
    @override
    def resizeEvent(self, a0) -> None:
        super().resizeEvent(a0)
        self.clock.setVisible(self.width() >= self.CLOCK_FROM)


# La ventana sin grabación: una frase, una aclaración y los dos modos de empezar. Un sitio
# vacío que no explica nada es el momento en que más se necesita la explicación.
class Welcome(QWidget):
    def __init__(self, open_audio, open_folder) -> None:
        super().__init__()
        emblem = QLabel()
        emblem.setPixmap(emblem_pixmap(EMBLEM_SIZE))
        emblem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        headline = QLabel(WELCOME[0])
        headline.setObjectName("welcomeHeadline")
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail = QLabel(WELCOME[1])
        detail.setObjectName("placeholder")
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)

        open_button = QPushButton("Open audio…")
        open_button.setObjectName("primary")
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


# Un cuadro oscuro con las cajas de las dos capas, pintado a mano para no depender de un
# archivo: lo mismo que el icono del programa, así la ventana vacía ya dice de qué va.
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
        self.setMinimumSize(1000, 600)
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        # La ventana no deja nada escrito al cerrarse: cada arranque es igual al anterior.
        # La única carpeta que se lleva memoria es la de los diálogos, y sólo mientras dura
        # la sesión.
        self.last_folder = ""

        self.session = Session()
        self.worker: Worker | None = None
        self.engine: bool | None = None  # None mientras carga
        self.pending_table: Path | None = None  # tabla que se abre en cuanto cargue su audio
        self.pending_audio: Path | None = None  # la elegida en Batch mientras cargaba otra
        self.dock_shown = False  # si el panel de cajas estaba abierto al pasar a Batch
        self.help_dialog: QDialog | None = None  # se arma la primera vez que se pide

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
        self.reviewer = Reviewer(self.session, self.plot, self.review_bar, self.present)
        self.reviewer.changed.connect(self.sync_review)

        self.batch = BatchView()
        self.batch.open_requested.connect(self.open_recording)
        self.batch.finished_file.connect(self.table_written)
        self.batch.said.connect(self.say)
        self.batch.state_changed.connect(self.sync_controls)

        self.build_actions()
        self.build_menus()
        self.build_toolbar()
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

    # --- Construccion -----------------------------------------------------------

    # Cada acción se define una vez y aparece donde toca: en el menú, en la barra o en las
    # dos. Los atajos van con la acción, así el menú los enseña solo.
    def build_actions(self) -> None:
        # Las dos vistas: un botón marcado por cada una.
        self.modes = QActionGroup(self)
        self.mode_actions: dict[str, QAction] = {}
        for mode, tip in ((SPECTROGRAM, "One recording at a time"), (BATCH, "A whole folder")):
            action = QAction(mode, self.modes)
            action.setCheckable(True)
            action.setToolTip(tip)
            action.triggered.connect(lambda _, name=mode: self.set_mode(name))
            self.mode_actions[mode] = action

        # Tres aperturas, como en Raven: el audio, la tabla que va encima y la carpeta.
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
        # La F sola se atiende en keyPressEvent, para que no dispare mientras se escribe una
        # especie; el tabulador en el texto la enseña en el menú sin registrarla.
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
        self.add_model_action = self.action("Add model from zip…", "", self.add_model)

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

    # Una barra de menús como la de Raven o Audacity: todo lo que se puede hacer, con su
    # atajo al lado. La barra de herramientas de abajo es el subconjunto de cada día.
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
            detect_menu.addAction(self.add_model_action)
        help_menu = bar.addMenu("&Help")
        if help_menu is not None:
            help_menu.addAction(self.help_action)
            help_menu.addAction(self.manual_action)

    # Tres zonas, como la barra de una ventana de Finder: a la izquierda las vistas y lo que
    # trae archivos, al centro qué está abierto, a la derecha lo que se hace con ello. Cada
    # grupo va separado del siguiente, y ningún botón repite lo que hace el de al lado.
    def build_toolbar(self) -> None:
        open_menu = QMenu(self)
        open_menu.addAction(self.open_audio_action)
        open_menu.addAction(self.open_table_action)
        open_menu.addAction(self.open_folder_action)
        self.open_button = MenuButton("Open", open_menu, "Open audio, annotations or a folder")

        self.title_block = TitleBlock()

        self.picker = ModelPicker()
        self.picker.chosen.connect(self.model_chosen)
        self.picker.browse.connect(self.browse_model)
        self.picker.add.connect(self.add_model)
        self.picker_label = QLabel("Model")
        self.picker_label.setObjectName("hint")
        self.picker_label.setToolTip(self.picker.toolTip())

        self.run_button = QToolButton()
        self.run_button.setDefaultAction(self.run_action)
        self.run_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.settings_button = Popup(
            "Settings", self.settings, "Brightness, contrast, volume and audio output"
        )

        export_menu = QMenu(self)
        for action in self.save_actions.values():
            export_menu.addAction(action)
        export_menu.addSeparator()
        export_menu.addAction(self.image_action)
        self.export_button = MenuButton("Save", export_menu, "Save a table or an image")

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for action in self.mode_actions.values():
            toolbar.addAction(action)
        toolbar.addSeparator()
        toolbar.addWidget(self.open_button)
        # Lo que sólo tiene sentido con un espectrograma delante se esconde en Batch.
        # (`addAction(QAction)` no devuelve nada; `addSeparator` y `addWidget` sí.)
        # El título es elástico y se queda con el hueco; en Batch, donde no hay título, el
        # hueco lo ocupa el espaciador para que el resto no se corra.
        self.spectrogram_only: list[QAction] = []
        placed = toolbar.addWidget(self.title_block)
        if placed is not None:
            self.spectrogram_only.append(placed)
        self.batch_only: list[QAction] = []
        placed = toolbar.addWidget(self.stretch())
        if placed is not None:
            self.batch_only.append(placed)
        toolbar.addWidget(self.picker_label)
        toolbar.addWidget(self.picker)
        placed = toolbar.addWidget(self.run_button)
        if placed is not None:
            self.spectrogram_only.append(placed)
        toolbar.addSeparator()
        for widget in (self.settings_button, self.export_button):
            placed = toolbar.addWidget(widget)
            if placed is not None:
                self.spectrogram_only.append(placed)
        separator = toolbar.addSeparator()
        if separator is not None:
            self.spectrogram_only.append(separator)
        if self.boxes_action is not None:
            toolbar.addAction(self.boxes_action)
            self.spectrogram_only.append(self.boxes_action)
        toolbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addToolBar(toolbar)

    # Hueco elástico: dos de estos, uno a cada lado, dejan el título en el centro.
    def stretch(self) -> QWidget:
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return spacer

    # La vista de espectrograma: el mensaje de bienvenida hasta que haya audio; con audio, el
    # espectrograma, la barra de tiempo y una fila de contexto (leyenda y score, o la revisión).
    def build_page(self) -> QWidget:
        self.welcome = Welcome(self.open_audio, self.open_folder)
        # El espectrograma con el scroll de la banda pegado a su derecha.
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
        # El centro manda sobre el panel: al estrechar la ventana encoge él primero.
        page.setMinimumWidth(PAGE_MIN_WIDTH)
        return page

    # Una sola fila debajo del transporte, y sólo cuando hay algo que decir: la leyenda de
    # las capas a la izquierda, el umbral y Review a la derecha. Durante la revisión la fila
    # entera pasa a ser la revisión: es lo único que se está haciendo.
    def build_context(self) -> QWidget:
        self.layers = Layers([(s, COLORS[s], STYLES[s], WIDTHS[s]) for s in SOURCES])
        self.layers.changed.connect(self.draw_boxes)

        # Sólo esconde cajas: nada de lo que se escribe depende de esto. Arranca en el punto
        # de operación del modelo, que es lo que el modelo llama una detección.
        thresholds = score_grid(DETECT_THRESHOLD, 1.0)
        self.score = Slider(
            "Score ≥", thresholds, thresholds.index(SCORE_THRESHOLD), "{:.2f}", label_width=52
        )
        self.score.setMaximumWidth(SCORE_WIDTH)
        self.score.setToolTip("Hide detections below this score")
        self.score.changed.connect(lambda: self.session.set_score(self.score.value()))
        self.review_button = QToolButton()
        self.review_button.setText("Review")
        self.review_button.setToolTip("Go through the visible detections one by one")
        self.review_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.review_button.clicked.connect(self.reviewer.start)

        # Una fila con dos caras y nunca las dos a la vez: mirar o revisar. Mirando, el
        # umbral a la izquierda y el paso siguiente a la derecha, con la fila entera de por
        # medio; revisando, la revisión se la queda toda porque es lo único que se hace.
        self.model_tools = QWidget()
        row = QHBoxLayout(self.model_tools)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        row.addWidget(self.layers)
        row.addWidget(self.score)
        row.addStretch(1)
        row.addWidget(self.review_button)

        self.context = QStackedWidget()
        self.context.addWidget(self.model_tools)
        self.context.addWidget(self.review_bar)
        return self.context

    def show_settings(self) -> None:
        self.settings_button.showMenu()

    def manual(self) -> Path | None:
        return next((path for path in MANUAL_PATHS if path.is_file()), None)

    def open_manual(self) -> None:
        path = self.manual()
        if path is None:
            self.say("The manual (Manual.pdf) is not next to the program.")
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # Una ventana aparte, con scroll y sin bloquear: se deja abierta al lado mientras se
    # aprende, y F1 la trae al frente. Un QMessageBox no hace scroll y esta lista es más alta
    # que la pantalla de un portátil.
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
            close.setObjectName("primary")
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

    # A la izquierda lo que acaba de pasar; a la derecha, mientras dure, qué está corriendo
    # con su barra, y las coordenadas del puntero. Una barra de progreso sin nombre al lado
    # no dice qué espera.
    def build_status_bar(self) -> None:
        self.task = QLabel("")
        self.task.setObjectName("hint")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setTextVisible(False)
        self.readout = QLabel("")
        self.readout.setObjectName("readout")
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

    # --- Arranque ---------------------------------------------------------------

    # El primero de la lista queda elegido: con un solo modelo instalado, que es lo normal,
    # nunca hay que tocar el desplegable.
    def first_model(self) -> None:
        listed = self.picker.paths()
        if listed:
            self.picker.select(listed[0])

    # La carpeta de los diálogos: la del audio abierto, si no la última de esta sesión.
    def folder(self) -> str:
        if self.session.audio_path is not None:
            return str(self.session.audio_path.parent)
        return self.last_folder

    # --- Motor ------------------------------------------------------------------

    def start_engine(self) -> None:
        self.say("Loading the detection engine… you can open a recording meanwhile.")
        self.progress.setRange(0, 0)
        self.show_task("Loading detection engine")
        self.preloader = Worker(preload)
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
        # Sin motor el visor sigue sirviendo para mirar y escuchar tablas ya hechas.
        self.engine = False
        self.settings.reload_outputs()
        self.finish()
        self.say(f"The detection engine did not load ({message}). Viewing still works.")

    # --- Estado -----------------------------------------------------------------

    def say(self, message: str) -> None:
        self.bar.showMessage(message)

    def fail(self, message: str) -> None:
        self.say("Error.")
        QMessageBox.critical(self, "Error", message)

    # Ocupado de verdad: el hilo corre y todavía no entregó. Entre que entrega y termina hay
    # un instante en que `isRunning` sigue en alto; ahí no se está ocupado, se está saliendo.
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
        for action in self.spectrogram_only:
            action.setVisible(mode == SPECTROGRAM)
        for action in self.batch_only:
            action.setVisible(mode == BATCH)
        if mode == SPECTROGRAM and self.dock_shown:
            self.table.show()
            self.dock_shown = False
        self.sync_controls()

    # Lo que no se puede usar todavía no se muestra apagado: se muestra cuando sirve.
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
        self.add_model_action.setEnabled(self.picker.isEnabled())
        self.run_action.setEnabled(
            spectrogram and not busy and not batch and loaded and engine and model is not None
        )
        self.run_action.setToolTip(
            "Open a recording first (Ctrl+R)"
            if not loaded
            else "Choose a model first (Ctrl+R)"
            if model is None
            else "Waiting for the detection engine (Ctrl+R)"
            if not engine
            else "Batch is running (Ctrl+R)"
            if batch
            else f"Run {model.parent.name} over the open recording (Ctrl+R)"
        )
        self.batch_action.setEnabled(not batch)
        self.clear_action.setEnabled(spectrogram and not busy and detections)
        self.image_action.setEnabled(spectrogram and not busy and loaded)
        for source, action in self.save_actions.items():
            action.setEnabled(spectrogram and not busy and self.session.tables[source] is not None)
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
        self.export_button.setEnabled(loaded)
        self.canvas.setCurrentWidget(self.spectrogram if loaded else self.welcome)
        self.controls.setVisible(loaded)
        self.context.setVisible(loaded and (detections or reviewing))
        self.context.setCurrentIndex(1 if reviewing else 0)
        emphasize(self.run_button, self.run_action.isEnabled() and not detections)
        emphasize(self.review_button, detections and not reviewing)
        self.show_title()

    # Lo que está abierto, en el centro de la barra: el nombre y, debajo, su duración y lo
    # que se lleva encontrado. La barra de título del sistema queda para el gestor de ventanas.
    def show_title(self) -> None:
        path = self.session.audio_path
        if path is None:
            self.title_block.set_document(None)
            return
        parts = [f"{self.session.duration:.0f} s"]
        for source in SOURCES:
            table = self.session.visible(source)
            if table is not None and len(table):
                parts.append(f"{len(table)} {source.lower()}")
        self.title_block.set_document(path.name, " · ".join(parts))

    def on_changed(self) -> None:
        self.draw_boxes()
        self.sync_controls()

    def start(self, task, done, message: str, reports: bool = False) -> None:
        if self.busy():
            return
        if self.worker is not None:
            self.worker.wait()  # ya entregó: lo que le queda es salir
        self.say(message)
        self.worker = Worker(task, reports)
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
        # La barra se queda mientras el motor o una tarea sigan cargando.
        if self.busy():
            pass
        elif self.engine is None:
            self.show_task("Loading detection engine")
        else:
            self.show_task(None)
        self.sync_controls()
        if self.pending_audio is not None and not self.busy():
            path, self.pending_audio = self.pending_audio, None
            self.load_audio(path)

    # --- Apertura de archivos ---------------------------------------------------

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

    # Lo que se abre o se suelta en la ventana se enruta por extensión; una carpeta va a Batch.
    def open_path(self, path: Path) -> None:
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
        elif suffix == MODEL_ZIP_SUFFIX:
            self.add_model(path)
        else:
            self.say(f"Cannot open '{path.name}'.")

    # Con el audio viene la tabla que el modelo le dejó al lado, si la hay.
    def load_audio(self, path: Path) -> None:
        if self.pending_table is None and output_for(path).is_file():
            self.pending_table = output_for(path)
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

    # Ctrl+Shift+R: la vista Batch es la que lleva la corrida, así que se muestra antes de
    # arrancar; sin carpeta, lo primero es elegir una.
    def run_batch(self) -> None:
        self.set_mode(BATCH)
        if not self.batch.listed():
            self.batch.browse()
        elif self.batch.can_run():
            self.batch.run()
        else:
            self.say(self.batch.why())

    # Doble clic en Batch: si otra está cargando, espera su turno.
    def open_recording(self, path: Path) -> None:
        self.set_mode(SPECTROGRAM)
        if path == self.session.audio_path:
            return
        if self.busy():
            self.pending_audio = path
        else:
            self.load_audio(path)

    # Batch terminó la grabación que está abierta: su tabla nueva entra sola.
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
        if source == DETECTIONS:
            self.say(f"{len(table)} detections loaded.")
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

    # Anterior / siguiente: en revisión es la caja en curso; si no, la primera fuera de pantalla.
    def skip(self, direction: int) -> None:
        if self.reviewer.active:
            self.reviewer.step(direction)
        else:
            self.jump(direction)

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

    # En revisión cada caja se presenta igual: encuadrada en tiempo y en banda, con el cabezal
    # en su inicio para que Space la haga sonar.
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
            self.settings.brightness.value(),
            self.settings.contrast.value(),
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
        self.plot.draw_boxes(layers, start, stop)
        counts = {s: len(t) if (t := self.session.visible(s)) is not None else 0 for s in SOURCES}
        both = all(counts.values())
        for source in SOURCES:
            self.layers.set_state(source, counts[source] if both else 0)
        # Las mismas capas, en miniatura, sobre la barra de tiempo.
        marks = [
            (float(begin), float(end), layer.color)
            for layer in layers
            if layer.table is not None
            for begin, end in zip(layer.table[BEGIN], layer.table[END], strict=True)
        ]
        self.transport.set_marks(marks)

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

    # Las teclas sueltas (sin modificador) del espectrograma. Las combinaciones con Ctrl son
    # acciones del menú y no pasan por acá.
    @override
    def keyPressEvent(self, a0) -> None:
        if a0 is None or self.session.waveform is None or self.mode() != SPECTROGRAM:
            if a0 is not None:
                super().keyPressEvent(a0)
            return
        key = a0.key()
        reviewing = self.reviewer.active and not a0.modifiers()
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
