import os
import sys
from pathlib import Path
from typing import override

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFontDatabase, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.config import SCORE_THRESHOLD, P, score_grid
from data.raven import BEGIN, END, HIGH, LOW, renumber
from viewer.controls import Layers, Popup, Slider, ViewPanel
from viewer.inference import DETECT_THRESHOLD, detect, preload
from viewer.plot import Layer, SpectrogramView
from viewer.session import (
    ACCEPTED,
    ANNOTATIONS,
    COLORS,
    DETECTIONS,
    FINDINGS,
    ISSUE,
    QUALITY,
    RECORDING,
    REJECTED,
    SOURCES,
    STYLES,
    VERDICT,
    WIDTHS,
    Session,
    common_label,
    issue,
    label,
    read_table,
)
from viewer.spectrogram import load_audio
from viewer.table import BoxTable
from viewer.tasks import Worker
from viewer.transport import Transport

BASE_TITLE = "Visor de espectrogramas"
SCORE_WIDTH = 220
HELP_WIDTH = 560
CARET = "\u25be"
# En orden de preferencia; si no hay ninguna se queda la del escritorio.
UI_FONTS = ("Inter", "Cantarell", "Noto Sans", "Ubuntu", "DejaVu Sans")
UI_POINT_SIZE = 10

AUDIO = "*.wav *.flac *.mp3 *.WAV *.FLAC *.MP3"
TABLES = "*.txt *.csv"
MODELS = "*.pth *.pt"
# Qué se abre, con qué atajo y con qué filtro de archivos.
OPEN = {
    "audio": ("Audio…", "Ctrl+O", f"Audio ({AUDIO})"),
    "table": ("Anotaciones o hallazgos…", "Ctrl+T", f"Tablas ({TABLES})"),
    "model": ("Modelo…", "Ctrl+M", f"Checkpoint ({MODELS})"),
}
SUFFIXES = {
    **dict.fromkeys((".wav", ".flac", ".mp3"), "audio"),
    **dict.fromkeys((".txt", ".csv"), "table"),
    **dict.fromkeys((".pth", ".pt"), "model"),
}
HELP = {
    "Ver": [
        ("Rueda", "recorre el audio"),
        ("Ctrl + rueda", "ancho de la ventana, de 0,25 a 30 s"),
        ("Shift + rueda", "acerca en frecuencia, alrededor del puntero"),
        ("↑ ↓", "sube y baja la banda visible"),
        ("F", "vuelve a la banda entera"),
        ("Vista", "brillo, contraste, volumen y los extremos de la banda"),
        ("", "el ancho de ventana y la banda sólo cambian si los cambias tú"),
    ],
    "Oír": [
        ("Espacio", "reproduce o pausa"),
        ("Clic", "lleva el cabezal a ese punto"),
        ("Volumen", "0 dB es la grabación llevada a su pico; súbelo para las llamadas lejanas"),
    ],
    "Moverse": [
        ("← →", "avanza y retrocede"),
        ("Re Pág / Av Pág", "una ventana entera"),
        ("Inicio / Fin", "principio y final del audio"),
        ("N / P", "detección siguiente y anterior"),
    ],
    "Revisar los hallazgos de CLOD": [
        (". / ,", "hallazgo siguiente y anterior; abre su grabación sin tocar tu zoom"),
        ("A", "aceptar: añade la caja propuesta o borra la anotación sobrante"),
        ("R", "rechazar: la anotación se queda como está"),
        ("1 2 3", "muestra u oculta cada capa de cajas"),
        ("Ctrl+E", "tabla de revisión"),
    ],
}

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
#reviewBar {{
    background: {panel};
    border: 1px solid {line};
    border-left: 3px solid {accent};
    border-radius: 6px;
}}
#reviewBar QPushButton {{ padding: 5px 14px; }}
#queue {{ font-weight: 600; }}
#hint {{ color: {muted}; }}
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
            panel=(window.lighter(112) if dark else window.lighter(103)).name(),
            muted=app.palette().placeholderText().color().name(),
            accent=COLORS[FINDINGS],
            mono=QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family(),
        )
    )


class Viewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(BASE_TITLE)
        self.setMinimumSize(900, 560)
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        self.session = Session()
        self.worker: Worker | None = None
        self.reviewing = -1  # posición en la cola de hallazgos, ordenada por Calidad
        self.pending: pd.Series | None = None  # hallazgo esperando a que cargue su audio

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
        self.view.band.changed.connect(lambda: self.plot.set_band(*self.view.band.values()))
        self.plot.banded.connect(self.view.band.set_values)

        self.table = BoxTable(self.session)
        self.table.picked.connect(self.focus)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.table)
        self.table.hide()

        self.build_toolbar()

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.transport)
        layout.addWidget(self.build_legend())
        layout.addWidget(self.build_review())
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.build_status_bar()
        self.session.changed.connect(self.on_changed)
        self.sync_controls()
        self.start_engine()

    # --- Construccion -----------------------------------------------------------

    def build_toolbar(self) -> None:
        self.open_actions = {}
        open_menu = QMenu(self)
        for kind, (text, shortcut, _) in OPEN.items():
            action = QAction(text, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(lambda _, name=kind: self.open_dialog(name))
            open_menu.addAction(action)
            self.open_actions[kind] = action
        opener = self.menu_button("Abrir", open_menu)

        self.run_action = QAction("Detectar", self)
        self.run_action.setShortcut(QKeySequence("Ctrl+R"))
        self.run_action.setToolTip("Pasa el modelo por la grabación abierta (Ctrl+R)")
        self.run_action.triggered.connect(self.run_model)

        self.viewer_button = Popup(
            f"Vista {CARET}", self.view, "Brillo, contraste, volumen y banda"
        )

        self.image_action = QAction("Imagen del tramo…", self)
        self.image_action.setShortcut(QKeySequence("Ctrl+S"))
        self.image_action.triggered.connect(self.export_image)
        self.save_actions = {source: QAction(f"{source}…", self) for source in SOURCES}
        for source, action in self.save_actions.items():
            action.triggered.connect(lambda _, name=source: self.export_table(name))
        export_menu = QMenu(self)
        export_menu.addAction(self.image_action)
        export_menu.addSeparator()
        for action in self.save_actions.values():
            export_menu.addAction(action)
        export = self.menu_button("Guardar", export_menu)

        self.help_action = QAction("Ayuda", self)
        self.help_action.setShortcut(QKeySequence("F1"))
        self.help_action.setToolTip("Qué hace cada control (F1)")
        self.help_action.triggered.connect(self.show_help)

        review = self.table.toggleViewAction()
        if review is not None:
            review.setText("Revisión")
            review.setToolTip("Tabla de anotaciones y detecciones (Ctrl+E)")
            review.setShortcut(QKeySequence("Ctrl+E"))

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toolbar.addWidget(opener)
        toolbar.addAction(self.run_action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(self.viewer_button)
        toolbar.addWidget(export)
        if review is not None:
            toolbar.addAction(review)
        toolbar.addAction(self.help_action)
        toolbar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.addToolBar(toolbar)
        # Los atajos del menu solo llegan si sus acciones cuelgan de la ventana.
        for action in (*self.open_actions.values(), self.image_action, *self.save_actions.values()):
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

    # Leyenda de las capas y, sólo cuando hay modelo, su umbral y el salto entre detecciones.
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
        self.prev_button = self.chevron("◀", "Detección anterior (P)", lambda: self.jump(-1))
        self.next_button = self.chevron("▶", "Detección siguiente (N)", lambda: self.jump(1))

        self.model_tools = QWidget()
        tools = QHBoxLayout(self.model_tools)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(8)
        tools.addWidget(self.score)
        tools.addWidget(self.prev_button)
        tools.addWidget(self.next_button)

        legend = QWidget()
        row = QHBoxLayout(legend)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        row.addWidget(self.layers)
        row.addStretch(1)
        row.addWidget(self.model_tools)
        return legend

    def chevron(self, text: str, tip: str, action) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.clicked.connect(lambda: action())
        return button

    # La barra de revisión: qué hallazgo se está mirando y las dos únicas decisiones que
    # caben sobre él. Sólo aparece cuando hay una cola cargada.
    def build_review(self) -> QWidget:
        self.queue = QLabel("")
        self.queue.setObjectName("queue")
        self.finding = QLabel("")
        self.finding.setObjectName("hint")

        self.back_button = self.chevron("◀", "Hallazgo anterior (,)", lambda: self.walk(-1))
        self.forward_button = self.chevron("▶", "Hallazgo siguiente (.)", lambda: self.walk(1))
        self.accept_button = QPushButton("Aceptar")
        self.accept_button.setToolTip("Añade la caja propuesta o borra la sobrante (A)")
        self.accept_button.clicked.connect(lambda: self.decide(ACCEPTED))
        self.reject_button = QPushButton("Rechazar")
        self.reject_button.setToolTip("La anotación se queda como está (R)")
        self.reject_button.clicked.connect(lambda: self.decide(REJECTED))
        for button in (self.accept_button, self.reject_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.review = QFrame()
        self.review.setObjectName("reviewBar")
        row = QHBoxLayout(self.review)
        row.setContentsMargins(12, 7, 10, 7)
        row.setSpacing(12)
        row.addWidget(self.queue)
        row.addWidget(self.finding)
        row.addStretch(1)
        row.addWidget(self.back_button)
        row.addWidget(self.accept_button)
        row.addWidget(self.reject_button)
        row.addWidget(self.forward_button)
        self.review.hide()
        return self.review

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
        box.setWindowTitle("Controles del visor")
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

    def start_engine(self) -> None:
        self.say("Preparando el motor de detección...")
        self.preloader = Worker(preload)
        self.preloader.ok.connect(self.engine_ready)
        self.preloader.error.connect(self.engine_failed)
        self.preloader.start()

    def engine_ready(self, _) -> None:
        self.sync_controls()
        if self.session.waveform is None:
            self.say("Abre un audio (Ctrl+O) o arrástralo a la ventana.")

    def engine_failed(self, message: str) -> None:
        # Sin motor el visor sigue sirviendo para mirar y escuchar tablas ya hechas.
        self.sync_controls()
        self.say(f"El motor de detección no cargó ({message}). El visor funciona igual.")

    # --- Estado -----------------------------------------------------------------

    def say(self, message: str) -> None:
        self.bar.showMessage(message)

    def fail(self, message: str) -> None:
        self.say("Error.")
        QMessageBox.critical(self, "Error", message)

    # Lo que no se puede usar todavía no se muestra apagado: se muestra cuando sirve.
    def sync_controls(self, busy: bool = False) -> None:
        loaded = self.session.waveform is not None
        detections = self.session.tables[DETECTIONS] is not None
        findings = self.session.tables[FINDINGS] is not None
        for action in self.open_actions.values():
            action.setEnabled(not busy)
        self.run_action.setEnabled(not busy and loaded and self.session.model_path is not None)
        self.image_action.setEnabled(not busy and loaded)
        for source, action in self.save_actions.items():
            action.setEnabled(not busy and self.session.tables[source] is not None)
        for widget in (self.transport, self.viewer_button):
            widget.setEnabled(loaded)
        self.model_tools.setVisible(detections)
        self.review.setVisible(findings)
        for button in (self.back_button, self.forward_button):
            button.setEnabled(not busy)
        for button in (self.accept_button, self.reject_button):
            button.setEnabled(not busy and self.reviewing >= 0)

    def on_changed(self) -> None:
        self.draw_boxes()
        self.sync_controls()

    def start(self, task, done, message: str, reports: bool = False) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.say(message)
        self.sync_controls(busy=True)
        self.progress.setRange(0, 0)  # indeterminado hasta el primer reporte
        self.progress.show()
        self.worker = Worker(task, reports)
        self.worker.ok.connect(done)
        self.worker.error.connect(self.fail)
        self.worker.progress.connect(self.show_progress)
        self.worker.finished.connect(self.finish)
        self.worker.start()

    def show_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def finish(self) -> None:
        self.progress.hide()
        self.sync_controls()

    # --- Apertura de archivos ---------------------------------------------------

    def open_dialog(self, kind: str) -> None:
        text, _, file_filter = OPEN[kind]
        folder = str(self.session.audio_path.parent) if self.session.audio_path else ""
        chosen, _ = QFileDialog.getOpenFileName(self, text.rstrip("…"), folder, file_filter)
        if chosen:
            self.load(kind, Path(chosen))

    # Lo que se suelta en la ventana se enruta por extensión.
    def open_path(self, path: Path) -> None:
        kind = SUFFIXES.get(path.suffix.lower())
        if kind is None:
            self.say(f"No sé abrir '{path.name}'.")
        else:
            self.load(kind, path)

    def load(self, kind: str, path: Path) -> None:
        if kind == "audio":
            self.start(
                lambda: (path, load_audio(path, P.target_sr)),
                self.audio_loaded,
                f"Cargando {path.name}...",
            )
        elif kind == "table":
            self.start(lambda: read_table(path), self.table_loaded, f"Cargando {path.name}...")
        elif kind == "model":
            self.session.model_path = path
            self.run_action.setToolTip(f"{path.name} (Ctrl+R)")
            self.say(f"Modelo listo: {path.name}   (Ctrl+R para detectar)")
            self.sync_controls()

    def run_model(self) -> None:
        audio, model = self.session.audio_path, self.session.model_path
        if audio is None or model is None:
            return
        self.start(
            lambda report: detect(audio, model, on_progress=report),
            self.detections_ready,
            f"Ejecutando '{model.name}'...",
            reports=True,
        )

    def audio_loaded(self, loaded: tuple) -> None:
        path, waveform = loaded
        self.session.set_audio(path, waveform, P.target_sr)
        self.setWindowTitle(f"{path.name} - {BASE_TITLE}")
        self.plot.set_waveform(waveform, P.target_sr)
        self.transport.set_audio(waveform, P.target_sr)
        self.view.band.set_limits(int(P.nyquist_hz))
        # El Raven de la grabación vive junto al wav: revisando se quiere siempre.
        sidecar = path.with_suffix(".txt")
        if sidecar.is_file():
            self.session.set_table(*read_table(sidecar))
        self.say(f"{path.name} · {self.session.duration:.1f} s")
        if self.pending is not None:
            self.show_finding(self.pending)
            self.pending = None

    def table_loaded(self, loaded: tuple[str, pd.DataFrame]) -> None:
        source, table = loaded
        self.session.set_table(source, table)
        if source != FINDINGS:
            self.say(f"{len(table)} anotaciones cargadas.")
            return
        self.reviewing = -1
        self.queue.setText(f"0 / {len(table)}")
        self.finding.setText(f"{table[RECORDING].nunique()} grabaciones · '.' para el primero")
        self.say(f"{len(table)} hallazgos por revisar.")

    def detections_ready(self, table: pd.DataFrame) -> None:
        # Si la corrida pasó por `compare_models.py`, el slider arranca en el umbral que eligió
        # en val (su `operating_point.json`); si no, se queda donde estaba.
        operating = table.attrs.get("operating_point")
        if operating is not None:
            self.score.set_value(operating)
        self.session.set_table(DETECTIONS, table)
        self.say(
            f"{len(table)} detecciones del modelo."
            + ("" if operating is None else f"   Umbral de la comparación: {operating:.2f}")
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
            self.say("No hay más detecciones en esa dirección.")
            return
        target = float(candidates[0] if direction > 0 else candidates[-1])
        self.transport.center(target)

    # La cola recorre los hallazgos por Calidad, peores primero, y abre el audio de cada uno.
    def walk(self, direction: int) -> None:
        table = self.session.tables[FINDINGS]
        if table is None or table.empty:
            return
        self.reviewing = (self.reviewing + direction) % len(table)
        row = table.iloc[self.reviewing]
        path = Path(row[RECORDING])
        if path == self.session.audio_path:
            self.show_finding(row)
        elif path.is_file():
            self.pending = row
            self.load("audio", path)
        else:
            self.say(f"No encontré {path.name}, el audio del hallazgo.")

    def show_finding(self, row: pd.Series) -> None:
        table = self.session.tables[FINDINGS]
        # El encuadre es del usuario: sólo se desplaza si la caja quedó fuera de pantalla,
        # igual que al elegirla en la tabla. La banda y el ancho de ventana no se tocan.
        start, stop = self.transport.time_window()
        if not (start <= row[BEGIN] and row[END] <= stop):
            self.transport.center(0.5 * (row[BEGIN] + row[END]))
        self.plot.set_highlight((row[BEGIN], row[END], row[LOW], row[HIGH]))
        self.queue.setText(f"{self.reviewing + 1} / {0 if table is None else len(table)}")
        self.finding.setText(
            f"{issue(row[ISSUE])} · calidad {row[QUALITY]:.3f} · {row[VERDICT] or 'sin veredicto'}"
        )
        # Dice dónde cae la caja en vez de ir hasta ella: si esta fuera de la banda, se ve
        # en qué frecuencias hay que mirar.
        self.say(
            f"{Path(row[RECORDING]).name} · {label(row)} · "
            f"{row[BEGIN]:.2f}–{row[END]:.2f} s · {row[LOW]:,.0f}–{row[HIGH]:,.0f} Hz"
        )
        self.sync_controls()

    def decide(self, verdict: str) -> None:
        table = self.session.tables[FINDINGS]
        if table is None or not 0 <= self.reviewing < len(table):
            return
        done = self.session.judge(self.reviewing, verdict)
        self.show_finding(table.iloc[self.reviewing])
        self.say(f"{done}.   ('.' para el siguiente)")

    def focus(self, row) -> None:
        if row is None:
            self.plot.set_highlight(None)
            return
        start, stop = self.transport.time_window()
        if not (start <= row.begin and row.end <= stop):
            self.transport.center(0.5 * (row.begin + row.end))
        self.plot.set_highlight((row.begin, row.end, row.low, row.high))

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
        # Una capa apagada entra como None: ni se dibuja ni se cuenta. Si todas sus cajas
        # dicen lo mismo, esa etiqueta va a la leyenda y no encima de cada caja.
        notes, layers = [], []
        for source in SOURCES:
            table = self.session.visible(source)
            note = "" if table is None else common_label(table)
            notes.append(note)
            layers.append(
                Layer(
                    table if self.layers.enabled(source) else None,
                    COLORS[source],
                    STYLES[source],
                    WIDTHS[source],
                    source == ANNOTATIONS,
                    not note,
                )
            )
        shown = self.plot.draw_boxes(layers, start, stop)
        for source, count, note in zip(SOURCES, shown, notes, strict=True):
            table = self.session.tables[source]
            self.layers.set_state(source, count, 0 if table is None else len(table), note)

    # --- Exportacion ------------------------------------------------------------

    def save_path(self, title: str, suffix: str, file_filter: str) -> Path | None:
        audio = self.session.audio_path
        stem = audio.stem if audio is not None else "espectrograma"
        folder = audio.parent if audio is not None else Path.cwd()
        chosen, _ = QFileDialog.getSaveFileName(
            self, title, str(folder / f"{stem}{suffix}"), file_filter
        )
        return Path(chosen) if chosen else None

    def export_image(self) -> None:
        if self.session.waveform is None:
            return
        start, stop = self.transport.time_window()
        path = self.save_path("Guardar imagen", f"_{start:.2f}-{stop:.2f}s.png", "PNG (*.png)")
        if path is None:
            return
        try:
            self.plot.export_png(path)
        except Exception as exc:
            self.fail(f"No se pudo guardar la imagen:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"Imagen guardada en {path.name}")

    def export_table(self, source: str) -> None:
        # Los veredictos son de todo el dataset; las cajas, sólo del audio abierto.
        if source == FINDINGS:
            table = self.session.tables[FINDINGS]
            suffix, file_filter = "_veredictos.csv", "CSV (*.csv)"
        else:
            table = self.session.visible(source)
            suffix = f".{'detections' if source == DETECTIONS else 'annotations'}.txt"
            file_filter = "Raven (*.txt);;CSV (*.csv)"
        if table is None:
            return
        table = table.copy()
        if source != FINDINGS:
            table = renumber(table)
        path = self.save_path(f"Guardar {source.lower()}", suffix, file_filter)
        if path is None:
            return
        try:
            table.to_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", index=False)
        except Exception as exc:
            self.fail(f"No se pudo guardar la tabla:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"{len(table)} filas en {path.name}")

    # --- Eventos ----------------------------------------------------------------

    def dropped(self, event) -> Path | None:
        mime = event.mimeData() if event is not None else None
        urls = mime.urls() if mime is not None and mime.hasUrls() else []
        if not urls or not self.open_actions["audio"].isEnabled():
            return None
        path = Path(urls[0].toLocalFile())
        return path if path.is_file() and path.suffix.lower() in SUFFIXES else None

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
        if a0 is None:
            return
        key = a0.key()
        if key == Qt.Key.Key_Space:
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
            self.plot.pan_band(1 if key == Qt.Key.Key_Up else -1)
        elif key == Qt.Key.Key_F:
            self.plot.set_band(0.0, P.nyquist_hz)
        elif key == Qt.Key.Key_N:
            self.jump(1)
        elif key == Qt.Key.Key_P:
            self.jump(-1)
        elif key in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3):
            self.layers.toggle(SOURCES[key - Qt.Key.Key_1])
        elif key == Qt.Key.Key_Comma:
            self.walk(-1)
        elif key == Qt.Key.Key_Period:
            self.walk(1)
        elif key == Qt.Key.Key_A:
            self.decide(ACCEPTED)
        elif key == Qt.Key.Key_R:
            self.decide(REJECTED)
        else:
            super().keyPressEvent(a0)

    @override
    def closeEvent(self, a0) -> None:
        self.plot.close_renderer()
        super().closeEvent(a0)


def main() -> None:
    if sys.platform == "linux":
        os.environ.setdefault("QT_QPA_PLATFORMTHEME", "xdgdesktopportal")
        os.environ.setdefault("QT_WAYLAND_DECORATION", "adwaita")
    app = QApplication(sys.argv)
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
    sys.exit(app.exec())
