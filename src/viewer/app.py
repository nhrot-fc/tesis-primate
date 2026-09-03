import os
import sys
from pathlib import Path
from typing import override

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollBar,
    QSizePolicy,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.config import P
from viewer.inference import detect, preload
from viewer.plot import (
    ANNOTATION_COLOR,
    COLORMAPS,
    DETECTION_COLOR,
    RESOLUTIONS,
    SpectrogramView,
    read_boxes,
)
from viewer.spectrogram import load_audio, pcm16
from viewer.widgets import AudioPlayer, Choice, Dropdown, Layers, Worker

BASE_TITLE = "Visor de espectrogramas"
ANNOTATIONS = "Anotaciones"
DETECTIONS = "Modelo"
TIME_STEP = 0.05
WINDOW_VALUES = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
BATCH_SIZE = 8

AUDIO_FILTER = "Audio (*.wav *.flac *.mp3 *.WAV *.FLAC *.MP3)"
TABLE_FILTER = "Raven (*.txt *.csv)"
MODEL_FILTER = "Checkpoint (*.pth *.pt)"


class Viewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(BASE_TITLE)
        self.setMinimumSize(880, 560)
        self.resize(1240, 820)
        self.setAcceptDrops(True)

        self.audio_path: Path | None = None
        self.model_path: Path | None = None
        self.duration = 0.0
        self.annotations: pd.DataFrame | None = None
        self.detections: pd.DataFrame | None = None
        self.worker: Worker | None = None

        self.plot = SpectrogramView()
        self.plot.scrolled.connect(self.step)
        self.plot.moved.connect(self.track)
        self.plot.clicked.connect(self.seek)

        # --- Acciones: una sola barra reemplaza las tres filas de selectores de archivo ---
        self.actions_ = {
            "audio": QAction("Abrir audio", self),
            "table": QAction("Anotaciones", self),
            "model": QAction("Modelo", self),
            "run": QAction("Detectar", self),
            "image": QAction("Guardar imagen", self),
            "export": QAction("Guardar detecciones", self),
        }
        for key, shortcut, slot in (
            ("audio", "Ctrl+O", self.open_audio),
            ("table", "Ctrl+T", self.open_annotations),
            ("model", "Ctrl+M", self.open_model),
            ("run", "Ctrl+R", self.run_model),
            ("image", "Ctrl+S", self.export_image),
            ("export", "Ctrl+E", self.export_detections),
        ):
            self.actions_[key].setShortcut(QKeySequence(shortcut))
            self.actions_[key].triggered.connect(lambda _, run=slot: run())

        self.visuals_action = QAction("Visualización", self)
        self.visuals_action.setCheckable(True)
        self.visuals_action.setShortcut(QKeySequence("V"))

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        toolbar.addAction(self.actions_["audio"])
        toolbar.addAction(self.actions_["table"])
        toolbar.addAction(self.actions_["model"])
        toolbar.addSeparator()
        toolbar.addAction(self.actions_["run"])
        toolbar.addSeparator()
        toolbar.addAction(self.actions_["image"])
        toolbar.addAction(self.actions_["export"])
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addAction(self.visuals_action)
        self.addToolBar(toolbar)

        # --- Transporte: reproduccion, recorrido y ancho de ventana en una sola fila ---
        self.timebar = QScrollBar(Qt.Orientation.Horizontal)
        self.timebar.valueChanged.connect(self.refresh)
        self.player = AudioPlayer()
        self.player.moved.connect(self.follow_playhead)
        self.player.failed.connect(self.say)
        self.player.stopped.connect(self.playback_stopped)
        self.span_box = Dropdown(
            "Ventana",
            [(f"{value:g} s", value) for value in WINDOW_VALUES],
            WINDOW_VALUES.index(5.0),
        )
        self.span_box.setMaximumWidth(160)
        self.span_box.changed.connect(self.rescale)
        self.plot.zoomed.connect(self.span_box.step)  # Ctrl+rueda cambia el ancho

        transport = QHBoxLayout()
        transport.setSpacing(10)
        transport.addWidget(self.player)
        transport.addWidget(self.timebar, 1)
        transport.addWidget(self.span_box)

        # --- Filtro: lo unico que se toca sin parar durante una revision ---
        self.score = Choice("Score ≥", [i / 100 for i in range(101)], 50, "{:.2f}")
        self.score.changed.connect(self.draw_boxes)
        self.score.setMaximumWidth(320)
        self.layers = Layers([(ANNOTATIONS, ANNOTATION_COLOR), (DETECTIONS, DETECTION_COLOR)])
        self.layers.changed.connect(self.draw_boxes)
        self.prev_button = QPushButton("◀ anterior")
        self.next_button = QPushButton("siguiente ▶")
        self.prev_button.setToolTip("Detección anterior (P)")
        self.next_button.setToolTip("Detección siguiente (N)")
        for button, direction in ((self.prev_button, -1), (self.next_button, 1)):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _, d=direction: self.jump(d))

        filters = QHBoxLayout()
        filters.setSpacing(16)
        filters.addWidget(self.score)
        filters.addWidget(self.layers)
        filters.addStretch(1)
        filters.addWidget(self.prev_button)
        filters.addWidget(self.next_button)

        # --- Visualizacion: se ajusta una vez y estorba, asi que va plegada ---
        self.resolution = Dropdown(
            "Resolución", [(f"{n_fft} / {hop}", (n_fft, hop)) for n_fft, hop in RESOLUTIONS], 2
        )
        self.brightness = Choice("Brillo (dB)", list(range(-60, 61, 2)), 30)
        self.contrast = Choice("Contraste", [round(0.2 + 0.05 * i, 2) for i in range(97)], 16)
        self.colormap = Dropdown("Colormap", [(name, name) for name in COLORMAPS])
        for control in (self.resolution, self.brightness, self.contrast):
            control.changed.connect(self.draw_spectrogram)
        self.colormap.changed.connect(self.change_colormap)

        grid = QGridLayout()
        grid.setContentsMargins(0, 4, 0, 0)
        grid.setHorizontalSpacing(28)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.addWidget(self.resolution, 0, 0)
        grid.addWidget(self.brightness, 0, 1)
        grid.addWidget(self.colormap, 1, 0)
        grid.addWidget(self.contrast, 1, 1)
        self.visuals = QWidget()
        self.visuals.setLayout(grid)
        self.visuals.setVisible(False)
        self.visuals_action.toggled.connect(self.visuals.setVisible)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)
        layout.addWidget(self.plot, 1)
        layout.addLayout(transport)
        layout.addLayout(filters)
        layout.addWidget(self.visuals)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.readout = QLabel("")
        self.readout.setMinimumWidth(150)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.bar = QStatusBar()
        self.bar.addPermanentWidget(self.progress)
        self.bar.addPermanentWidget(self.readout)
        self.setStatusBar(self.bar)
        self.engine = False
        self.sync_controls()
        self.say("Preparando el motor de detección...")
        self.preloader = Worker(preload)
        self.preloader.ok.connect(self.engine_ready)
        self.preloader.error.connect(self.engine_failed)
        self.preloader.start()

    def engine_ready(self, _) -> None:
        self.engine = True
        self.sync_controls()
        self.say("Abre un audio (Ctrl+O) o arrástralo a la ventana.")

    def engine_failed(self, message: str) -> None:
        # Sin motor el visor sigue sirviendo para mirar y escuchar tablas ya hechas.
        self.engine = True
        self.sync_controls()
        self.say(f"El motor de detección no cargó ({message}). El visor funciona igual.")

    # --- Estado -----------------------------------------------------------------

    def span(self) -> float:
        return float(self.span_box.value())

    def say(self, message: str) -> None:
        self.bar.showMessage(message)

    def sync_controls(self, busy: bool = False) -> None:
        loaded = self.plot.waveform is not None
        has_detections = self.detections is not None
        for widget in (
            self.timebar,
            self.player,
            self.span_box,
            self.visuals,
        ):
            widget.setEnabled(loaded)
        for key in ("audio", "table", "model"):
            self.actions_[key].setEnabled(not busy)
        self.actions_["run"].setEnabled(not busy and loaded and self.model_path is not None)
        self.actions_["image"].setEnabled(not busy and loaded)
        self.actions_["export"].setEnabled(not busy and has_detections)
        self.score.setEnabled(has_detections)
        self.prev_button.setEnabled(has_detections)
        self.next_button.setEnabled(has_detections)

    def fail(self, message: str) -> None:
        self.say("Error.")
        QMessageBox.critical(self, "Error", message)

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

    def ask(self, title: str, file_filter: str) -> Path | None:
        folder = str(self.audio_path.parent) if self.audio_path is not None else ""
        chosen, _ = QFileDialog.getOpenFileName(self, title, folder, file_filter)
        return Path(chosen) if chosen else None

    def open_audio(self, path: Path | None = None) -> None:
        path = path or self.ask("Abrir audio", AUDIO_FILTER)
        if path is not None:
            self.start(
                lambda: (path, load_audio(path, P.target_sr)),
                self.audio_loaded,
                "Cargando audio...",
            )

    def open_annotations(self, path: Path | None = None) -> None:
        path = path or self.ask("Abrir anotaciones", TABLE_FILTER)
        if path is not None:
            self.start(lambda: read_boxes(path), self.annotations_loaded, "Cargando anotaciones...")

    def open_model(self, path: Path | None = None) -> None:
        path = path or self.ask("Abrir modelo", MODEL_FILTER)
        if path is not None:
            self.model_path = path
            self.actions_["model"].setToolTip(str(path))
            self.say(f"Modelo listo: {path.name}   (Ctrl+R para detectar)")
            self.sync_controls()

    def run_model(self) -> None:
        if self.audio_path is None or self.model_path is None:
            return
        audio, model = self.audio_path, self.model_path
        self.start(
            lambda report: detect(audio, model, batch_size=BATCH_SIZE, on_progress=report),
            self.detections_ready,
            f"Ejecutando '{model.name}'...",
            reports=True,
        )

    def audio_loaded(self, loaded: tuple) -> None:
        self.audio_path, waveform = loaded
        self.duration = waveform.size / P.target_sr
        self.detections = None
        if self.audio_path is None:
            print("Error: audio_loaded recibió None")
            return
        self.setWindowTitle(f"{self.audio_path.name} - {BASE_TITLE}")
        self.actions_["audio"].setToolTip(str(self.audio_path))
        self.plot.set_waveform(waveform, P.target_sr)
        self.player.set_audio(pcm16(waveform), P.target_sr)
        self.timebar.blockSignals(True)
        self.timebar.setValue(0)
        self.timebar.blockSignals(False)
        self.rescale()

    def annotations_loaded(self, table: pd.DataFrame) -> None:
        self.annotations = table
        self.draw_boxes()
        self.sync_controls()
        self.say(f"{len(table)} anotaciones cargadas.")

    def detections_ready(self, table: pd.DataFrame) -> None:
        self.detections = table
        # El slider arranca en el punto de operación con el que se eligió el checkpoint:
        # es el umbral en el que su recall y su FP/TP fueron medidos.
        operating = table.attrs.get("operating_score_threshold")
        if operating is not None:
            self.score.set_value(operating)
        self.draw_boxes()
        self.sync_controls()
        self.say(f"{len(table)} detecciones del modelo.")

    # --- Recorrido --------------------------------------------------------------

    def rescale(self) -> None:
        # Conserva el instante actual al cambiar el ancho de ventana.
        start = self.timebar.value() * TIME_STEP
        span = self.span()
        self.timebar.blockSignals(True)
        self.timebar.setRange(0, max(int((self.duration - span) / TIME_STEP), 0))
        self.timebar.setSingleStep(max(int(0.1 * span / TIME_STEP), 1))
        self.timebar.setPageStep(max(int(span / TIME_STEP), 1))
        self.timebar.setValue(int(start / TIME_STEP))
        self.timebar.blockSignals(False)
        self.refresh()

    def time_window(self) -> tuple[float, float]:
        start = self.timebar.value() * TIME_STEP
        return start, min(start + self.span(), self.duration)

    def refresh(self) -> None:
        self.player.set_origin(self.time_window()[0])
        self.draw_spectrogram()
        self.draw_boxes()

    def step(self, direction: int) -> None:
        self.timebar.setValue(self.timebar.value() + direction * self.timebar.singleStep())

    def seek(self, seconds: float) -> None:
        self.player.set_origin(seconds)
        self.plot.set_playhead(seconds)

    def jump(self, direction: int) -> None:
        # Centra la ventana en la primera detección que no esté en pantalla. La referencia
        # es el borde y no el centro: al principio y al final del audio la barra no puede
        # centrar la caja, y con el centro la misma detección volvía a salir elegida.
        table = self.visible_detections()
        if table is None or table.empty:
            return
        start, stop = self.time_window()
        times = table["Begin Time (s)"].to_numpy()
        candidates = times[times >= stop] if direction > 0 else times[times < start]
        if candidates.size == 0:
            self.say("No hay más detecciones en esa dirección.")
            return
        target = float(candidates[0] if direction > 0 else candidates[-1])
        self.timebar.setValue(int(max(target - self.span() / 2, 0.0) / TIME_STEP))

    def follow_playhead(self, seconds: float) -> None:
        start, stop = self.time_window()
        if not start <= seconds < stop:
            self.timebar.setValue(int(seconds / TIME_STEP))
        self.plot.set_playhead(seconds)

    def playback_stopped(self) -> None:
        self.plot.set_playhead(None)
        self.player.set_origin(self.time_window()[0])

    # --- Dibujo -----------------------------------------------------------------

    def change_colormap(self) -> None:
        self.plot.set_colormap(self.colormap.value())

    def draw_spectrogram(self) -> None:
        start, stop = self.time_window()
        n_fft, hop = self.resolution.value()
        self.plot.draw(
            start, self.span(), n_fft, hop, self.brightness.value(), self.contrast.value()
        )
        self.say(f"{start:.2f} - {stop:.2f} s   de   {self.duration:.2f} s")

    def draw_boxes(self) -> None:
        start, stop = self.time_window()
        # El ultimo campo situa la etiqueta arriba o abajo de la caja.
        layers = [
            (ANNOTATIONS, self.annotations, ANNOTATION_COLOR, True),
            (DETECTIONS, self.detections, DETECTION_COLOR, False),
        ]
        # Una capa apagada entra como None: ni se dibuja ni se cuenta.
        tables = [
            (table if self.layers.enabled(name) else None, color, above)
            for name, table, color, above in layers
        ]
        shown = self.plot.draw_boxes(tables, start, stop, self.score.value())
        for (name, table, _, _), count in zip(layers, shown, strict=True):
            self.layers.set_count(name, count, 0 if table is None else len(table))

    def track(self, seconds: float, hz: float) -> None:
        if self.plot.waveform is not None:
            self.readout.setText(f"{seconds:.3f} s    {hz:,.0f} Hz")

    # --- Exportacion ------------------------------------------------------------

    def visible_detections(self) -> pd.DataFrame | None:
        if self.detections is None:
            return None
        table = self.detections[self.detections["Score"] >= self.score.value()].copy()
        table["Selection"] = range(1, len(table) + 1)
        return table

    def save_path(self, title: str, suffix: str, file_filter: str) -> Path | None:
        stem = self.audio_path.stem if self.audio_path is not None else "espectrograma"
        folder = self.audio_path.parent if self.audio_path is not None else Path.cwd()
        chosen, _ = QFileDialog.getSaveFileName(
            self, title, str(folder / f"{stem}{suffix}"), file_filter
        )
        return Path(chosen) if chosen else None

    def export_image(self) -> None:
        if self.plot.waveform is None:
            return
        start, stop = self.time_window()
        path = self.save_path("Guardar imagen", f"_{start:.2f}-{stop:.2f}s.png", "PNG (*.png)")
        if path is None:
            return
        try:
            self.plot.export_png(path)
        except Exception as exc:
            self.fail(f"No se pudo guardar la imagen:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"Imagen guardada en {path.name}")

    def export_detections(self) -> None:
        table = self.visible_detections()
        if table is None:
            return
        path = self.save_path(
            "Guardar detecciones", ".detections.txt", "Raven (*.txt);;CSV (*.csv)"
        )
        if path is None:
            return
        try:
            table.to_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", index=False)
        except Exception as exc:
            self.fail(f"No se pudieron guardar las detecciones:\n{type(exc).__name__}: {exc}")
        else:
            self.say(f"{len(table)} detecciones (score ≥ {self.score.value():.2f}) en {path.name}")

    # --- Eventos ----------------------------------------------------------------

    def dropped(self, event) -> tuple[str, Path] | None:
        mime = event.mimeData() if event is not None else None
        urls = mime.urls() if mime is not None and mime.hasUrls() else []
        if not urls or not self.actions_["audio"].isEnabled():
            return None
        path = Path(urls[0].toLocalFile())
        if not path.is_file():
            return None
        for key, suffixes in (
            ("audio", {".wav", ".flac", ".mp3"}),
            ("table", {".txt", ".csv"}),
            ("model", {".pth", ".pt"}),
        ):
            if path.suffix.lower() in suffixes:
                return key, path
        return None

    @override
    def dragEnterEvent(self, a0) -> None:
        if a0 is not None and self.dropped(a0) is not None:
            a0.acceptProposedAction()

    @override
    def dropEvent(self, a0) -> None:
        target = self.dropped(a0) if a0 is not None else None
        if a0 is not None and target is not None:
            a0.acceptProposedAction()
            key, path = target
            {"audio": self.open_audio, "table": self.open_annotations, "model": self.open_model}[
                key
            ](path)

    @override
    def keyPressEvent(self, a0) -> None:
        if a0 is None:
            return
        steps = {
            Qt.Key.Key_Left: -self.timebar.singleStep(),
            Qt.Key.Key_Right: self.timebar.singleStep(),
            Qt.Key.Key_PageUp: -self.timebar.pageStep(),
            Qt.Key.Key_PageDown: self.timebar.pageStep(),
        }
        key = a0.key()
        if key == Qt.Key.Key_Space:
            self.player.toggle()
        elif key in steps:
            self.timebar.setValue(self.timebar.value() + steps[Qt.Key(key)])
        elif key == Qt.Key.Key_Home:
            self.timebar.setValue(self.timebar.minimum())
        elif key == Qt.Key.Key_End:
            self.timebar.setValue(self.timebar.maximum())
        elif key == Qt.Key.Key_N:
            self.jump(1)
        elif key == Qt.Key.Key_P:
            self.jump(-1)
        elif key == Qt.Key.Key_1:
            self.layers.toggle(ANNOTATIONS)
        elif key == Qt.Key.Key_2:
            self.layers.toggle(DETECTIONS)
        else:
            super().keyPressEvent(a0)

    @override
    def closeEvent(self, a0) -> None:
        self.plot.close_renderer()
        super().closeEvent(a0)


def main() -> None:
    os.environ.setdefault("QT_QPA_PLATFORMTHEME", "xdgdesktopportal")
    os.environ.setdefault("QT_WAYLAND_DECORATION", "adwaita")
    app = QApplication(sys.argv)
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
    sys.exit(app.exec())
