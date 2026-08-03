import os
import sys
from pathlib import Path
from typing import override

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt
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
    QVBoxLayout,
    QWidget,
)

from core.config import P
from viewer.inference import detect
from viewer.plot import (
    ANNOTATION_COLOR,
    COLORMAPS,
    DETECTION_COLOR,
    SpectrogramView,
    read_boxes,
)
from viewer.spectrogram import load_audio, pcm16
from viewer.widgets import AudioPlayer, Choice, Dropdown, FileField, Layers, Worker

BASE_TITLE = "Visor de espectrogramas"
ANNOTATIONS = "Anotaciones"
DETECTIONS = "Modelo"
TIME_STEP = 0.05
WINDOW_VALUES = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
NFFT_VALUES = [256, 512, 1024, 2048, 4096, 8192]
HOP_VALUES = [32, 64, 128, 256, 512, 1024]
BATCH_VALUES = [1, 2, 4, 8, 16, 32, 64]


class Viewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(BASE_TITLE)
        self.setMinimumSize(880, 560)
        self.resize(1240, 800)
        self.setAcceptDrops(True)

        self.audio_path: Path | None = None
        self.duration = 0.0
        self.annotations: pd.DataFrame | None = None
        self.detections: pd.DataFrame | None = None
        self.worker: Worker | None = None

        self.audio_field = FileField("Audio", "Audio (*.wav *.flac *.mp3 *.WAV *.FLAC *.MP3)")
        self.model_field = FileField("Modelo", "Checkpoint (*.pth *.pt)")
        self.annotation_field = FileField("Anotaciones", "Raven (*.txt *.csv)")
        self.audio_field.changed.connect(self.load_audio)
        self.annotation_field.changed.connect(self.load_annotations)
        self.model_field.changed.connect(self.sync_controls)

        self.run_button = QPushButton("Detectar vocalizaciones")
        self.image_button = QPushButton("Guardar imagen")
        self.table_button = QPushButton("Guardar detecciones")
        self.run_button.clicked.connect(self.run_model)
        self.image_button.clicked.connect(self.export_image)
        self.table_button.clicked.connect(self.export_detections)

        self.plot = SpectrogramView()
        self.plot.scrolled.connect(self.step)
        self.plot.moved.connect(self.track)

        self.timebar = QScrollBar(Qt.Orientation.Horizontal)
        self.timebar.valueChanged.connect(self.refresh)

        self.player = AudioPlayer()
        self.player.moved.connect(self.follow_playhead)
        self.player.stopped.connect(self.playback_stopped)

        self.n_fft = Choice("n_fft", NFFT_VALUES, NFFT_VALUES.index(2048))
        self.hop = Choice("hop", HOP_VALUES, HOP_VALUES.index(256))
        self.brightness = Choice("Brillo (dB)", list(range(-60, 61, 2)), 30)
        self.contrast = Choice("Contraste", [round(0.2 + 0.05 * i, 2) for i in range(97)], 16)
        self.score = Choice("Score >=", [i / 100 for i in range(101)], 50, "{:.2f}")
        self.span = Choice("Ventana (s)", WINDOW_VALUES, WINDOW_VALUES.index(5.0))
        self.colormap = Dropdown("Colormap", COLORMAPS)
        # Lotes pequenos: la barra avanza de forma fluida. Lotes grandes: la GPU
        # rinde mas pero el progreso llega a saltos.
        self.batch = Choice("Lote", BATCH_VALUES, BATCH_VALUES.index(2))
        self.batch.setToolTip(
            "Ventanas que el modelo procesa a la vez.\n"
            "Valores bajos dan progreso fluido; valores altos aprovechan mejor la GPU."
        )
        for control in (self.n_fft, self.hop, self.brightness, self.contrast):
            control.changed.connect(self.draw_spectrogram)
        self.score.changed.connect(self.draw_boxes)
        self.span.changed.connect(self.rescale)
        self.colormap.changed.connect(self.change_colormap)

        self.status = QLabel("Selecciona un audio.")
        self.counts = QLabel("")
        self.layers = Layers([(ANNOTATIONS, ANNOTATION_COLOR), (DETECTIONS, DETECTION_COLOR)])
        self.layers.changed.connect(self.draw_boxes)
        self.readout = QLabel("")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.progress = QProgressBar()
        self.progress.hide()

        buttons = QHBoxLayout()
        buttons.addWidget(self.image_button)
        buttons.addWidget(self.table_button)
        buttons.addStretch(1)
        buttons.addWidget(self.run_button)

        header = QVBoxLayout()
        header.setSpacing(6)
        header.addWidget(self.audio_field)
        header.addWidget(self.annotation_field)
        header.addWidget(self.model_field)
        header.addLayout(buttons)

        controls = QGridLayout()
        controls.setHorizontalSpacing(28)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        controls.addWidget(self.n_fft, 0, 0)
        controls.addWidget(self.brightness, 0, 1)
        controls.addWidget(self.hop, 1, 0)
        controls.addWidget(self.contrast, 1, 1)
        controls.addWidget(self.score, 2, 0)
        controls.addWidget(self.span, 2, 1)
        controls.addWidget(self.colormap, 3, 0)
        controls.addWidget(self.batch, 3, 1)

        transport = QHBoxLayout()
        transport.addWidget(self.timebar, 1)
        transport.addWidget(self.player)

        footer = QHBoxLayout()
        footer.addWidget(self.layers)
        footer.addWidget(self.status, 1)
        footer.addWidget(self.counts)
        footer.addWidget(self.readout)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self.plot, 1)
        layout.addLayout(transport)
        layout.addLayout(controls)
        layout.addWidget(self.progress)
        layout.addLayout(footer)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.sync_controls()

    def sync_controls(self, busy: bool = False) -> None:
        loaded = self.plot.waveform is not None
        for widget in (
            self.timebar,
            self.player,
            self.n_fft,
            self.hop,
            self.brightness,
            self.contrast,
            self.span,
            self.colormap,
        ):
            widget.setEnabled(loaded)
        for field in (self.audio_field, self.annotation_field, self.model_field):
            field.setEnabled(not busy)
        self.batch.setEnabled(not busy)
        self.run_button.setEnabled(not busy and loaded and self.model_field.path() is not None)
        self.image_button.setEnabled(not busy and loaded)
        self.table_button.setEnabled(not busy and self.detections is not None)
        self.score.setEnabled(self.detections is not None)

    def fail(self, message: str) -> None:
        self.status.setText("Error.")
        QMessageBox.critical(self, "Error", message)

    def start(self, task, done, message: str, reports: bool = False, on_error=None) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.status.setText(message)
        self.sync_controls(busy=True)
        self.progress.setRange(0, 0)  # indeterminado hasta el primer reporte
        self.progress.show()
        self.worker = Worker(task, reports)
        self.worker.ok.connect(done)
        self.worker.error.connect(on_error or self.fail)
        self.worker.progress.connect(self.show_progress)
        self.worker.finished.connect(self.finish)
        self.worker.start()

    def show_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def finish(self) -> None:
        self.progress.hide()
        self.sync_controls()

    def checked_path(self, field: FileField) -> Path | None:
        path = field.path()
        if path is None:
            self.fail("Selecciona un archivo.")
        elif not path.is_file():
            self.fail(f"No existe el archivo:\n{path}")
        else:
            return path
        return None

    def load_audio(self) -> None:
        path = self.checked_path(self.audio_field)
        if path is not None:
            self.start(
                lambda: (path, load_audio(path, P.target_sr)),
                self.audio_loaded,
                "Cargando audio...",
            )

    def load_annotations(self) -> None:
        path = self.checked_path(self.annotation_field)
        if path is not None:
            self.start(lambda: read_boxes(path), self.annotations_loaded, "Cargando anotaciones...")

    def run_model(self) -> None:
        model = self.checked_path(self.model_field)
        if self.audio_path is None or model is None:
            return
        audio, batch = self.audio_path, self.batch.value()
        self.start(
            lambda report: detect(audio, model, batch_size=batch, on_progress=report),
            self.detections_ready,
            f"Ejecutando '{model.name}'...",
            reports=True,
        )

    def audio_loaded(self, loaded: tuple) -> None:
        self.audio_path, waveform = loaded
        self.duration = waveform.size / P.target_sr
        self.detections = None
        self.setWindowTitle(f"{self.audio_path.name} - {BASE_TITLE}")
        self.plot.set_waveform(waveform, P.target_sr)
        self.player.set_audio(pcm16(waveform), P.target_sr)
        self.timebar.blockSignals(True)
        self.timebar.setValue(0)
        self.timebar.blockSignals(False)
        self.rescale()

    def rescale(self) -> None:
        """Reajusta la barra de tiempo al ancho de ventana, conservando el instante actual."""
        start = self.timebar.value() * TIME_STEP
        span = self.span.value()
        self.timebar.blockSignals(True)
        self.timebar.setRange(0, max(int((self.duration - span) / TIME_STEP), 0))
        self.timebar.setSingleStep(max(int(0.1 * span / TIME_STEP), 1))
        self.timebar.setPageStep(max(int(span / TIME_STEP), 1))
        self.timebar.setValue(int(start / TIME_STEP))
        self.timebar.blockSignals(False)
        self.refresh()

    def change_colormap(self) -> None:
        self.plot.set_colormap(self.colormap.value())

    def annotations_loaded(self, table: pd.DataFrame) -> None:
        self.annotations = table
        self.draw_boxes()
        self.status.setText(f"{len(table)} anotaciones cargadas.")

    def detections_ready(self, table: pd.DataFrame) -> None:
        self.detections = table
        # El slider arranca en el punto de operación con el que se eligió el checkpoint:
        # es el umbral en el que su recall y su FP/TP fueron medidos.
        operating = table.attrs.get("operating_score_threshold")
        if operating is not None:
            self.score.set_value(operating)
        self.draw_boxes()
        self.status.setText(f"{len(table)} detecciones del modelo.")

    def time_window(self) -> tuple[float, float]:
        start = self.timebar.value() * TIME_STEP
        return start, min(start + self.span.value(), self.duration)

    def refresh(self) -> None:
        self.player.set_origin(self.time_window()[0])
        self.draw_spectrogram()
        self.draw_boxes()

    def follow_playhead(self, seconds: float) -> None:
        start, stop = self.time_window()
        if not start <= seconds < stop:
            self.timebar.setValue(int(seconds / TIME_STEP))
        self.plot.set_playhead(seconds)

    def playback_stopped(self) -> None:
        self.plot.set_playhead(None)
        self.player.set_origin(self.time_window()[0])

    def draw_spectrogram(self) -> None:
        start, stop = self.time_window()
        self.plot.draw(
            start,
            self.span.value(),
            self.n_fft.value(),
            self.hop.value(),
            self.brightness.value(),
            self.contrast.value(),
        )
        self.status.setText(f"{start:.2f} - {stop:.2f} s  de  {self.duration:.2f} s")

    def draw_boxes(self) -> None:
        start, stop = self.time_window()
        layers = [
            (ANNOTATIONS, "anot.", self.annotations, ANNOTATION_COLOR),
            (DETECTIONS, "det.", self.detections, DETECTION_COLOR),
        ]
        # Una capa apagada entra como None: ni se dibuja ni se cuenta.
        tables = [
            (table if self.layers.enabled(name) else None, color)
            for name, _, table, color in layers
        ]
        shown = self.plot.draw_boxes(tables, start, stop, self.score.value())
        parts = [
            f"{short} {count}/{len(table)}"
            for (name, short, table, _), count in zip(layers, shown, strict=True)
            if table is not None and self.layers.enabled(name)
        ]
        self.counts.setText("   ".join(parts))

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
            self.status.setText(f"Imagen guardada en {path.name}")

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
            score = self.score.value()
            self.status.setText(f"{len(table)} detecciones (score >= {score:.2f}) en {path.name}")

    def step(self, direction: int) -> None:
        self.timebar.setValue(self.timebar.value() + direction * self.timebar.singleStep())

    def track(self, seconds: float, hz: float) -> None:
        if self.plot.waveform is not None:
            self.readout.setText(f"{seconds:.3f} s   {hz:,.0f} Hz")

    def dropped_field(self, event) -> tuple[FileField, Path] | None:
        mime = event.mimeData() if event is not None else None
        urls = mime.urls() if mime is not None and mime.hasUrls() else []
        if not urls or not self.audio_field.isEnabled():
            return None
        path = Path(urls[0].toLocalFile())
        if not path.is_file():
            return None
        for field in (self.audio_field, self.annotation_field, self.model_field):
            if path.suffix.lower() in field.suffixes():
                return field, path
        return None

    @override
    def dragEnterEvent(self, a0) -> None:
        if a0 is not None and self.dropped_field(a0) is not None:
            a0.acceptProposedAction()

    @override
    def dropEvent(self, a0) -> None:
        target = self.dropped_field(a0) if a0 is not None else None
        if a0 is not None and target is not None:
            a0.acceptProposedAction()
            field, path = target
            field.set_path(path)

    @override
    def keyPressEvent(self, a0) -> None:
        steps = {
            Qt.Key.Key_Left: -self.timebar.singleStep(),
            Qt.Key.Key_Right: self.timebar.singleStep(),
            Qt.Key.Key_PageUp: -self.timebar.pageStep(),
            Qt.Key.Key_PageDown: self.timebar.pageStep(),
        }
        if a0 is None:
            return
        if a0.key() == Qt.Key.Key_Space:
            self.player.toggle()
        elif a0.key() in steps:
            self.timebar.setValue(self.timebar.value() + steps[Qt.Key(a0.key())])
        elif a0.key() == Qt.Key.Key_Home:
            self.timebar.setValue(self.timebar.minimum())
        elif a0.key() == Qt.Key.Key_End:
            self.timebar.setValue(self.timebar.maximum())
        else:
            super().keyPressEvent(a0)


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
