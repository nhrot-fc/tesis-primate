import os
import sys
from pathlib import Path
from typing import override

import pandas as pd
import pyqtgraph as pg
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollBar,
    QVBoxLayout,
    QWidget,
)

from core.config import P
from viewer.plot import ANNOTATION_COLOR, DETECTION_COLOR, SpectrogramView, read_boxes
from viewer.spectrogram import load_audio, pcm16
from viewer.widgets import AudioPlayer, Choice, FileField, Worker

WINDOW_S = 5.0
TIME_STEP = 0.05
NFFT_VALUES = [256, 512, 1024, 2048, 4096, 8192]
HOP_VALUES = [32, 64, 128, 256, 512, 1024]


class Viewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Visor de espectrogramas")
        self.setMinimumSize(880, 560)
        self.resize(1240, 800)

        self.audio_path: Path | None = None
        self.model: tuple | None = None
        self.duration = 0.0
        self.annotations: pd.DataFrame | None = None
        self.detections: pd.DataFrame | None = None
        self.worker: Worker | None = None

        self.audio_field = FileField("Audio", "Audio (*.wav *.flac *.mp3 *.WAV *.FLAC *.MP3)")
        self.model_field = FileField("Modelo", "Checkpoint (*.pth *.pt)")
        self.annotation_field = FileField("Anotaciones", "Raven (*.txt *.csv)")
        self.audio_field.changed.connect(self.load_audio)
        self.annotation_field.changed.connect(self.load_annotations)

        self.load_button = QPushButton("Cargar modelo")
        self.run_button = QPushButton("Detectar vocalizaciones")
        self.load_button.clicked.connect(self.load_model)
        self.run_button.clicked.connect(self.run_model)

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
        for control in (self.n_fft, self.hop, self.brightness, self.contrast):
            control.changed.connect(self.draw_spectrogram)
        self.score.changed.connect(self.draw_boxes)

        self.status = QLabel("Selecciona un audio.")
        self.readout = QLabel("")
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.run_button)

        header = QVBoxLayout()
        header.setSpacing(6)
        header.addWidget(self.audio_field)
        header.addWidget(self.model_field)
        header.addWidget(self.annotation_field)
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

        transport = QHBoxLayout()
        transport.addWidget(self.timebar, 1)
        transport.addWidget(self.player)

        footer = QHBoxLayout()
        footer.addWidget(self.status, 1)
        footer.addWidget(self.readout)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self.plot, 1)
        layout.addLayout(transport)
        layout.addLayout(controls)
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
        ):
            widget.setEnabled(loaded)
        for field in (self.audio_field, self.model_field, self.annotation_field):
            field.setEnabled(not busy)
        self.load_button.setEnabled(not busy)
        self.run_button.setEnabled(not busy and loaded and self.model is not None)
        self.score.setEnabled(self.detections is not None)

    def fail(self, message: str) -> None:
        self.status.setText("Error.")
        QMessageBox.critical(self, "Error", message)

    def start(self, task, done, message: str) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.status.setText(message)
        self.sync_controls(busy=True)
        self.worker = Worker(task)
        self.worker.ok.connect(done)
        self.worker.error.connect(self.fail)
        self.worker.finished.connect(self.sync_controls)
        self.worker.start()

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

    def load_model(self) -> None:
        path = self.checked_path(self.model_field)
        if path is not None:
            self.start(lambda: self.read_model(path), self.model_loaded, "Cargando modelo...")

    def run_model(self) -> None:
        if self.model is None or self.audio_path is None:
            return
        model, labels, device = self.model
        audio = self.audio_path
        self.start(
            lambda: self.infer(model, audio, labels, device),
            self.detections_ready,
            "Ejecutando el modelo...",
        )

    def read_model(self, path: Path) -> tuple:
        from infer import load_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, labels = load_model(path, device)
        return model, labels, device

    def infer(self, model, audio: Path, labels, device: str) -> pd.DataFrame:
        from pipelines.inference_pipeline import predict

        return predict(model, audio, labels, device, score_threshold=0.05)

    def audio_loaded(self, loaded: tuple) -> None:
        self.audio_path, waveform = loaded
        self.duration = waveform.numel() / P.target_sr
        self.detections = None
        self.plot.set_waveform(waveform, P.target_sr)
        self.player.set_audio(pcm16(waveform), P.target_sr)
        self.timebar.setRange(0, max(int((self.duration - WINDOW_S) / TIME_STEP), 0))
        self.timebar.setSingleStep(int(0.5 / TIME_STEP))
        self.timebar.setPageStep(int(WINDOW_S / TIME_STEP))
        self.timebar.setValue(0)
        self.refresh()

    def annotations_loaded(self, table: pd.DataFrame) -> None:
        self.annotations = table
        self.draw_boxes()
        self.status.setText(f"{len(table)} anotaciones cargadas.")

    def model_loaded(self, loaded: tuple) -> None:
        self.model = loaded
        self.status.setText(f"Modelo cargado: {len(loaded[1])} clases.")

    def detections_ready(self, table: pd.DataFrame) -> None:
        self.detections = table
        self.draw_boxes()
        self.status.setText(f"{len(table)} detecciones del modelo.")

    def time_window(self) -> tuple[float, float]:
        start = self.timebar.value() * TIME_STEP
        return start, min(start + WINDOW_S, self.duration)

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
            WINDOW_S,
            self.n_fft.value(),
            self.hop.value(),
            self.brightness.value(),
            self.contrast.value(),
        )
        self.status.setText(f"{start:.2f} - {stop:.2f} s  de  {self.duration:.2f} s")

    def draw_boxes(self) -> None:
        start, stop = self.time_window()
        tables = [(self.annotations, ANNOTATION_COLOR), (self.detections, DETECTION_COLOR)]
        self.plot.draw_boxes(tables, start, stop, self.score.value())

    def step(self, direction: int) -> None:
        self.timebar.setValue(self.timebar.value() + direction * self.timebar.singleStep())

    def track(self, seconds: float, hz: float) -> None:
        if self.plot.waveform is not None:
            self.readout.setText(f"{seconds:.3f} s   {hz:,.0f} Hz")

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
