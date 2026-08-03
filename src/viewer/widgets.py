import math
from pathlib import Path
from typing import override

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QWidget,
)

LABEL_WIDTH = 100
TICK_MS = 30
SWATCH_WIDTH = 14


class Worker(QThread):
    ok = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, task, reports: bool = False) -> None:
        """Con `reports=True` la tarea recibe un callback (hechos, total) para el progreso."""
        super().__init__()
        self.task = task
        self.reports = reports

    @override
    def run(self) -> None:
        try:
            self.ok.emit(self.task(self.progress.emit) if self.reports else self.task())
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class FileField(QWidget):
    changed = pyqtSignal()

    def __init__(self, text: str, file_filter: str) -> None:
        super().__init__()
        self.file_filter = file_filter
        name = QLabel(text)
        name.setFixedWidth(LABEL_WIDTH)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("sin seleccionar")
        self.edit.setReadOnly(True)
        self.edit.setAcceptDrops(False)  # deja pasar el drop a la ventana principal
        button = QPushButton("Examinar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(name)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)
        button.clicked.connect(self._browse)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar", self.edit.text(), self.file_filter
        )
        if path:
            self.edit.setText(path)
            self.changed.emit()

    def path(self) -> Path | None:
        text = self.edit.text().strip()
        return Path(text) if text else None

    def set_path(self, path: Path) -> None:
        self.edit.setText(str(path))
        self.changed.emit()

    def suffixes(self) -> set[str]:
        """Extensiones aceptadas por el filtro, en minusculas y con punto."""
        inside = self.file_filter[self.file_filter.find("(") + 1 : self.file_filter.rfind(")")]
        return {pattern.removeprefix("*").lower() for pattern in inside.split()}


class Choice(QWidget):
    changed = pyqtSignal()

    def __init__(self, text: str, values: list, index: int, fmt: str = "{:g}") -> None:
        super().__init__()
        self.values = values
        self.fmt = fmt

        name = QLabel(text)
        name.setFixedWidth(LABEL_WIDTH)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(values) - 1)
        self.slider.setValue(index)
        self.readout = QLabel(fmt.format(values[index]))
        self.readout.setFixedWidth(48)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(name)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.readout)
        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self) -> None:
        self.readout.setText(self.fmt.format(self.value()))
        self.changed.emit()

    def value(self):
        return self.values[self.slider.value()]


class Dropdown(QWidget):
    changed = pyqtSignal()

    def __init__(self, text: str, options: list[str], index: int = 0) -> None:
        super().__init__()
        name = QLabel(text)
        name.setFixedWidth(LABEL_WIDTH)
        self.combo = QComboBox()
        self.combo.addItems(options)
        self.combo.setCurrentIndex(index)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(name)
        layout.addWidget(self.combo, 1)
        self.combo.currentIndexChanged.connect(self.changed.emit)

    def value(self) -> str:
        return self.combo.currentText()


class Legend(QWidget):
    """Muestras de color para identificar el origen de cada caja."""

    def __init__(self, entries: list[tuple[str, str]]) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for text, color in entries:
            swatch = QLabel()
            swatch.setFixedWidth(SWATCH_WIDTH)
            swatch.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
            layout.addWidget(swatch)
            layout.addWidget(QLabel(text))
            layout.addSpacing(6)


class AudioPlayer(QWidget):
    """Transporte de reproduccion sobre el audio mono ya cargado en memoria."""

    moved = pyqtSignal(float)
    stopped = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.pcm = QByteArray()
        self.buffer = QBuffer()
        self.sink: QAudioSink | None = None
        self.sr = 1
        self.origin = 0.0
        self.paused = False

        self.clock = QLabel("0.00 s")
        self.clock.setFixedWidth(64)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for text, tip, slot in (
            ("▶", "Reproducir (Espacio)", self.play),
            ("‖", "Pausa (Espacio)", self.pause),
            ("■", "Detener", self.stop),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.setFixedWidth(36)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(slot)
            layout.addWidget(button)
        layout.addWidget(self.clock)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)

    def set_audio(self, pcm: bytes, sr: int) -> None:
        self.stop()
        self.pcm = QByteArray(pcm)
        self.sr = sr
        self.set_origin(0.0)

    def set_origin(self, seconds: float) -> None:
        """Mueve el punto de arranque; se ignora mientras suena el audio."""
        if self.sink is None:
            self.origin = max(seconds, 0.0)
            self.clock.setText(f"{self.origin:.2f} s")

    def duration(self) -> float:
        return self.pcm.size() / (2 * self.sr)

    def playing(self) -> bool:
        return self.sink is not None and not self.paused

    def play(self) -> None:
        if self.sink is not None:
            if self.paused:
                self.paused = False
                self.sink.resume()
                self.timer.start()
            return
        if self.pcm.isEmpty():
            return
        if self.origin >= self.duration():
            self.origin = 0.0

        fmt = QAudioFormat()
        fmt.setSampleRate(self.sr)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        self.buffer.close()
        self.buffer.setData(self.pcm)
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.buffer.seek(2 * int(self.origin * self.sr))

        self.sink = QAudioSink(QMediaDevices.defaultAudioOutput(), fmt)
        self.sink.start(self.buffer)
        self.timer.start()

    def pause(self) -> None:
        if self.sink is None or self.paused:
            return
        self.paused = True
        self.timer.stop()
        self.sink.suspend()

    def stop(self) -> None:
        self.timer.stop()
        if self.sink is not None:
            self.sink.stop()
            self.sink = None
        self.buffer.close()
        self.paused = False
        self.stopped.emit()

    def toggle(self) -> None:
        self.pause() if self.playing() else self.play()

    def _tick(self) -> None:
        if self.sink is None:
            return
        seconds = self.origin + self.sink.processedUSecs() / 1_000_000

        if math.isclose(seconds, self.duration(), abs_tol=1e-3) or seconds > self.duration():
            self.stop()
            return

        self.clock.setText(f"{seconds:.2f} s")
        self.moved.emit(seconds)
