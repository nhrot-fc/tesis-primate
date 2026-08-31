import math
from collections.abc import Callable
from typing import override

from PyQt6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QMutex,
    Qt,
    QThread,
    QTimer,
    QWaitCondition,
    pyqtSignal,
)
from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QWidget,
)

LABEL_WIDTH = 78
READOUT_WIDTH = 52
TICK_MS = 30
SWATCH_WIDTH = 10


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


class Latest(QThread):
    """Hilo de un solo trabajo a la vez: si llega otro mientras calcula, el que
    esperaba se descarta. Es lo que mantiene fluido el scroll del espectrograma."""

    done = pyqtSignal(int, object)

    def __init__(self) -> None:
        super().__init__()
        self.mutex = QMutex()
        self.waiting = QWaitCondition()
        self.job: tuple[int, Callable[[], object]] | None = None
        self.counter = 0
        self.closing = False
        self.start()

    def submit(self, task: Callable[[], object]) -> int:
        self.mutex.lock()
        self.counter += 1
        job_id = self.counter
        self.job = (job_id, task)
        self.waiting.wakeOne()
        self.mutex.unlock()
        return job_id

    def close(self) -> None:
        self.mutex.lock()
        self.closing = True
        self.waiting.wakeOne()
        self.mutex.unlock()
        self.wait()

    @override
    def run(self) -> None:
        while True:
            self.mutex.lock()
            while self.job is None and not self.closing:
                self.waiting.wait(self.mutex)
            if self.closing or self.job is None:
                self.mutex.unlock()
                if self.closing:
                    return
                continue  # despertar espurio: no hay nada que calcular
            job_id, task = self.job
            self.job = None
            self.mutex.unlock()
            try:
                self.done.emit(job_id, task())
            except Exception:
                self.done.emit(job_id, None)  # una banda fallida no merece una alerta


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
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.readout = QLabel(fmt.format(values[index]))
        self.readout.setFixedWidth(READOUT_WIDTH)
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

    def set_value(self, value) -> None:
        """Mueve el slider al valor disponible más cercano."""
        nearest = min(range(len(self.values)), key=lambda i: abs(self.values[i] - value))
        self.slider.setValue(nearest)

    def step(self, delta: int) -> None:
        self.slider.setValue(self.slider.value() + delta)


class Dropdown(QWidget):
    changed = pyqtSignal()

    def __init__(self, text: str, options: list[tuple[str, object]], index: int = 0) -> None:
        super().__init__()
        self.combo = QComboBox()
        for label, data in options:
            self.combo.addItem(label, data)
        self.combo.setCurrentIndex(index)
        self.combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if text:
            name = QLabel(text)
            name.setFixedWidth(LABEL_WIDTH)
            layout.addWidget(name)
        layout.addWidget(self.combo, 1)
        self.combo.currentIndexChanged.connect(self.changed.emit)

    def value(self):
        return self.combo.currentData()

    def step(self, delta: int) -> None:
        self.combo.setCurrentIndex(
            min(max(self.combo.currentIndex() + delta, 0), self.combo.count() - 1)
        )


class Layers(QWidget):
    """Leyenda con interruptor: identifica el origen de cada caja y lo oculta."""

    changed = pyqtSignal()

    def __init__(self, entries: list[tuple[str, str]]) -> None:
        super().__init__()
        self.boxes: dict[str, QCheckBox] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for text, color in entries:
            swatch = QLabel()
            swatch.setFixedWidth(SWATCH_WIDTH)
            swatch.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
            box = QCheckBox(text)
            box.setChecked(True)
            box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            # Lambda: `toggled` emite el estado y `changed` no lleva argumentos.
            box.toggled.connect(lambda _: self.changed.emit())
            self.boxes[text] = box
            layout.addWidget(swatch)
            layout.addWidget(box)
            layout.addSpacing(8)

    def enabled(self, text: str) -> bool:
        return self.boxes[text].isChecked()

    def toggle(self, text: str) -> None:
        self.boxes[text].setChecked(not self.boxes[text].isChecked())

    def set_count(self, text: str, shown: int, total: int) -> None:
        self.boxes[text].setText(f"{text}  {shown}/{total}" if total else text)


class AudioPlayer(QWidget):
    """Transporte de reproduccion sobre el audio mono ya cargado en memoria."""

    moved = pyqtSignal(float)
    stopped = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.pcm = QByteArray()
        self.buffer = QBuffer()
        self.sink: QAudioSink | None = None
        self.sr = 1
        self.origin = 0.0
        self.paused = False

        self.clock = QLabel("0.00 s")
        self.clock.setFixedWidth(120)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.play_button = QToolButton()
        for button, text, tip, slot in (
            (self.play_button, "▶", "Reproducir / pausar (Espacio)", self.toggle),
            (QToolButton(), "■", "Detener", self.stop),
        ):
            button.setText(text)
            button.setToolTip(tip)
            button.setFixedWidth(30)
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
            self._show(self.origin)

    def _show(self, seconds: float) -> None:
        self.clock.setText(f"{seconds:.2f} / {self.duration():.2f} s")

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
                self.play_button.setText("‖")
            return
        if self.pcm.isEmpty():
            return
        if self.origin >= self.duration():
            self.origin = 0.0

        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            self.failed.emit("No hay dispositivo de salida de audio.")
            return

        fmt = QAudioFormat()
        fmt.setSampleRate(self.sr)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)

        self.buffer.close()
        self.buffer.setData(self.pcm)
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.buffer.seek(2 * int(self.origin * self.sr))

        # Con padre, el sink lo destruye Qt cuando toca y no al soltar la referencia
        # de Python, que podia caer mientras su hilo de audio seguia leyendo.
        self.sink = QAudioSink(device, fmt, self)
        self.sink.start(self.buffer)
        self.timer.start()
        self.play_button.setText("‖")

    def pause(self) -> None:
        if self.sink is None or self.paused:
            return
        self.paused = True
        self.timer.stop()
        self.sink.suspend()
        self.play_button.setText("▶")

    def stop(self) -> None:
        self.timer.stop()
        if self.sink is not None:
            self.sink.stop()
            self.sink.deleteLater()
            self.sink = None
        self.buffer.close()
        self.paused = False
        self.play_button.setText("▶")
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

        self._show(seconds)
        self.moved.emit(seconds)
