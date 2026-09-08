import math

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt, QTimer, pyqtSignal
from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

TICK_MS = 30
CLOCK_WIDTH = 118


# Reproduccion del audio mono ya cargado en memoria.
class AudioPlayer(QWidget):
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

        self.button = QToolButton()
        self.button.setText("▶")
        self.button.setToolTip("Reproducir / pausar (Espacio)")
        self.button.setFixedWidth(32)
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button.clicked.connect(self.toggle)
        self.clock = QLabel("0.00 / 0.00 s")
        self.clock.setFixedWidth(CLOCK_WIDTH)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.button)
        layout.addWidget(self.clock)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.tick)

    def set_audio(self, pcm: bytes, sr: int) -> None:
        self.stop()
        self.pcm = QByteArray(pcm)
        self.sr = sr
        self.set_origin(0.0)

    def set_origin(self, seconds: float) -> None:
        # Se ignora mientras suena el audio: ahi manda el reloj del sink.
        if self.sink is None:
            self.origin = max(seconds, 0.0)
            self.display_time(self.origin)

    def seek(self, seconds: float) -> None:
        # Si esta sonando, reanuda desde el punto nuevo en vez de obligar a detener.
        resume = self.playing()
        self.stop()
        self.set_origin(seconds)
        if resume:
            self.play()

    # Cambiar el volumen rehace el PCM: se conserva el punto y, si sonaba, sigue sonando.
    def set_pcm(self, pcm: bytes) -> None:
        at, resume = self.elapsed(), self.playing()
        self.stop()
        self.pcm = QByteArray(pcm)
        self.set_origin(at)
        if resume:
            self.play()

    def elapsed(self) -> float:
        if self.sink is None:
            return self.origin
        return self.origin + self.sink.processedUSecs() / 1_000_000

    def display_time(self, seconds: float) -> None:
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
                self.button.setText("‖")
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
        self.button.setText("‖")

    def pause(self) -> None:
        if self.sink is None or self.paused:
            return
        self.paused = True
        self.timer.stop()
        self.sink.suspend()
        self.button.setText("▶")

    def stop(self) -> None:
        self.timer.stop()
        if self.sink is not None:
            self.sink.stop()
            self.sink.deleteLater()
            self.sink = None
        self.buffer.close()
        self.paused = False
        self.button.setText("▶")
        self.stopped.emit()

    def toggle(self) -> None:
        self.pause() if self.playing() else self.play()

    def tick(self) -> None:
        if self.sink is None:
            return
        seconds = self.elapsed()
        if math.isclose(seconds, self.duration(), abs_tol=1e-3) or seconds > self.duration():
            self.stop()
            return
        self.display_time(seconds)
        self.moved.emit(seconds)
