import math

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFontDatabase
from PyQt6.QtMultimedia import QAudioDevice, QAudioFormat, QAudioSink, QMediaDevices
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QStyle, QToolButton, QWidget

TICK_MS = 30
CLOCK_WIDTH = 118


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
        # None: la salida del sistema, leída en cada play
        self.device: QAudioDevice | None = None

        style = self.style()
        self.icons = (
            (
                style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
                style.standardIcon(QStyle.StandardPixmap.SP_MediaPause),
            )
            if style is not None
            else None
        )

        self.button = QToolButton()
        self.button.setToolTip("Play / pause (Space)")
        self.button.setFixedWidth(34)
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button.clicked.connect(self.toggle)
        self.show_playing(False)
        self.clock = QLabel("0.00 / 0.00 s")
        self.clock.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.clock.setFixedWidth(CLOCK_WIDTH)
        self.clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.button)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.tick)

    def show_playing(self, playing: bool) -> None:
        if self.icons is None:
            self.button.setText("❚❚" if playing else "▶")
        else:
            self.button.setIcon(self.icons[playing])

    def set_audio(self, pcm: bytes, sr: int) -> None:
        self.stop()
        self.pcm = QByteArray(pcm)
        self.sr = sr
        self.set_origin(0.0)

    def set_origin(self, seconds: float) -> None:
        # Mientras suena manda el reloj del sink
        if self.sink is None:
            self.origin = max(seconds, 0.0)
            self.display_time(self.origin)

    def seek(self, seconds: float) -> None:
        resume = self.playing()
        self.stop()
        self.set_origin(seconds)
        if resume:
            self.play()

    def set_pcm(self, pcm: bytes) -> None:
        at, resume = self.elapsed(), self.playing()
        self.stop()
        self.pcm = QByteArray(pcm)
        self.set_origin(at)
        if resume:
            self.play()

    def set_device(self, device: QAudioDevice | None) -> None:
        at, resume = self.elapsed(), self.playing()
        self.stop()
        self.device = device
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
                self.show_playing(True)
            return
        if self.pcm.isEmpty():
            return
        if self.origin >= self.duration():
            self.origin = 0.0

        device = self.device if self.device is not None else QMediaDevices.defaultAudioOutput()
        if device.isNull():
            self.failed.emit("No audio output device.")
            return

        self.buffer.close()
        self.buffer.setData(self.pcm)
        self.buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.buffer.seek(2 * int(self.origin * self.sr))

        # Con padre: lo destruye Qt, no el GC mientras su hilo de audio lee
        self.sink = QAudioSink(device, self.format(), self)
        self.sink.start(self.buffer)
        self.timer.start()
        self.show_playing(True)

    def format(self) -> QAudioFormat:
        fmt = QAudioFormat()
        fmt.setSampleRate(self.sr)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        return fmt

    def pause(self) -> None:
        if self.sink is None or self.paused:
            return
        self.paused = True
        self.timer.stop()
        self.sink.suspend()
        self.show_playing(False)

    def stop(self) -> None:
        self.timer.stop()
        if self.sink is not None:
            self.sink.stop()
            self.sink.deleteLater()
            self.sink = None
        self.buffer.close()
        self.paused = False
        self.show_playing(False)
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
