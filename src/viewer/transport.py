from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QScrollBar, QWidget

from viewer.player import AudioPlayer
from viewer.spectrogram import Waveform, pcm16

TIME_STEP = 0.05
# La mediana de los hallazgos dura 0,2 s: por debajo de 1 s también hace falta ventana.
SPANS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
SPAN_WIDTH = 78


# Qué tramo del tiempo se ve, dónde está el cabezal y qué suena. La banda de frecuencia
# se ajusta una vez y vive en el panel de vista, no acá.
class Transport(QWidget):
    changed = pyqtSignal()
    playhead = pyqtSignal(object)  # float mientras hay cabezal, None cuando se apaga
    failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.duration = 0.0
        self.waveform: Waveform | None = None
        self.gain = 1.0

        self.player = AudioPlayer()
        self.player.moved.connect(self.follow)
        self.player.stopped.connect(self.on_stopped)
        self.player.failed.connect(self.failed)

        self.bar = QScrollBar(Qt.Orientation.Horizontal)
        self.bar.valueChanged.connect(self.on_scroll)

        self.spans = QComboBox()
        for value in SPANS:
            self.spans.addItem(f"{value:g} s", value)
        self.spans.setCurrentIndex(SPANS.index(5.0))
        self.spans.setFixedWidth(SPAN_WIDTH)
        self.spans.setToolTip("Ancho de la ventana (Ctrl + rueda)")
        self.spans.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.spans.currentIndexChanged.connect(self.rescale)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self.player)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.spans)

    def set_audio(self, waveform: Waveform, sr: int) -> None:
        self.duration = waveform.size / sr
        self.waveform = waveform
        self.player.set_audio(pcm16(waveform, self.gain), sr)
        self.bar.blockSignals(True)
        self.bar.setValue(0)
        self.bar.blockSignals(False)
        self.rescale()

    def span(self) -> float:
        return float(self.spans.currentData())

    def time_window(self) -> tuple[float, float]:
        start = self.bar.value() * TIME_STEP
        return start, min(start + self.span(), self.duration)

    def rescale(self) -> None:
        # Conserva el instante actual al cambiar el ancho de ventana.
        start = self.bar.value() * TIME_STEP
        span = self.span()
        self.bar.blockSignals(True)
        self.bar.setRange(0, max(int((self.duration - span) / TIME_STEP), 0))
        self.bar.setSingleStep(max(int(0.1 * span / TIME_STEP), 1))
        self.bar.setPageStep(max(int(span / TIME_STEP), 1))
        self.bar.setValue(int(start / TIME_STEP))
        self.bar.blockSignals(False)
        self.on_scroll()

    def on_scroll(self) -> None:
        self.player.set_origin(self.time_window()[0])
        self.changed.emit()

    def step(self, direction: int) -> None:
        self.bar.setValue(self.bar.value() + direction * self.bar.singleStep())

    def page(self, direction: int) -> None:
        self.bar.setValue(self.bar.value() + direction * self.bar.pageStep())

    def to_edge(self, end: bool) -> None:
        self.bar.setValue(self.bar.maximum() if end else self.bar.minimum())

    def zoom(self, delta: int) -> None:
        index = self.spans.currentIndex() + delta
        self.spans.setCurrentIndex(min(max(index, 0), self.spans.count() - 1))

    def set_gain(self, db: float) -> None:
        self.gain = 10.0 ** (db / 20.0)
        if self.waveform is not None:
            self.player.set_pcm(pcm16(self.waveform, self.gain))

    def center(self, seconds: float) -> None:
        self.bar.setValue(int(max(seconds - self.span() / 2, 0.0) / TIME_STEP))

    def seek(self, seconds: float) -> None:
        self.player.seek(seconds)
        self.playhead.emit(seconds)

    def toggle_play(self) -> None:
        self.player.toggle()

    def follow(self, seconds: float) -> None:
        start, stop = self.time_window()
        if not start <= seconds < stop:
            self.bar.setValue(int(seconds / TIME_STEP))
        self.playhead.emit(seconds)

    def on_stopped(self) -> None:
        self.playhead.emit(None)
        self.player.set_origin(self.time_window()[0])
