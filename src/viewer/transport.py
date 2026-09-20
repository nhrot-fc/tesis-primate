from typing import override

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QRegion
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollBar,
    QStyle,
    QStyleOptionSlider,
    QToolButton,
    QWidget,
)

from viewer.player import AudioPlayer
from viewer.spectrogram import Waveform, pcm16

TIME_STEP = 0.05
# La llamada mediana dura 0,2 s: por debajo de 1 s también hace falta ventana.
SPANS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0]
SPAN_WIDTH = 78
SKIP_WIDTH = 28
MARK_ALPHA = 150
MARK_MIN_PX = 2


# Un rótulo apagado delante de un mando: dice qué es sin competir con el valor.
def caption(text: str, tip: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName("hint")
    label.setToolTip(tip)
    return label


def skip_button(parent: QWidget, pixmap: QStyle.StandardPixmap, fallback: str) -> QToolButton:
    button = QToolButton()
    button.setFixedWidth(SKIP_WIDTH)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    style = parent.style()
    if style is None:
        button.setText(fallback)
    else:
        button.setIcon(style.standardIcon(pixmap))
    return button


# El scroll del tiempo con las cajas pintadas encima, como la barra de un editor: se ve dónde
# hay llamadas en toda la grabación y se salta ahí de un clic.
class Timeline(QScrollBar):
    def __init__(self) -> None:
        super().__init__(Qt.Orientation.Horizontal)
        self.marks: list[tuple[float, float, str]] = []  # begin, end, color
        self.duration = 0.0

    def set_marks(self, marks: list[tuple[float, float, str]], duration: float) -> None:
        self.marks = marks
        self.duration = duration
        self.update()

    @override
    def paintEvent(self, a0) -> None:
        super().paintEvent(a0)
        style = self.style()
        if not self.marks or self.duration <= 0 or style is None:
            return
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = style.subControlRect(
            QStyle.ComplexControl.CC_ScrollBar, option, QStyle.SubControl.SC_ScrollBarGroove, self
        )
        handle = style.subControlRect(
            QStyle.ComplexControl.CC_ScrollBar, option, QStyle.SubControl.SC_ScrollBarSlider, self
        )
        # El asa queda limpia: es lo que se está viendo; las marcas dicen qué hay alrededor.
        painter = QPainter(self)
        painter.setClipRegion(QRegion(groove).subtracted(QRegion(handle)))
        painter.setPen(Qt.PenStyle.NoPen)
        scale = groove.width() / self.duration
        for begin, end, color in self.marks:
            tint = QColor(color)
            tint.setAlpha(MARK_ALPHA)
            painter.setBrush(tint)
            x = groove.left() + begin * scale
            painter.drawRect(
                int(x), groove.top(), max(int((end - begin) * scale), MARK_MIN_PX), groove.height()
            )
        painter.end()


# Qué tramo del tiempo se ve, dónde está el cabezal y qué suena. La banda de frecuencia es
# de `controls.Band`; la ventana la pone a su lado.
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

        self.bar = Timeline()
        self.bar.valueChanged.connect(self.on_scroll)

        self.spans = QComboBox()
        for value in SPANS:
            self.spans.addItem(f"{value:g} s", value)
        self.spans.setCurrentIndex(SPANS.index(5.0))
        self.spans.setFixedWidth(SPAN_WIDTH)
        self.spans.setToolTip("How much time fits on screen (Ctrl + wheel, Ctrl+1 / Ctrl+3)")
        self.spans.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.spans.currentIndexChanged.connect(self.rescale)
        # El rótulo va con el combo: "5 s" solo no dice que es el ancho de la ventana.
        self.spans_label = caption("Window", self.spans.toolTip())

        # Al principio y al final, a los lados de Play, como en Audacity.
        self.to_start = skip_button(self, QStyle.StandardPixmap.SP_MediaSkipBackward, "⏮")
        self.to_start.setToolTip("Go to the start (Home)")
        self.to_start.clicked.connect(lambda: self.to_edge(False))
        self.to_end = skip_button(self, QStyle.StandardPixmap.SP_MediaSkipForward, "⏭")
        self.to_end.setToolTip("Go to the end (End)")
        self.to_end.clicked.connect(lambda: self.to_edge(True))

        # Reproducir, la barra de tiempo con todo el sitio que sobra, y el reloj al final.
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        buttons = QHBoxLayout()
        buttons.setSpacing(2)
        buttons.addWidget(self.to_start)
        buttons.addWidget(self.player)
        buttons.addWidget(self.to_end)
        layout.addLayout(buttons)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.player.clock)

    def set_audio(self, waveform: Waveform, sr: int) -> None:
        self.duration = waveform.size / sr
        self.waveform = waveform
        self.player.set_audio(pcm16(waveform, self.gain), sr)
        self.bar.blockSignals(True)
        self.bar.setValue(0)
        self.bar.blockSignals(False)
        self.bar.set_marks([], self.duration)
        self.rescale()

    def set_marks(self, marks: list[tuple[float, float, str]]) -> None:
        self.bar.set_marks(marks, self.duration)

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

    # Encuadra un tramo: la ventana más angosta que lo deja con aire, centrada en él.
    def fit(self, begin: float, end: float, margin: float = 2.5, floor: float = 1.0) -> None:
        wanted = max((end - begin) * margin, floor)
        index = next((i for i, span in enumerate(SPANS) if span >= wanted), len(SPANS) - 1)
        self.spans.setCurrentIndex(index)
        self.center(0.5 * (begin + end))

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
