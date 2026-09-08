from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QSlider, QSpinBox, QWidget

LABEL_WIDTH = 78
READOUT_WIDTH = 46
SWATCH_WIDTH = 10
HZ_WIDTH = 72
HZ_STEP = 500


class Slider(QWidget):
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
        self.slider.valueChanged.connect(self.on_change)

    def on_change(self) -> None:
        self.readout.setText(self.fmt.format(self.value()))
        self.changed.emit()

    def value(self):
        return self.values[self.slider.value()]

    def set_value(self, value) -> None:
        # Al valor disponible más cercano.
        nearest = min(range(len(self.values)), key=lambda i: abs(self.values[i] - value))
        self.slider.setValue(nearest)


# Leyenda con interruptor: identifica el origen de cada caja y lo oculta.
class Layers(QWidget):
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


# Los dos extremos de la banda visible. Se escriben a mano y el zoom los reescribe.
class Band(QWidget):
    changed = pyqtSignal()

    def __init__(self, text: str) -> None:
        super().__init__()
        self.low = QSpinBox()
        self.high = QSpinBox()
        for box in (self.low, self.high):
            box.setSingleStep(HZ_STEP)
            box.setGroupSeparatorShown(True)
            box.setFixedWidth(HZ_WIDTH)
            box.valueChanged.connect(lambda _: self.changed.emit())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(QLabel(text))
        layout.addWidget(self.low)
        layout.addWidget(QLabel("–"))
        layout.addWidget(self.high)

    # Sólo se reabre entera cuando el tope cambia: si no, se respeta lo que haya puesto.
    def set_limits(self, top: int) -> None:
        if top == self.high.maximum():
            return
        for box in (self.low, self.high):
            box.setRange(0, top)
        self.set_values(0, top)

    def values(self) -> tuple[float, float]:
        return float(self.low.value()), float(self.high.value())

    # Lo que escribe el zoom no debe rebotar como si lo hubiera tecleado alguien.
    def set_values(self, low: float, high: float) -> None:
        for box, value in ((self.low, low), (self.high, high)):
            box.blockSignals(True)
            box.setValue(int(round(value)))
            box.blockSignals(False)
