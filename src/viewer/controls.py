from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from inference.catalog import available_models

LABEL_WIDTH = 92
READOUT_WIDTH = 48
SLIDER_WIDTH = 168
SWATCH_W, SWATCH_H = 20, 12
HZ_WIDTH = 76
HZ_STEP = 500

# Los rangos de los mandos de imagen: viven acá porque el panel es su único dueño.
BRIGHTNESS = (list(range(-60, 61, 2)), 30)
# 1,6 y no 1: en grises el ruido de fondo se queda en gris medio con el rango entero,
# y el contraste lo baja a negro sin llegar a comerse las llamadas debiles.
CONTRAST = ([round(0.2 + 0.05 * i, 2) for i in range(97)], 28)
VOLUME = (list(range(-20, 31, 2)), 10)


class Slider(QWidget):
    changed = pyqtSignal()

    def __init__(
        self, text: str, values: list, index: int, fmt: str = "{:g}", label_width: int = LABEL_WIDTH
    ) -> None:
        super().__init__()
        self.values = values
        self.fmt = fmt

        name = QLabel(text)
        name.setFixedWidth(label_width)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(values) - 1)
        self.slider.setValue(index)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.readout = QLabel(fmt.format(values[index]))
        self.readout.setObjectName("readout")
        self.readout.setFixedWidth(READOUT_WIDTH)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
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


# La muestra de la leyenda se pinta con el mismo trazo que la caja: el color y el guion
# identifican la capa, así que quien no distingue los colores sigue teniendo la linea.
def swatch(color: str, style: Qt.PenStyle, width: int) -> QPixmap:
    pixmap = QPixmap(SWATCH_W, SWATCH_H)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    pen = QPen(QColor(color), width)
    pen.setStyle(style)
    painter.setPen(pen)
    painter.drawLine(1, SWATCH_H // 2, SWATCH_W - 1, SWATCH_H // 2)
    painter.end()
    return pixmap


# Leyenda con interruptor: identifica el origen de cada caja y lo oculta. Una capa sin
# tabla no aparece: sin anotaciones ni modelo la fila queda vacía.
class Layers(QWidget):
    changed = pyqtSignal()

    def __init__(self, entries: list[tuple[str, str, Qt.PenStyle, int]]) -> None:
        super().__init__()
        self.boxes: dict[str, QCheckBox] = {}
        self.chips: dict[str, QWidget] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        for text, color, style, width in entries:
            mark = QLabel()
            mark.setPixmap(swatch(color, style, width))
            box = QCheckBox(text)
            box.setChecked(True)
            box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            # Lambda: `toggled` emite el estado y `changed` no lleva argumentos.
            box.toggled.connect(lambda _: self.changed.emit())

            chip = QWidget()
            inner = QHBoxLayout(chip)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(7)
            inner.addWidget(mark)
            inner.addWidget(box)
            chip.setVisible(False)

            self.boxes[text] = box
            self.chips[text] = chip
            layout.addWidget(chip)

    def enabled(self, text: str) -> bool:
        return self.boxes[text].isChecked()

    def toggle(self, text: str) -> None:
        self.boxes[text].setChecked(not self.boxes[text].isChecked())

    # `shown` son las cajas en pantalla; `total`, las de la tabla (filtrada por score).
    def set_state(self, text: str, shown: int, total: int) -> None:
        self.chips[text].setVisible(total > 0)
        self.boxes[text].setText(f"{text}   {shown}/{total}" if total else text)


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

        name = QLabel(text)
        name.setFixedWidth(LABEL_WIDTH)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(name)
        layout.addWidget(self.low)
        layout.addWidget(QLabel("–"))
        layout.addWidget(self.high)
        layout.addStretch(1)

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


# Cómo se ve y cómo se oye: se ajusta una vez por grabación, así que vive en un desplegable
# en lugar de ocupar dos filas fijas debajo del espectrograma.
class ViewPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.brightness = Slider("Brightness (dB)", *BRIGHTNESS)
        self.contrast = Slider("Contrast", *CONTRAST)
        # 0 dB es la grabación llevada a su pico; por encima recorta, que es lo que
        # hace audible una llamada lejana.
        self.volume = Slider("Volume (dB)", *VOLUME, fmt="{:+g}")
        self.volume.setToolTip("0 dB is the recording normalized to its peak; above that it clips")
        self.band = Band("Band (Hz)")
        self.band.setToolTip("Shift + wheel zooms in frequency; F shows the full band")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        for control in (self.brightness, self.contrast, self.volume):
            control.setMinimumWidth(LABEL_WIDTH + SLIDER_WIDTH + READOUT_WIDTH)
            layout.addWidget(control)
        layout.addWidget(self.band)


# Un botón que abre un panel en vez de un menú de acciones.
class Popup(QToolButton):
    def __init__(self, text: str, panel: QWidget, tip: str = "") -> None:
        super().__init__()
        self.setText(text)
        self.setToolTip(tip)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        menu = QMenu(self)
        action = QWidgetAction(menu)
        action.setDefaultWidget(panel)
        menu.addAction(action)
        self.setMenu(menu)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)


# El modelo es uno para todo el programa: lo que hay en `models\` más lo que se busque a mano.
# Las dos entradas de acción van al final de la lista, y al elegirlas la selección vuelve
# al modelo que había.
class ModelPicker(QComboBox):
    chosen = pyqtSignal(object)  # Path del checkpoint, o None
    browse = pyqtSignal()
    add = pyqtSignal()

    BROWSE, ADD = "Browse for a checkpoint…", "Add model from zip…"

    def __init__(self) -> None:
        super().__init__()
        self.setPlaceholderText("Select a model…")
        self.setToolTip("Model used by Detect and by the Batch view (Ctrl+M browses)")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.current: Path | None = None
        self.reload()
        self.activated.connect(self.on_activated)

    def paths(self) -> list[Path]:
        return [self.itemData(i) for i in range(self.count()) if self.itemData(i) is not None]

    # Relee `models\` conservando lo elegido a mano; `select` deja ese checkpoint elegido.
    def reload(self, select: Path | None = None) -> None:
        listed = available_models()
        extra = [p for p in self.paths() if p not in listed]
        self.blockSignals(True)
        self.clear()
        for path in [*listed, *extra]:
            self.addItem(path.parent.name, path)
        self.insertSeparator(self.count())
        self.addItem(self.BROWSE)
        self.addItem(self.ADD)
        self.blockSignals(False)
        self.select(select or self.current)

    def select(self, path: Path | None) -> None:
        if path is not None and path not in self.paths():
            self.insertItem(0, path.parent.name, path)
        index = -1 if path is None else self.findData(path)
        self.setCurrentIndex(index)
        if path != self.current:
            self.current = path
            self.chosen.emit(path)

    def on_activated(self, index: int) -> None:
        text, path = self.itemText(index), self.itemData(index)
        if path is not None:
            self.select(path)
            return
        # Las acciones no son una selección: se vuelve a la que había.
        self.setCurrentIndex(-1 if self.current is None else self.findData(self.current))
        if text == self.BROWSE:
            self.browse.emit()
        elif text == self.ADD:
            self.add.emit()
