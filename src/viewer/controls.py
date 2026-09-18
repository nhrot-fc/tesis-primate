from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollBar,
    QSlider,
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
# Alturas de la banda visible en Hz, de más cerca a más lejos; la última entrada es la banda
# entera. La caja mediana de un hallazgo mide 2,7 kHz: con la banda entera ocupa un octavo.
BAND_HEIGHTS = [200, 500, 1000, 2000, 3000, 5000, 8000, 12000, 16000]
FULL_BAND = "Full"
HEIGHT_WIDTH = 84
# Paso del scroll vertical y fracción de la banda que mueven las flechas
HZ_STEP = 50
PAN_FRACTION = 0.25

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


# La banda de frecuencia como en Raven: una altura elegida de una lista (el zoom) y un scroll
# vertical que la mueve. Son dos widgets porque van en sitios distintos: el scroll pegado al
# espectrograma y la altura junto al ancho de ventana.
class Band(QObject):
    changed = pyqtSignal(float, float)  # low, high en Hz

    def __init__(self) -> None:
        super().__init__()
        self.top = 0.0
        self.current = (0.0, 0.0)
        self.bar = QScrollBar(Qt.Orientation.Vertical)
        # El mínimo abajo: el scroll sube cuando la banda sube.
        self.bar.setInvertedAppearance(True)
        self.bar.setInvertedControls(True)
        self.bar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.bar.setToolTip("Move the band (↑ ↓)")
        self.bar.valueChanged.connect(self.emit_changed)
        self.heights = QComboBox()
        self.heights.setFixedWidth(HEIGHT_WIDTH)
        self.heights.setToolTip("Band height (Shift + wheel); F shows the full band")
        self.heights.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.heights.currentIndexChanged.connect(self.rescale)

    # Sólo se rehace cuando cambia el tope (otra frecuencia de muestreo); queda en banda entera.
    def set_limits(self, top: float) -> None:
        if top == self.top:
            return
        self.top = top
        self.heights.blockSignals(True)
        self.heights.clear()
        for height in BAND_HEIGHTS:
            if height < top:
                text = f"{height / 1000:g} kHz" if height >= 1000 else f"{height} Hz"
                self.heights.addItem(text, float(height))
        self.heights.addItem(FULL_BAND, float(top))
        self.heights.blockSignals(False)
        self.set_values(0.0, top)

    def height(self) -> float:
        return float(self.heights.currentData() or self.top)

    def values(self) -> tuple[float, float]:
        low = self.bar.value() * HZ_STEP
        return low, low + self.height()

    def place_bar(self, low: float) -> None:
        height = self.height()
        self.bar.blockSignals(True)
        self.bar.setRange(0, max(int(round((self.top - height) / HZ_STEP)), 0))
        self.bar.setSingleStep(max(int(PAN_FRACTION * height / HZ_STEP), 1))
        self.bar.setPageStep(max(int(height / HZ_STEP), 1))
        self.bar.setValue(int(round(low / HZ_STEP)))
        self.bar.blockSignals(False)

    def move_to(self, low: float) -> None:
        self.place_bar(low)
        self.emit_changed()

    def emit_changed(self) -> None:
        self.current = self.values()
        self.changed.emit(*self.current)

    # Cambió la altura desde el combo: se conserva el centro que había.
    def rescale(self) -> None:
        low, high = self.current
        self.move_to(0.5 * (low + high) - self.height() / 2)

    # Lo que fija el espectrograma (el recorte a [0, Nyquist]) no debe rebotar como un cambio.
    def set_values(self, low: float, high: float) -> None:
        if not self.heights.count():
            return
        nearest = min(
            range(self.heights.count()),
            key=lambda i: abs(float(self.heights.itemData(i)) - (high - low)),
        )
        self.heights.blockSignals(True)
        self.heights.setCurrentIndex(nearest)
        self.heights.blockSignals(False)
        self.place_bar(low)
        self.current = (low, high)

    # Un paso de la lista de alturas, dejando quieto `around_hz` (el puntero) o el centro.
    def zoom(self, delta: int, around_hz: float | None = None) -> None:
        index = min(max(self.heights.currentIndex() + delta, 0), self.heights.count() - 1)
        if index == self.heights.currentIndex():
            return
        low, high = self.current
        anchor = 0.5 * (low + high) if around_hz is None else around_hz
        fraction = (anchor - low) / (high - low) if high > low else 0.5
        self.heights.blockSignals(True)
        self.heights.setCurrentIndex(index)
        self.heights.blockSignals(False)
        self.move_to(anchor - fraction * self.height())

    def pan(self, direction: int) -> None:
        self.bar.setValue(self.bar.value() + direction * self.bar.singleStep())

    def full(self) -> None:
        self.heights.setCurrentIndex(self.heights.count() - 1)

    # La altura más chica que deja la caja con aire alrededor, centrada en ella.
    def frame(self, low: float, high: float, margin: float = 1.5) -> None:
        wanted = (high - low) * margin
        index = next(
            (i for i in range(self.heights.count()) if float(self.heights.itemData(i)) >= wanted),
            self.heights.count() - 1,
        )
        self.heights.blockSignals(True)
        self.heights.setCurrentIndex(index)
        self.heights.blockSignals(False)
        self.move_to(0.5 * (low + high) - self.height() / 2)


# Cómo se ve y cómo se oye: se ajusta una vez por grabación, así que vive en un desplegable
# en lugar de ocupar filas fijas debajo del espectrograma. La banda va con el transporte.
class ViewPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.brightness = Slider("Brightness (dB)", *BRIGHTNESS)
        self.contrast = Slider("Contrast", *CONTRAST)
        # 0 dB es la grabación llevada a su pico; por encima recorta, que es lo que
        # hace audible una llamada lejana.
        self.volume = Slider("Volume (dB)", *VOLUME, fmt="{:+g}")
        self.volume.setToolTip("0 dB is the recording normalized to its peak; above that it clips")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        for control in (self.brightness, self.contrast, self.volume):
            control.setMinimumWidth(LABEL_WIDTH + SLIDER_WIDTH + READOUT_WIDTH)
            layout.addWidget(control)


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
