from pathlib import Path
from typing import override

from PyQt6.QtCore import QObject, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtMultimedia import QAudioDevice, QMediaDevices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QScrollBar,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from inference.catalog import available_models

LABEL_WIDTH = 92
# Caracteres que se ven del nombre del modelo en la lista cerrada.
MODEL_NAME_LENGTH = 14
READOUT_WIDTH = 48
SLIDER_WIDTH = 168
SLIDER_LEAST = 72
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
# Cuánto más grande que el texto normal va el titular de una página vacía.
HEADLINE_POINTS = 4


# El titular de una página vacía: la tipografía de la ventana, unos puntos más grande.
def headline_font(widget: QWidget) -> QFont:
    font = QFont(widget.font())
    font.setPointSize(font.pointSize() + HEADLINE_POINTS)
    return font


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
        self.slider.setMinimumWidth(SLIDER_LEAST)
        self.slider.setRange(0, len(values) - 1)
        self.slider.setValue(index)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.readout = QLabel(fmt.format(values[index]))
        self.readout.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
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


# Leyenda con interruptor: un checkbox por capa, con su trazo de icono, que la oculta. Una
# capa sin tabla no aparece: sin anotaciones ni modelo la fila queda vacía.
class Layers(QWidget):
    changed = pyqtSignal()

    def __init__(self, entries: list[tuple[str, str, Qt.PenStyle, int]]) -> None:
        super().__init__()
        self.boxes: dict[str, QCheckBox] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for text, color, style, width in entries:
            box = QCheckBox(text)
            box.setIcon(QIcon(swatch(color, style, width)))
            box.setChecked(True)
            box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            # Lambda: `toggled` emite el estado y `changed` no lleva argumentos.
            box.toggled.connect(lambda _: self.changed.emit())
            box.setVisible(False)
            self.boxes[text] = box
            layout.addWidget(box)

    def enabled(self, text: str) -> bool:
        return self.boxes[text].isChecked()

    def toggle(self, text: str) -> None:
        self.boxes[text].setChecked(not self.boxes[text].isChecked())

    # La leyenda dice qué es cada trazo y lo apaga; cuántas cajas hay lo dicen el título de
    # la ventana y el panel de cajas. Una capa sin tabla ni aparece.
    def set_state(self, text: str, total: int) -> None:
        self.boxes[text].setVisible(total > 0)


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
        self.heights.setToolTip(
            "How much of the frequency range fits on screen "
            "(Shift + wheel, Ctrl+↑ / Ctrl+↓); F shows it all"
        )
        self.heights.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.heights.currentIndexChanged.connect(self.rescale)
        self.heights_label = QLabel("Band")
        self.heights_label.setToolTip(self.heights.toolTip())

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


# Cómo se ve y cómo se oye: se ajusta una vez por sesión, así que vive en un desplegable en
# lugar de ocupar filas fijas debajo del espectrograma. La banda va con el transporte.
class SettingsPanel(QWidget):
    device_changed = pyqtSignal(object)  # QAudioDevice elegido; None es el del sistema

    SYSTEM_DEFAULT = "System default"

    def __init__(self) -> None:
        super().__init__()
        self.brightness = Slider("Brightness", *BRIGHTNESS)
        self.contrast = Slider("Contrast", *CONTRAST)
        # 0 dB es la grabación llevada a su pico; por encima recorta, que es lo que
        # hace audible una llamada lejana.
        self.volume = Slider("Volume", *VOLUME, fmt="{:+g} dB")
        self.volume.setToolTip("0 dB is the recording normalized to its peak; above that it clips")

        # La salida de audio, como en Audacity. La lista se pide después de que cargue el
        # motor de detección (`reload_outputs`), no acá: en Linux, tocar QMediaDevices antes
        # de que torch importe triton rompe el `dlopen` de libtriton. Hasta entonces sólo
        # está la del sistema, que es la que usa el reproductor sin elegir nada.
        self.output = QComboBox()
        self.output.setToolTip("Where the audio plays")
        self.output.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.output.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.output.addItem(self.SYSTEM_DEFAULT, None)
        self.devices: QMediaDevices | None = None  # avisa cuando enchufan o quitan una salida
        self.output.currentIndexChanged.connect(
            lambda _: self.device_changed.emit(self.output.currentData())
        )
        output_row = QWidget()
        row = QHBoxLayout(output_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        name = QLabel("Audio output")
        name.setFixedWidth(LABEL_WIDTH)
        row.addWidget(name)
        row.addWidget(self.output, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        for control in (self.brightness, self.contrast, self.volume, output_row):
            control.setMinimumWidth(LABEL_WIDTH + SLIDER_WIDTH + READOUT_WIDTH)
            layout.addWidget(control)

    # Se conserva lo elegido si sigue enchufado; si no, vuelve al del sistema.
    def reload_outputs(self) -> None:
        if self.devices is None:
            self.devices = QMediaDevices(self)
            self.devices.audioOutputsChanged.connect(self.reload_outputs)
        chosen = self.output.currentData()
        chosen_id = chosen.id() if isinstance(chosen, QAudioDevice) else None
        self.output.blockSignals(True)
        self.output.clear()
        self.output.addItem(self.SYSTEM_DEFAULT, None)
        for device in QMediaDevices.audioOutputs():
            self.output.addItem(device.description(), device)
            if chosen_id is not None and device.id() == chosen_id:
                self.output.setCurrentIndex(self.output.count() - 1)
        self.output.blockSignals(False)
        if self.output.currentData() is not chosen:
            self.device_changed.emit(self.output.currentData())


# El ancho de un panel lo decide el `sizeHint` de lo que lleva dentro, y el de una tabla es
# mezquino: sin esto el panel abre en su mínimo, que es el de una ventana chica.
class Panel(QWidget):
    def __init__(self, preferred: int, least: int) -> None:
        super().__init__()
        self.preferred = preferred
        self.setMinimumWidth(least)

    @override
    def sizeHint(self) -> QSize:
        # El ancho lo pone el panel, no la suma de sus tablas: una columna larga no debe
        # robarle sitio al espectrograma, que es lo que se mira.
        hint = super().sizeHint()
        hint.setWidth(self.preferred)
        return hint


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
        self.setToolTip("Model used by Detect and by Batch (Ctrl+M browses)")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # El ancho no lo fijan las dos entradas de acción del final, que son las más largas:
        # la lista desplegada sí las escribe enteras.
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(MODEL_NAME_LENGTH)
        self.current: Path | None = None
        self.reload()
        self.activated.connect(self.on_activated)

    def paths(self) -> list[Path]:
        return [self.itemData(i) for i in range(self.count()) if self.itemData(i) is not None]

    # `findData` compara los Path por identidad: uno leído de los ajustes nunca aparecería.
    def index_of(self, path: Path | None) -> int:
        return next((i for i in range(self.count()) if self.itemData(i) == path), -1)

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
        self.setCurrentIndex(self.index_of(path))
        if path != self.current:
            self.current = path
            self.chosen.emit(path)

    def on_activated(self, index: int) -> None:
        text, path = self.itemText(index), self.itemData(index)
        if path is not None:
            self.select(path)
            return
        # Las acciones no son una selección: se vuelve a la que había.
        self.setCurrentIndex(self.index_of(self.current))
        if text == self.BROWSE:
            self.browse.emit()
        elif text == self.ADD:
            self.add.emit()
