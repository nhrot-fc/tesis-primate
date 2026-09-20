import math
from pathlib import Path
from typing import NamedTuple, override

import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics
from PyQt6.QtWidgets import QGraphicsRectItem
from pyqtgraph.exporters import ImageExporter

from data.raven import BEGIN, END, HIGH, LOW, SCORE
from viewer.session import label
from viewer.spectrogram import Waveform, db_baseline, db_levels, stft_db
from viewer.tasks import Latest

COLORMAP = "magma"
# Blanco y ámbar: ninguno de los dos es el color de una capa de cajas.
PLAYHEAD_COLOR = "#ffffff"
HIGHLIGHT_COLOR = "#ffd54f"
HIGHLIGHT_MARGIN = 0.012
# Un relleno tenue: la caja se lee como region aunque su borde cruce una zona clara.
FILL_ALPHA = 34
CAPTION_ALPHA = 165
CAPTION_POINT_SIZE = 8
AXIS_PAD = 10
AXIS_SAMPLE = "00000"
# Arriba y a la derecha el eje sólo cierra el marco: sin valores que escribir, cuanto más
# fino, más espectrograma. Lo que se gana ahí es lo que más se mira.
EDGE_AXIS = 2
EXPORT_WIDTH = 2400
# Se calcula una banda mas ancha que la ventana visible: mientras el scroll no salga
# de ella, moverse no cuesta ni una FFT.
BAND = 4.0

# n_fft = 8 * hop en todas: el solape fija la resolucion en frecuencia.
RESOLUTIONS = [(512, 64), (1024, 128), (2048, 256), (4096, 512), (8192, 1024)]
COLUMNS_ON_SCREEN = 900

# La banda visible no baja de esto; sus alturas las elige `controls.Band`.
MIN_BAND_HZ = 200.0
# Separación entre marcas a partir de la cual el eje escribe kHz en vez de Hz.
KHZ_FROM_HZ = 500.0
# La caja en revisión se dibuja con asas más grandes que las de pyqtgraph.
HANDLE_SIZE = 9
# El rótulo sólo se escribe si la caja mide al menos esta fracción de su ancho: en una
# ventana de 30 s las cajas son de veinte píxeles y los rótulos se pisaban unos a otros.
CAPTION_MIN_FILL = 0.6


# En banda ancha "20k" se lee de un vistazo y "20000" hay que contarlo. Cuando las marcas
# caen a menos de 500 Hz el kilo ya no las distingue y vuelven los Hz enteros.
class FrequencyAxis(pg.AxisItem):
    @override
    def tickStrings(self, values, scale, spacing) -> list[str]:
        if spacing < KHZ_FROM_HZ:
            return [f"{value:,.0f}" for value in values]
        return [f"{value / 1000:g}k" if value else "0" for value in values]


class Layer(NamedTuple):
    table: pd.DataFrame | None  # None cuando la capa esta apagada: ni se dibuja ni se cuenta
    color: str
    style: Qt.PenStyle
    width: int
    top: bool  # rotula pegado al borde superior de la caja; si no, al inferior


def resolution(span: float, sr: int) -> tuple[int, int]:
    # El hop que deja ~900 columnas en pantalla, redondeado en octavas a las de la lista.
    hop = span * sr / COLUMNS_ON_SCREEN
    return min(RESOLUTIONS, key=lambda pair: abs(math.log2(pair[1] / hop)))


class SpectrogramView(pg.PlotWidget):
    scrolled = pyqtSignal(int)
    zoomed = pyqtSignal(int)
    moved = pyqtSignal(float, float)
    clicked = pyqtSignal(float)
    banded = pyqtSignal(float, float)
    band_zoomed = pyqtSignal(int, float)  # paso y frecuencia bajo el puntero

    def __init__(self) -> None:
        super().__init__(axisItems={"left": FrequencyAxis(orientation="left")})
        item = self.getPlotItem()

        if item is None:
            raise RuntimeError("SpectrogramView has no PlotItem")

        item.showAxes(True, showValues=(True, False, False, True))  # pyright: ignore[reportArgumentType]
        item.getAxis("bottom").enableAutoSIPrefix(False)
        item.getAxis("left").enableAutoSIPrefix(False)
        item.setMenuEnabled(False)
        item.hideButtons()

        metrics = QFontMetrics(self.font())
        item.getAxis("left").setWidth(metrics.horizontalAdvance(AXIS_SAMPLE) + AXIS_PAD)
        item.getAxis("bottom").setHeight(metrics.height() + AXIS_PAD)
        item.getAxis("right").setWidth(EDGE_AXIS)
        item.getAxis("top").setHeight(EDGE_AXIS)
        # Sin marco ni margen propios: el espectrograma llega al borde del hueco que le toca.
        # El margen interno del PlotItem se queda: llegar a él es `item.layout`, un atributo
        # que pyqtgraph pone encima del método `layout()` de Qt, y son dos píxeles.
        self.setFrameStyle(0)
        self.setContentsMargins(0, 0, 0, 0)

        self.vb = item.getViewBox()
        self.vb.setMouseEnabled(x=False, y=False)
        self.vb.setDefaultPadding(0.0)
        self.image = pg.ImageItem()
        self.image.setColorMap(COLORMAP)  # pyqtgraph resuelve el nombre; es el magma de matplotlib
        self.vb.addItem(self.image)
        self.playhead = pg.InfiniteLine(angle=90, movable=False)
        self.playhead.setPen(pg.mkPen(PLAYHEAD_COLOR, width=1.5))
        self.playhead.setZValue(20)
        self.playhead.hide()
        self.vb.addItem(self.playhead, ignoreBounds=True)
        self.highlight = QGraphicsRectItem()
        self.highlight.setPen(pg.mkPen(HIGHLIGHT_COLOR, width=3, style=Qt.PenStyle.DashLine))
        self.highlight.setZValue(15)
        self.highlight.setVisible(False)
        self.vb.addItem(self.highlight)
        # La caja en revisión: se arrastra entera o por cualquiera de sus cuatro esquinas.
        self.roi = pg.RectROI(
            [0.0, 0.0],
            [1.0, 1.0],
            pen=pg.mkPen(HIGHLIGHT_COLOR, width=2),
            hoverPen=pg.mkPen(HIGHLIGHT_COLOR, width=3),
            movable=True,
            rotatable=False,
            removable=False,
        )
        self.roi.handleSize = HANDLE_SIZE  # las que se agregan salen de este tamaño
        for position, center in (([0, 0], [1, 1]), ([1, 0], [0, 1]), ([0, 1], [1, 0])):
            self.roi.addScaleHandle(position, center)
        for handle in self.roi.getHandles():  # y la que RectROI trae de fábrica
            handle.radius = HANDLE_SIZE
            handle.buildPath()
        self.roi.setZValue(16)
        self.roi.hide()
        self.vb.addItem(self.roi)
        if self.sceneObj is not None:
            self.sceneObj.sigMouseMoved.connect(self.on_move)

        self.waveform: Waveform | None = None
        self.sr = 1
        self.baseline = (-100.0, 0.0)
        self.baseline_n_fft = 0
        self.levels = (0.0, 1.0)
        self.band: tuple[float, float, int, int] | None = None
        self.pending: tuple[float, float, int, int] | None = None
        self.job = 0
        self.pool: list[tuple[QGraphicsRectItem, pg.TextItem]] = []
        caption_font = QFont(self.font())
        caption_font.setPointSize(CAPTION_POINT_SIZE)
        self.caption_metrics = QFontMetrics(caption_font)

        self.renderer = Latest()
        self.renderer.done.connect(self.on_render)

    def set_waveform(self, waveform: Waveform, sr: int) -> None:
        previous = self.sr
        self.waveform = waveform
        self.sr = sr
        self.baseline_n_fft = 0
        self.band = None
        self.set_highlight(None)
        self.set_editable(None)
        # La banda elegida se conserva de una grabación a otra: moverla es cosa del usuario.
        if sr != previous:
            self.set_band(0.0, sr / 2)

    def draw(self, start: float, span: float, brightness: float, contrast: float) -> None:
        if self.waveform is None:
            return
        n_fft, hop = resolution(span, self.sr)
        self.levels = (brightness, contrast)
        self.vb.setXRange(start, start + span, padding=0)
        if self.covers(start, span, n_fft, hop):
            self.image.setLevels(db_levels(self.baseline, *self.levels))
            return

        band_span = span * BAND
        band_start = max(start - (band_span - span) / 2, 0.0)
        first = int(band_start * self.sr)
        chunk = self.waveform[first : first + int(band_span * self.sr)]
        # La linea base recorre el audio entero, asi que solo se recalcula al cambiar
        # n_fft y viaja al hilo junto con la banda.
        whole = self.waveform if n_fft != self.baseline_n_fft else None
        sr = self.sr

        self.pending = (band_start, chunk.size / sr, n_fft, hop)
        self.job = self.renderer.submit(
            lambda: (
                stft_db(chunk, n_fft, min(hop, n_fft)),
                db_baseline(whole, sr, n_fft) if whole is not None else None,
            )
        )

    # La banda visible se acota a [0, Nyquist], que es todo lo que hay dibujado.
    def set_band(self, low: float, high: float) -> None:
        top = self.sr / 2
        span = min(max(high - low, MIN_BAND_HZ), top)
        center = min(max(0.5 * (low + high), span / 2), top - span / 2)
        self.vb.setYRange(center - span / 2, center + span / 2, padding=0)
        self.banded.emit(center - span / 2, center + span / 2)

    def covers(self, start: float, span: float, n_fft: int, hop: int) -> bool:
        if self.band is None:
            return False
        band_start, band_span, band_n_fft, band_hop = self.band
        inside = band_start <= start and start + span <= band_start + band_span + 1e-6
        return inside and (n_fft, hop) == (band_n_fft, band_hop)

    def on_render(self, job_id: int, result) -> None:
        if job_id != self.job or result is None or self.pending is None:
            return
        image, baseline = result
        band_start, band_span, n_fft, _ = self.pending
        if baseline is not None:
            self.baseline = baseline
            self.baseline_n_fft = n_fft
        self.band = self.pending
        self.image.setImage(image, autoLevels=False)
        self.image.setRect(QRectF(band_start, 0.0, band_span, self.sr / 2))
        self.image.setLevels(db_levels(self.baseline, *self.levels))

    def box_slot(self, index: int) -> tuple[QGraphicsRectItem, pg.TextItem]:
        while len(self.pool) <= index:
            rect = QGraphicsRectItem()
            # Fondo translucido: sin el, el rotulo sobre una zona clara es ilegible.
            text = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush(0, 0, 0, CAPTION_ALPHA))
            font = QFont(self.font())
            font.setPointSize(CAPTION_POINT_SIZE)
            text.setFont(font)
            self.vb.addItem(rect)
            self.vb.addItem(text)
            self.pool.append((rect, text))
        return self.pool[index]

    def draw_boxes(self, layers: list[Layer], start: float, stop: float) -> list[int]:
        # Reusa los items ya creados: redibujar al mover la barra no construye nada.
        counts, used = [], 0
        (view_x0, _), (view_y0, view_y1) = self.vb.viewRange()
        seconds_per_px = float(self.vb.viewPixelSize()[0]) or 1.0
        for layer in layers:
            table = layer.table
            if table is None or table.empty:
                counts.append(0)
                continue
            visible = table[(table[END] > start) & (table[BEGIN] < stop)]
            counts.append(len(visible))
            pen = pg.mkPen(layer.color, width=layer.width, style=layer.style)
            tint = QColor(layer.color)
            tint.setAlpha(FILL_ALPHA)
            brush = pg.mkBrush(tint)
            scored = SCORE in table.columns
            for _, row in visible.iterrows():
                rect, text = self.box_slot(used)
                x0, y0 = row[BEGIN], row[LOW]
                width, height = row[END] - x0, row[HIGH] - y0
                rect.setRect(QRectF(x0, y0, width, height))
                rect.setPen(pen)
                rect.setBrush(brush)
                rect.setVisible(True)
                # Cada caja lleva siempre su rotulo, y cada capa lo pega a un borde distinto
                # para que una deteccion encima de su anotacion no lo tape. Si la caja sale
                # de la pantalla, el rotulo se queda en el borde visible.
                caption = label(row)
                if scored:
                    caption = f"{caption} {row[SCORE]:.2f}".strip()
                x = max(x0, view_x0)
                y = min(y0 + height, view_y1) if layer.top else max(y0, view_y0)
                text.setText(caption, color=layer.color)
                text.setAnchor((0, 0) if layer.top else (0, 1))
                text.setPos(x, y)
                fits = width / seconds_per_px >= CAPTION_MIN_FILL * (
                    self.caption_metrics.horizontalAdvance(caption)
                )
                text.setVisible(fits)
                used += 1
        for rect, text in self.pool[used:]:
            rect.setVisible(False)
            text.setVisible(False)
        return counts

    def set_highlight(self, box: tuple[float, float, float, float] | None) -> None:
        if box is None:
            self.highlight.setVisible(False)
            return
        begin, end, low, high = box
        # El marco se separa un poco del borde: encima de la caja sus lineas se confunden.
        (x0, x1), (y0, y1) = self.vb.viewRange()
        dx, dy = HIGHLIGHT_MARGIN * (x1 - x0), HIGHLIGHT_MARGIN * (y1 - y0)
        rect = QRectF(begin - dx, low - dy, end - begin + 2 * dx, high - low + 2 * dy)
        self.highlight.setRect(rect)
        self.highlight.setVisible(True)

    # La caja que se está revisando, con asas; None la esconde.
    def set_editable(self, box: tuple[float, float, float, float] | None) -> None:
        if box is None:
            self.roi.hide()
            return
        begin, end, low, high = box
        self.roi.setPos([begin, low], update=False)
        self.roi.setSize([end - begin, high - low])
        self.roi.show()

    def editable_box(self) -> tuple[float, float, float, float]:
        position, size = self.roi.pos(), self.roi.size()
        begin, low = float(position[0]), float(position[1])
        return begin, begin + float(size[0]), low, low + float(size[1])

    def set_playhead(self, seconds: float | None) -> None:
        if seconds is None:
            self.playhead.hide()
        else:
            self.playhead.setPos(seconds)
            self.playhead.show()

    def export_png(self, path: Path) -> None:
        exporter = ImageExporter(self.getPlotItem())
        exporter.parameters()["width"] = EXPORT_WIDTH
        exporter.export(str(path))

    def close_renderer(self) -> None:
        self.renderer.close()

    def on_move(self, position) -> None:
        point = self.vb.mapSceneToView(position)
        self.moved.emit(point.x(), point.y())

    @override
    def mousePressEvent(self, ev) -> None:
        if ev is not None and ev.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.vb.mapSceneToView(ev.position()).x())
        super().mousePressEvent(ev)

    @override
    def wheelEvent(self, ev) -> None:
        if ev is None:
            return
        direction = -1 if ev.angleDelta().y() > 0 else 1
        modifiers = ev.modifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            # Alrededor del puntero: se apunta a la caja y se acerca sin perderla.
            self.band_zoomed.emit(direction, self.vb.mapSceneToView(ev.position()).y())
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            self.zoomed.emit(direction)
        else:
            self.scrolled.emit(direction)
