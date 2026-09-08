import math
from pathlib import Path
from typing import override

import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QGraphicsRectItem
from pyqtgraph.exporters import ImageExporter

from viewer.session import BEGIN, END, HIGH, LOW, label
from viewer.spectrogram import Waveform, db_baseline, db_levels, stft_db
from viewer.tasks import Latest

COLORMAP = "magma"
PLAYHEAD_COLOR = "#ffffff"
HIGHLIGHT_COLOR = "#ffffff"
HIGHLIGHT_MARGIN = 0.012
AXIS_PAD = 12
AXIS_SAMPLE = "00000"
EXPORT_WIDTH = 2400
# Se calcula una banda mas ancha que la ventana visible: mientras el scroll no salga
# de ella, moverse no cuesta ni una FFT.
BAND = 4.0

# n_fft = 8 * hop en todas: el solape fija la resolucion en frecuencia.
RESOLUTIONS = [(512, 64), (1024, 128), (2048, 256), (4096, 512), (8192, 1024)]
COLUMNS_ON_SCREEN = 900

# Zoom del eje de frecuencia. La caja mediana de un hallazgo mide 2,7 kHz sobre los
# 22 kHz del eje completo: sin acotar la banda ocupa un octavo de la pantalla.
MIN_BAND_HZ = 200.0
BAND_STEP = 1.25
PAN_FRACTION = 0.25


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

    def __init__(self) -> None:
        super().__init__()
        item = self.getPlotItem()

        if item is None:
            raise RuntimeError("No se pudo obtener el PlotItem del SpectrogramView")

        item.showAxes(True, showValues=(True, False, False, True))  # pyright: ignore[reportArgumentType]
        item.getAxis("bottom").enableAutoSIPrefix(False)
        item.getAxis("left").enableAutoSIPrefix(False)
        item.setMenuEnabled(False)
        item.hideButtons()

        metrics = QFontMetrics(self.font())
        for key in ("left", "right"):
            item.getAxis(key).setWidth(metrics.horizontalAdvance(AXIS_SAMPLE) + AXIS_PAD)
        for key in ("top", "bottom"):
            item.getAxis(key).setHeight(metrics.height() + AXIS_PAD)

        self.vb = item.getViewBox()
        self.vb.setMouseEnabled(x=False, y=False)
        self.vb.setDefaultPadding(0.0)
        self.image = pg.ImageItem()
        colormap = pg.colormap.getFromMatplotlib(COLORMAP)
        if colormap is not None:
            self.image.setColorMap(colormap)
        self.vb.addItem(self.image)
        self.playhead = pg.InfiniteLine(angle=90, movable=False)
        self.playhead.setPen(pg.mkPen(PLAYHEAD_COLOR, width=2))
        self.playhead.setZValue(20)
        self.playhead.hide()
        self.vb.addItem(self.playhead, ignoreBounds=True)
        self.highlight = QGraphicsRectItem()
        self.highlight.setPen(pg.mkPen(HIGHLIGHT_COLOR, width=2, style=Qt.PenStyle.DashLine))
        self.highlight.setZValue(15)
        self.highlight.setVisible(False)
        self.vb.addItem(self.highlight)
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

        self.renderer = Latest()
        self.renderer.done.connect(self.on_render)

    def set_waveform(self, waveform: Waveform, sr: int) -> None:
        previous = self.sr
        self.waveform = waveform
        self.sr = sr
        self.baseline_n_fft = 0
        self.band = None
        self.set_highlight(None)
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

    def pan_band(self, direction: int) -> None:
        low, high = self.vb.viewRange()[1]
        step = direction * PAN_FRACTION * (high - low)
        self.set_band(low + step, high + step)

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
            # Fondo translucido: sin el, el verde sobre magma claro es ilegible.
            text = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush(0, 0, 0, 150))
            self.vb.addItem(rect)
            self.vb.addItem(text)
            self.pool.append((rect, text))
        return self.pool[index]

    def draw_boxes(self, layers: list, start: float, stop: float) -> list[int]:
        # Reusa los items ya creados: redibujar al mover la barra no construye nada.
        counts, used = [], 0
        for table, color, above in layers:
            if table is None or table.empty:
                counts.append(0)
                continue
            visible = table[(table[END] > start) & (table[BEGIN] < stop)]
            counts.append(len(visible))
            pen = pg.mkPen(color, width=2)
            for _, row in visible.iterrows():
                rect, text = self.box_slot(used)
                x0, y0 = row[BEGIN], row[LOW]
                height = row[HIGH] - y0
                rect.setRect(QRectF(x0, y0, row[END] - x0, height))
                rect.setPen(pen)
                rect.setVisible(True)
                caption = label(row)
                if "Score" in row.index:
                    caption = f"{caption} {row['Score']:.2f}".strip()
                text.setText(caption, color=color)
                # Cada capa rotula a un lado del borde superior --una afuera y otra
                # adentro-- para que una deteccion encima de su anotacion no la tape.
                text.setAnchor((0, 1) if above else (0, 0))
                text.setPos(x0, y0 + height)
                text.setVisible(True)
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
            hz = self.vb.mapSceneToView(ev.position()).y()
            low, high = self.vb.viewRange()[1]
            factor = BAND_STEP**direction
            self.set_band(hz + (low - hz) * factor, hz + (high - hz) * factor)
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            self.zoomed.emit(direction)
        else:
            self.scrolled.emit(direction)
