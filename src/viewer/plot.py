from pathlib import Path
from typing import override

import pandas as pd
import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401
from PyQt6.QtCore import QRectF, pyqtSignal
from PyQt6.QtGui import QFontMetrics
from PyQt6.QtWidgets import QGraphicsRectItem

from viewer.spectrogram import Waveform, db_baseline, db_levels, stft_db

BOX_COLUMNS = ["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]
ANNOTATION_COLOR = "#00d8ff"
DETECTION_COLOR = "#8cff3d"
PLAYHEAD_COLOR = "#c8ffd0"
AXIS_PAD = 12
AXIS_SAMPLE = "00000"
EXPORT_WIDTH = 2400

COLORMAPS = ["magma", "inferno", "viridis", "cividis", "gray", "gray_r"]


def read_boxes(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    missing = [column for column in BOX_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"'{path.name}' no tiene las columnas: {', '.join(missing)}")
    return table


def box_label(row: pd.Series) -> str:
    text = "/".join(str(row[c]) for c in ("Species", "Call type") if c in row.index)
    if "Score" in row.index:
        text = f"{text} {row['Score']:.2f}".strip()
    return text


class SpectrogramView(pg.PlotWidget):
    scrolled = pyqtSignal(int)
    moved = pyqtSignal(float, float)

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
        self.set_colormap(COLORMAPS[0])
        self.vb.addItem(self.image)
        self.playhead = pg.InfiniteLine(angle=90, movable=False)
        self.playhead.setPen(pg.mkPen(PLAYHEAD_COLOR, width=2))
        self.playhead.setZValue(20)
        self.playhead.hide()
        self.vb.addItem(self.playhead, ignoreBounds=True)
        if self.sceneObj is not None:
            self.sceneObj.sigMouseMoved.connect(self._on_move)

        self.waveform: Waveform | None = None
        self.sr = 1
        self.baseline = (-100.0, 0.0)
        self.baseline_n_fft = 0
        self.boxes: list = []

    def set_colormap(self, name: str) -> None:
        colormap = pg.colormap.getFromMatplotlib(name)
        if colormap is not None:
            self.image.setColorMap(colormap)

    def set_waveform(self, waveform: Waveform, sr: int) -> None:
        self.waveform = waveform
        self.sr = sr
        self.baseline_n_fft = 0
        self.vb.setYRange(0.0, sr / 2, padding=0)

    def draw(
        self, start: float, span: float, n_fft: int, hop: int, brightness: float, contrast: float
    ) -> None:
        if self.waveform is None:
            return
        if n_fft != self.baseline_n_fft:
            self.baseline = db_baseline(self.waveform, self.sr, n_fft)
            self.baseline_n_fft = n_fft

        first = int(start * self.sr)
        chunk = self.waveform[first : first + int(span * self.sr)]
        self.image.setImage(stft_db(chunk, n_fft, min(hop, n_fft)), autoLevels=False)
        self.image.setRect(QRectF(start, 0.0, chunk.size / self.sr, self.sr / 2))
        self.image.setLevels(db_levels(self.baseline, brightness, contrast))
        self.vb.setXRange(start, start + span, padding=0)

    def draw_boxes(self, tables: list, start: float, stop: float, score: float) -> list[int]:
        while self.boxes:
            self.vb.removeItem(self.boxes.pop())
        counts = []
        for table, color in tables:
            if table is None or table.empty:
                counts.append(0)
                continue
            visible = table[(table["End Time (s)"] > start) & (table["Begin Time (s)"] < stop)]
            if "Score" in visible.columns:
                visible = visible[visible["Score"] >= score]
            counts.append(len(visible))
            for _, row in visible.iterrows():
                x0, y0 = row["Begin Time (s)"], row["Low Freq (Hz)"]
                width = row["End Time (s)"] - x0
                height = row["High Freq (Hz)"] - y0
                rect = QGraphicsRectItem(QRectF(x0, y0, width, height))
                rect.setPen(pg.mkPen(color, width=2))
                text = pg.TextItem(box_label(row), color=color, anchor=(0, 1))
                text.setPos(x0, y0 + height)
                self.vb.addItem(rect)
                self.vb.addItem(text)
                self.boxes += [rect, text]
        return counts

    def export_png(self, path: Path) -> None:
        exporter = pg.exporters.ImageExporter(self.getPlotItem())
        exporter.parameters()["width"] = EXPORT_WIDTH
        exporter.export(str(path))

    def set_playhead(self, seconds: float | None) -> None:
        if seconds is None:
            self.playhead.hide()
        else:
            self.playhead.setPos(seconds)
            self.playhead.show()

    def _on_move(self, position) -> None:
        point = self.vb.mapSceneToView(position)
        self.moved.emit(point.x(), point.y())

    @override
    def wheelEvent(self, ev) -> None:
        self.scrolled.emit(-1 if ev.angleDelta().y() > 0 else 1)
