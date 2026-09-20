"""Graba el vídeo de demostración del visor (`resources/demo.mp4`): un guion fijo abre una
grabación de `data/`, corre YOLO en CPU, escucha, revisa, guarda y procesa una carpeta en
Batch. Cada fotograma es la ventana (`grab()`) con un cursor dibujado, la tecla que se pulsa
y un subtítulo debajo; los fotogramas van por tubería a ffmpeg.

    uv run python docs/manual/demo.py [--audio WAV] [--model CHECKPOINT] [--out MP4]

Corre con la plataforma `offscreen` de Qt y sin CUDA, así el vídeo sale igual en cualquier
máquina y la detección es la de una PC sin GPU. No escribe nada junto al audio: lo que
Batch y Save producen va a una carpeta temporal."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_FONT_DPI"] = "96"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # la demo es la de una PC sin GPU
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PyQt6.QtCore import QEventLoop, QPoint, QPointF, QRect, Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu, QTableView, QToolButton, QWidget  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO = PROJECT / "data/raw/bolivian_squirrel_monkey__SB/20231218.wav"
DEFAULT_MODEL = PROJECT / "runs/yolo26s_v2/best.pt"
DEFAULT_OUT = PROJECT / "resources/demo.mp4"
# Otras dos grabaciones para la carpeta de Batch
BATCH_EXTRA = (
    PROJECT / "data/raw/shock_headed_capuchin_monkey__CC/20241209_131346.wav",
    PROJECT / "data/raw/howler_monkey__AS/20240120_053301.wav",
)

WINDOW = (1120, 720)
FPS = 20
TICK_MS = 1000 // FPS
# Franja bajo la ventana para el subtítulo: no tapa nada de la interfaz.
CAPTION_HEIGHT = 48
CAPTION_BACKGROUND, CAPTION_TEXT = "#1c1b22", "#f4f2f8"
KEY_BACKGROUND, KEY_TEXT = "#f4f2f8", "#1c1b22"
CURSOR_SPEED = 0.22  # fracción del camino que recorre el cursor por fotograma
CLICK_FRAMES = 8
KEY_FRAMES = 24


def wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


# Un cursor de flecha clásico, pintado a mano: blanco con borde negro se ve sobre cualquier fondo.
def cursor_polygon(at: QPointF) -> QPolygonF:
    shape = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
    return QPolygonF([QPointF(at.x() + x, at.y() + y) for x, y in shape])


class Recorder:
    def __init__(self, viewer, out: Path) -> None:
        self.viewer = viewer
        width, height = WINDOW
        self.size = (width, height + CAPTION_HEIGHT)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{self.size[0]}x{self.size[1]}",
                "-r", str(FPS), "-i", "-",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium",
                "-movflags", "+faststart", str(out),
            ],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        self.cursor = QPointF(width * 0.55, height * 0.6)
        self.target = QPointF(self.cursor)
        self.caption = ""
        self.key = ""
        self.key_frames = 0
        self.click_frames = 0
        self.frames = 0

    # --- Un fotograma --------------------------------------------------------------

    def frame(self) -> None:
        width, height = self.size
        canvas = QPixmap(width, height)
        canvas.fill(QColor(CAPTION_BACKGROUND))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(0, 0, self.viewer.grab())
        origin = self.viewer.mapToGlobal(QPoint(0, 0))
        # Los menús abiertos son ventanas aparte: se pegan donde caen sobre la principal.
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMenu) and widget.isVisible():
                painter.drawPixmap(widget.pos() - origin, widget.grab())
        self.draw_click(painter)
        self.draw_cursor(painter)
        self.draw_key(painter)
        self.draw_caption(painter)
        painter.end()
        image = canvas.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        bits = image.constBits()
        assert bits is not None and self.ffmpeg.stdin is not None
        self.ffmpeg.stdin.write(bits.asstring(image.sizeInBytes()))
        self.frames += 1

    def draw_cursor(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor("black"), 1.5))
        painter.setBrush(QColor("white"))
        painter.drawPolygon(cursor_polygon(self.cursor))

    def draw_click(self, painter: QPainter) -> None:
        if self.click_frames <= 0:
            return
        age = CLICK_FRAMES - self.click_frames
        radius = 6 + 3 * age
        color = QColor("#1E66F5")
        color.setAlpha(int(200 * (1 - age / CLICK_FRAMES)))
        painter.setPen(QPen(color, 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(self.cursor, radius, radius)
        self.click_frames -= 1

    def draw_key(self, painter: QPainter) -> None:
        if self.key_frames <= 0 or not self.key:
            return
        font = QFont(painter.font())
        font.setPointSize(13)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad = 10
        box = QRect(0, 0, metrics.horizontalAdvance(self.key) + 2 * pad, metrics.height() + 2 * pad)
        box.moveBottomRight(QPoint(WINDOW[0] - 16, WINDOW[1] - 40))
        painter.setPen(QPen(QColor(KEY_TEXT), 2))
        painter.setBrush(QColor(KEY_BACKGROUND))
        painter.drawRoundedRect(box, 6, 6)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self.key)
        self.key_frames -= 1

    def draw_caption(self, painter: QPainter) -> None:
        font = QFont(painter.font())
        font.setPointSize(13)
        painter.setFont(font)
        painter.setPen(QColor(CAPTION_TEXT))
        band = QRect(0, WINDOW[1], WINDOW[0], CAPTION_HEIGHT)
        painter.drawText(band, Qt.AlignmentFlag.AlignCenter, self.caption)

    # --- El paso del tiempo ------------------------------------------------------------

    # Avanza `ms` de reloj real grabando fotogramas; el cursor se acerca a su destino.
    def run(self, ms: int) -> None:
        for _ in range(max(ms // TICK_MS, 1)):
            self.cursor += (self.target - self.cursor) * CURSOR_SPEED
            wait(TICK_MS)
            self.frame()

    def run_until(self, done, timeout_ms: int = 120_000) -> None:
        for _ in range(timeout_ms // TICK_MS):
            self.run(TICK_MS)
            if done():
                return
        raise SystemExit("timed out waiting for the viewer")

    def say(self, text: str) -> None:
        self.caption = text

    def press(self, key: str) -> None:
        self.key, self.key_frames = key, KEY_FRAMES

    # --- El cursor ----------------------------------------------------------------------

    def center_of(self, widget: QWidget) -> QPointF:
        rect = widget.rect()
        return QPointF(widget.mapTo(self.viewer, rect.center()))

    def move_to(self, widget: QWidget, ms: int = 600) -> None:
        self.target = self.center_of(widget)
        self.run(ms)

    # Lleva el cursor al widget, dibuja el clic y, si se pasa, ejecuta lo que el clic haría.
    def click(self, widget: QWidget, action=None, ms: int = 600) -> None:
        self.move_to(widget, ms)
        self.click_frames = CLICK_FRAMES
        self.run(100)
        if action is not None:
            action()
        self.run(150)

    # Abre el menú de un botón, resalta una entrada y lo cierra: lo que se ve al elegirla.
    def pick(self, button: QToolButton, text: str, hold_ms: int = 1100) -> None:
        menu = button.menu()
        assert menu is not None
        self.click(button)
        at = button.mapTo(self.viewer, QPoint(0, button.height()))
        menu.popup(self.viewer.mapToGlobal(at))
        action = next(a for a in menu.actions() if a.text().startswith(text))
        self.run(300)
        menu.setActiveAction(action)
        self.target = QPointF(at + menu.actionGeometry(action).center())
        self.run(hold_ms)
        self.click_frames = CLICK_FRAMES
        self.run(200)
        menu.hide()

    def close(self) -> None:
        assert self.ffmpeg.stdin is not None
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()
        print(f"{self.frames} frames, {self.frames / FPS:.0f} s")


def batch_folder(audio: Path) -> Path:
    folder = Path(tempfile.gettempdir()) / "demo" / "recordings"
    shutil.rmtree(folder.parent, ignore_errors=True)
    folder.mkdir(parents=True)
    for source in (audio, *BATCH_EXTRA):
        if source.is_file():
            shutil.copy2(source, folder / source.name)
    return folder


def cell_center(viewer, table: QTableView, row: int) -> QPointF:
    model, viewport = table.model(), table.viewport()
    assert model is not None and viewport is not None
    return QPointF(viewport.mapTo(viewer, table.visualRect(model.index(row, 0)).center()))


def toolbar_button(viewer, action) -> QWidget:
    toolbar = viewer.open_button.parentWidget()
    button = toolbar.widgetForAction(action)
    assert button is not None
    return button


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    app = QApplication([])
    from data.raven import renumber
    from viewer.app import BATCH, run
    from viewer.review import ACCEPTED, REJECTED, journal_path
    from viewer.session import ANNOTATIONS, DETECTIONS

    viewer = run(app, [])
    viewer.resize(*WINDOW)
    viewer.picker.select(args.model.resolve())
    wait(400)
    rec = Recorder(viewer, args.out)
    audio = args.audio.resolve()
    scratch = batch_folder(audio)
    # Sin diario de una toma anterior: la revisión empieza desde la primera detección.
    journal_path(audio).unlink(missing_ok=True)

    def loaded() -> bool:
        return not viewer.busy() and viewer.session.audio_path == audio

    # 1. Abrir la grabación
    rec.say("viewer.exe — open a recording, detect, review, save")
    rec.run(2200)
    rec.pick(viewer.open_button, "Open audio")
    rec.say(f"Open audio…  (Ctrl+O)   {audio.relative_to(PROJECT)}")
    rec.press("Ctrl+O")
    viewer.open_path(audio)
    rec.run_until(loaded)
    rec.run(1500)

    # 2. Mirar: zoom y desplazamiento
    rec.say("Ctrl + wheel zooms in time; the wheel scrolls")
    rec.target = rec.center_of(viewer.plot)
    rec.run(600)
    rec.press("Ctrl + wheel")
    viewer.transport.zoom(1)
    rec.run(900)
    for _ in range(6):
        viewer.transport.step(1)
        rec.run(150)
    rec.run(700)

    # 3. Las anotaciones hechas a mano, encima; la ventana va donde hay varias
    annotations = audio.with_suffix(".txt")
    if annotations.is_file():
        rec.pick(viewer.open_button, "Open annotations")
        rec.say("Open annotations…  (Ctrl+T): a Raven table drawn in green")
        rec.press("Ctrl+T")
        viewer.open_path(annotations)
        rec.run_until(lambda: viewer.session.tables[ANNOTATIONS] is not None)
        rec.run(600)
        rows = viewer.session.rows(ANNOTATIONS)
        if rows:
            busiest = max(rows, key=lambda r: sum(abs(o.begin - r.begin) < 4 for o in rows))
            viewer.transport.center(busiest.begin)
        rec.run(1800)

    # 4. Escuchar
    rec.say("Space plays from the playhead; click the spectrogram to move it")
    rec.target = rec.center_of(viewer.plot)
    rec.run(500)
    rec.press("Space")
    viewer.transport.toggle_play()
    rec.run(3000)
    rec.press("Space")
    viewer.transport.toggle_play()
    rec.run(600)

    # 5. Detectar (el motor ya cargó mientras tanto, si no se espera)
    rec.say("Detect  (Ctrl+R): runs yolo26s_v2 on the CPU")
    rec.move_to(viewer.run_button)
    rec.run_until(lambda: viewer.run_action.isEnabled())
    rec.click(viewer.run_button, viewer.run_model)
    rec.press("Ctrl+R")
    rec.run_until(lambda: viewer.session.tables[DETECTIONS] is not None)
    found = len(viewer.session.rows(DETECTIONS))
    rec.say(f"{found} detections in blue, each with its score")
    rec.run(2200)

    rec.say("Score ≥ hides the weaker ones; nothing is deleted")
    rec.click(viewer.score.slider)
    viewer.score.set_value(0.4)
    rec.run(1500)
    viewer.score.set_value(0.2)
    rec.run(1200)

    # 6. La lista de cajas
    boxes_button = toolbar_button(viewer, viewer.boxes_action)
    rec.say("Boxes  (Ctrl+E): the list; a click frames the box")
    rec.click(boxes_button, viewer.table.show)
    rec.press("Ctrl+E")
    rec.run(1200)
    table = viewer.table.pages[DETECTIONS].table
    for row in (2, 5):
        rec.target = cell_center(viewer, table, row)
        rec.run(500)
        rec.click_frames = CLICK_FRAMES
        table.selectRow(row)
        rec.run(1100)
    rec.click(boxes_button, viewer.table.hide)
    rec.run(500)

    # 7. Revisar
    rec.say("Review: one detection at a time, framed, with handles")
    rec.click(viewer.review_button, viewer.reviewer.start)
    rec.run_until(lambda: viewer.reviewer.active)
    rec.run(1800)
    for key, decision, text, label in (
        ("A", ACCEPTED, "A accepts: the box moves to the annotations", None),
        ("A", ACCEPTED, "A accepts", None),
        ("R", REJECTED, "R rejects: the box is dropped", None),
        ("A", ACCEPTED, "The label can be corrected before accepting", "SB/PCC"),
    ):
        rec.say(text)
        rec.target = rec.center_of(
            viewer.review_bar.accept_button if decision == ACCEPTED else viewer.review_bar
        )
        rec.run(1300)
        if label is not None:
            rec.click(viewer.review_bar.species)
            viewer.review_bar.species.setCurrentText(label)
            rec.run(1200)
            rec.target = rec.center_of(viewer.review_bar.accept_button)
            rec.run(500)
        rec.press(key)
        viewer.reviewer.decide(decision)
        rec.run(900)
    rec.say("Esc leaves; the review resumes next time")
    rec.press("Esc")
    viewer.reviewer.stop()
    rec.run(1800)

    # 8. Guardar (el diálogo de archivo no se puede mostrar: se escribe donde iría)
    rec.pick(viewer.export_button, "Save annotations")
    rec.say("Save annotations table…: a Raven table with the accepted boxes")
    saved = scratch / f"{audio.stem}.annotations.txt"
    table_out = viewer.session.visible(ANNOTATIONS)
    assert table_out is not None
    renumber(table_out.copy()).to_csv(saved, sep="\t", index=False)
    viewer.say(f"{len(table_out)} rows saved to {saved.name}")
    rec.run(2200)

    # 9. Batch: una carpeta entera
    rec.say("Batch: a whole folder")
    rec.click(toolbar_button(viewer, viewer.mode_actions[BATCH]), lambda: viewer.set_mode(BATCH))
    rec.run(600)
    rec.say("Drop a folder on the window, or Choose a folder…")
    rec.click(viewer.batch.choose_button, lambda: viewer.batch.set_folder(scratch))
    rec.run_until(lambda: viewer.batch.listed() and not viewer.batch.scanning())
    rec.run(1500)
    rec.say("Run: every recording gets its .detections.txt")
    rec.click(viewer.batch.run_button, viewer.batch.run)
    rec.run_until(lambda: not viewer.batch.running())
    rec.run(2000)
    rec.say("Double-click a row to open it in the Spectrogram view")
    rec.target = cell_center(viewer, viewer.batch.table, 1)
    rec.run(700)
    rec.click_frames = CLICK_FRAMES
    viewer.batch.open_requested.emit(viewer.batch.files.entries[1].path)
    rec.run_until(lambda: not viewer.busy() and viewer.mode() != BATCH)
    rec.run(2500)

    rec.say("Manual.pdf and F1 list every control")
    rec.run(2500)
    rec.close()
    viewer.close()
    shutil.rmtree(scratch.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
