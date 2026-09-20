"""Genera las capturas del manual (`docs/manual/fig/`) desde el propio visor, sin compositor:
abre una grabación de muestra, corre el modelo, entra en revisión y lista una carpeta en Batch,
y en cada estado guarda la ventana entera (`widget.grab()`) y los recortes que el manual amplía.

    uv run python docs/manual/screenshots.py [--audio WAV] [--model CHECKPOINT]

Corre con la plataforma `offscreen` de Qt: las capturas salen iguales en cualquier máquina.
La carpeta de Batch se arma en un temporal con tres copias del audio (una con tabla) para que
la corrida dure segundos y la columna Status muestre los casos."""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_FONT_DPI"] = "96"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from PyQt6.QtCore import QEventLoop, QPoint, QRect, QTimer  # noqa: E402
from PyQt6.QtGui import QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMenu, QWidget  # noqa: E402

FIG = Path(__file__).resolve().parent / "fig"
DEFAULT_AUDIO = "data/raw/weddells_saddleBack_tamarin__LW/20240214_101201.wav"
WINDOW = (1120, 720)
# Píxeles de aire alrededor de cada recorte
PAD = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--audio", type=Path, default=Path(DEFAULT_AUDIO))
    parser.add_argument("--model", type=Path, help="checkpoint; por defecto el primero listado")
    return parser.parse_args()


def wait(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


class Shooter:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        FIG.mkdir(parents=True, exist_ok=True)

    # Ocupado de verdad, y con las señales pendientes ya entregadas.
    def settle(self, extra: int = 0) -> None:
        quiet = 0
        for _ in range(2400):
            wait(50)
            busy = self.viewer.busy() or self.viewer.pending_table is not None
            quiet = 0 if busy else quiet + 1
            if quiet >= 6:
                break
        wait(extra)

    def engine(self) -> None:
        while self.viewer.engine is None:
            wait(100)

    def window(self) -> QPixmap:
        wait(150)
        return self.viewer.grab()

    def save(self, name: str, pixmap: QPixmap) -> None:
        path = FIG / f"{name}.png"
        pixmap.save(str(path))
        print(f"{path.relative_to(Path.cwd())}  {pixmap.width()}x{pixmap.height()}")

    def rect_of(self, widget: QWidget) -> QRect:
        origin = widget.mapTo(self.viewer, QPoint(0, 0))
        return QRect(origin, widget.size()).adjusted(-PAD, -PAD, PAD, PAD)

    def crop(self, name: str, pixmap: QPixmap, *widgets: QWidget) -> None:
        rect = self.rect_of(widgets[0])
        for widget in widgets[1:]:
            rect = rect.united(self.rect_of(widget))
        self.save(name, pixmap.copy(rect.intersected(pixmap.rect())))

    # Un menú abierto es otra ventana: se captura aparte y se pega donde cae en la principal.
    def with_menu(self, name: str, menu: QMenu, at: QPoint) -> None:
        menu.popup(self.viewer.mapToGlobal(at))
        wait(200)
        base = self.viewer.grab()
        popup = menu.grab()
        from PyQt6.QtGui import QPainter

        painter = QPainter(base)
        painter.drawPixmap(at, popup)
        painter.end()
        menu.hide()
        wait(50)
        self.save(name, base)


def batch_folder(audio: Path) -> Path:
    from inference.catalog import output_for

    folder = Path(tempfile.gettempdir()) / "manual" / "recordings"
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "site_A").mkdir(parents=True)
    (folder / "site_B").mkdir(parents=True)
    for target in ("site_A/20240214_101201.wav", "site_A/20240214_103015.wav",
                   "site_B/20240215_064400.wav"):  # fmt: skip
        shutil.copy2(audio, folder / target)
    # Una ya tiene tabla: la corrida la salta y Status lo dice.
    table = output_for(audio)
    if table.is_file():
        shutil.copy2(table, output_for(folder / "site_A/20240214_101201.wav"))
    return folder


def main() -> None:
    args = parse_args()
    app = QApplication([])
    from viewer.app import BATCH, SPECTROGRAM, run
    from viewer.session import DETECTIONS
    from viewer.transport import SPANS

    viewer = run(app, [])
    viewer.resize(*WINDOW)
    shot = Shooter(viewer)
    if args.model is not None:
        viewer.picker.select(args.model.resolve())
    wait(300)

    # 1. Ventana vacía: bienvenida, y el menú File abierto.
    shot.save("welcome", shot.window())
    bar = viewer.menuBar()
    assert bar is not None
    menus = [action.menu() for action in bar.actions()]
    file_menu, view_menu = menus[0], menus[1]
    assert file_menu is not None and view_menu is not None
    shot.with_menu("menu_file", file_menu, QPoint(2, bar.height()))

    # 2. Grabación abierta con su tabla de detecciones y las anotaciones encima.
    viewer.open_path(args.audio.resolve())
    shot.settle()
    annotations = args.audio.with_suffix(".txt")
    if annotations.is_file():
        viewer.open_path(annotations.resolve())
        shot.settle()
    shot.engine()
    shot.settle(300)
    # Con detecciones frescas del modelo, que es el flujo del manual.
    viewer.session.set_table(DETECTIONS, None)
    viewer.run_model()
    shot.settle(300)
    # Cinco segundos de ventana y la banda entera: las cajas se leen.
    viewer.band.full()
    viewer.transport.spans.setCurrentIndex(SPANS.index(5.0))
    viewer.transport.center(3.5)
    shot.settle(300)
    full = shot.window()
    shot.save("spectrogram", full)
    toolbar = viewer.open_button.parentWidget()
    assert toolbar is not None
    shot.crop("toolbar", full, toolbar)
    shot.crop("transport", full, viewer.controls)
    shot.crop("context", full, viewer.context)
    # Bajo su título en la barra de menús, como cuando se despliega con el ratón.
    view_x = bar.actionGeometry(bar.actions()[1]).x()
    shot.with_menu("menu_view", view_menu, QPoint(view_x, bar.height()))

    # 3. Panel de cajas abierto.
    viewer.table.show()
    wait(200)
    viewer.table.tabs.setCurrentIndex(1)
    wait(200)
    with_boxes = shot.window()
    shot.save("boxes", with_boxes)
    shot.crop("boxes_panel", with_boxes, viewer.table)
    viewer.table.hide()

    # 4. Settings desplegado, con la lista de salidas abierta.
    viewer.settings.reload_outputs()
    shot.save("settings", viewer.settings.grab())

    # 5. Revisión: la primera detección encuadrada, con asas y su fila de mandos.
    viewer.reviewer.start()
    shot.settle(400)
    review = shot.window()
    shot.save("review", review)
    shot.crop("review_bar", review, viewer.review_bar)
    viewer.reviewer.stop()
    shot.settle()

    # 6. Batch: la carpeta listada y, después, la corrida terminada.
    folder = batch_folder(args.audio)
    viewer.set_mode(BATCH)
    viewer.batch.set_folder(folder)
    while viewer.batch.scanning():
        wait(50)
    wait(300)
    listed = shot.window()
    shot.save("batch", listed)
    shot.crop("batch_controls", listed, viewer.batch.folder, viewer.batch.stop_button)
    viewer.batch.run()
    while viewer.batch.running():
        wait(100)
    wait(300)
    shot.save("batch_done", shot.window())
    viewer.set_mode(SPECTROGRAM)

    shutil.rmtree(folder.parent, ignore_errors=True)
    viewer.close()


if __name__ == "__main__":
    main()
