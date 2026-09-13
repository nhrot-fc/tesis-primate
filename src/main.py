import logging
import os
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from core.config import PROJECT_DIR
from core.runtime import setup_logging

logger = logging.getLogger("viewer")

SPLASH_SIZE = (440, 200)
SPLASH_BACKGROUND, SPLASH_TEXT, SPLASH_MUTED = "#1c1b22", "#f4f2f8", "#9a97a6"
TITLE = "Primate Vocalization Detector"


# Con el hook de fábrica PyQt aborta el proceso ante una excepción en un slot, y bajo
# `pythonw.exe` (paquete de Windows) lo haría sin decir nada. Con uno propio la registra y sigue.
def log_uncaught(kind, value, traceback) -> None:
    logger.critical("Unhandled error", exc_info=(kind, value, traceback))


# Un cartel pintado a mano: no depende de ningún archivo y sale antes que cualquier import
# pesado. Es lo único que se ve mientras cargan pandas, pyqtgraph y el audio.
def splash_pixmap() -> QPixmap:
    width, height = SPLASH_SIZE
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(SPLASH_BACKGROUND))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    title = QFont(painter.font())
    title.setPointSize(16)
    title.setBold(True)
    painter.setFont(title)
    painter.setPen(QColor(SPLASH_TEXT))
    painter.drawText(pixmap.rect().adjusted(28, 40, -28, -40), Qt.AlignmentFlag.AlignLeft, TITLE)
    painter.end()
    return pixmap


def main() -> None:
    setup_logging(log_file=PROJECT_DIR / "viewer.log")
    sys.excepthook = log_uncaught
    if sys.platform == "linux":
        os.environ.setdefault("QT_QPA_PLATFORMTHEME", "xdgdesktopportal")
        os.environ.setdefault("QT_WAYLAND_DECORATION", "adwaita")
    app = QApplication(sys.argv)
    splash = QSplashScreen(splash_pixmap())
    splash.showMessage(
        "   Loading…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
        QColor(SPLASH_MUTED),
    )
    splash.show()
    app.processEvents()

    # Lo que tarda: pandas, pyqtgraph, soundfile. torch viene después, en segundo plano.
    from viewer.app import run

    viewer = run(app, [Path(argument) for argument in sys.argv[1:]])
    splash.finish(viewer)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
