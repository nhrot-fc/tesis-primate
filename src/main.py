import logging
import sys

from core.config import PROJECT_DIR
from core.runtime import setup_logging
from viewer.app import main

logger = logging.getLogger("visor")


# Con el hook de fábrica PyQt aborta el proceso ante una excepción en un slot, y bajo
# `pythonw.exe` (paquete de Windows) lo haría sin decir nada. Con uno propio la registra y sigue.
def log_uncaught(kind, value, traceback) -> None:
    logger.critical("Error no controlado", exc_info=(kind, value, traceback))


if __name__ == "__main__":
    setup_logging(log_file=PROJECT_DIR / "visor.log")
    sys.excepthook = log_uncaught
    main()
