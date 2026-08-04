import logging
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)


def setup_project_path(project_dir: Path) -> None:
    settings.PROJECT_DIR = project_dir
    logger.info("Ruta del proyecto configurada en: %s", settings.PROJECT_DIR)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Consola siempre; si se pasa `log_file`, además una copia completa en disco."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    if log_file is not None:
        logging.getLogger(__name__).info("Log de esta corrida en %s", log_file)
