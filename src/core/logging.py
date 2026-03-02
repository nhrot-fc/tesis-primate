import json
import logging
from pathlib import Path

from src.core.config import settings

LOGGER_NAME = "tesis_primate"
DEFAULT_LOG_PATH = Path("logs") / "app.log"


def _log_event(level: int, event: str, **fields: object) -> None:
    """Función privada que centraliza la escritura del JSON."""
    payload = {"event": event, **fields}
    logger = logging.getLogger(LOGGER_NAME)
    logger.log(level, json.dumps(payload))


def log_debug(event: str, **fields: object) -> None:
    _log_event(logging.DEBUG, event, **fields)


def log_info(event: str, **fields: object) -> None:
    _log_event(logging.INFO, event, **fields)


def log_warning(event: str, **fields: object) -> None:
    _log_event(logging.WARNING, event, **fields)


def log_error(event: str, **fields: object) -> None:
    _log_event(logging.ERROR, event, **fields)


def setup_logging(log_path: Path | str = DEFAULT_LOG_PATH) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,  # Importante: sobreescribe cualquier config previa
    )
