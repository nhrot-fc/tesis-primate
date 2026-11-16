import logging
import random
import sys
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

LOG_BACKUPS = 2

if TYPE_CHECKING:
    from tqdm.auto import tqdm


def setup_logging(log_file: Path | None = None, max_bytes: int | None = None) -> None:
    handlers: list[logging.Handler] = [] if sys.stderr is None else [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_file, mode="a", encoding="utf-8")
            if max_bytes is None
            else RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=LOG_BACKUPS, encoding="utf-8"
            )
        )
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def log_location() -> Path | None:
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def share_tensors_by_file() -> None:
    import resource

    import torch

    torch.multiprocessing.set_sharing_strategy("file_system")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str | None = None) -> str:
    import torch

    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)
    return device


def progress(loader: Iterable[Any], desc: str) -> "tqdm":
    # disable=None: sin TTY no hay barra
    from tqdm.auto import tqdm

    return tqdm(loader, desc=desc, unit="batch", leave=False, disable=None)
