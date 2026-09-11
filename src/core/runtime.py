import logging
import random
import resource
from pathlib import Path

import numpy as np
import torch

from core.config import SEED


def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def share_tensors_by_file() -> None:
    # Compartir el caché por archivo pide más descriptores que los 1024 de fábrica.
    torch.multiprocessing.set_sharing_strategy("file_system")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str | None = None) -> str:
    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device).index or 0)
    return device
