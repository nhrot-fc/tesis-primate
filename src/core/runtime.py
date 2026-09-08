import logging
import random
import resource
from pathlib import Path

import numpy as np
import torch

from core.config import SEED, settings


def setup_logging(level: str | None = None, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=level or settings.LOG_LEVEL,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def share_tensors_by_file() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:  # el límite blando suele venir en 1024 aunque el duro sea mucho mayor
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
