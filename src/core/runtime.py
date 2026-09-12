import logging
import random
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from core.config import SEED

if TYPE_CHECKING:
    from tqdm.auto import tqdm


def setup_logging(log_file: Path | None = None) -> None:
    # Bajo `pythonw.exe` (paquete de Windows) no hay consola: queda sólo el archivo.
    handlers: list[logging.Handler] = [] if sys.stderr is None else [logging.StreamHandler()]
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
    import resource  # sólo Unix: el entrenamiento no corre en el paquete de Windows

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


def progress(loader: Iterable[Any], desc: str) -> "tqdm":
    # Barra por lote de entrenamiento y evaluación; `disable=None` la apaga sin TTY.
    from tqdm.auto import tqdm  # grupo `train`

    return tqdm(loader, desc=desc, unit="batch", leave=False, disable=None)
