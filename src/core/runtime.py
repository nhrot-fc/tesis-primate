import functools
import logging
import random
import sys
from collections.abc import Iterable
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.config import BATCH_SIZE
from core.errors import DeviceError

LOG_BACKUPS = 2
CPU_BATCH_SIZE = 4
# (GiB de VRAM mínimos, ventanas por lote): Faster R-CNN usa ~0.24 GiB por ventana
GPU_BATCH_SIZES = ((5.5, BATCH_SIZE), (3.5, 8))

logger = logging.getLogger(__name__)

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


@functools.cache
def cuda_problem(index: int = 0) -> str | None:
    # None si la GPU sirve. Sin kernels para la tarjeta (GTX 1050 con cu128) torch la ve
    # igual: sólo una operación lo revela.
    import torch

    if torch.version.cuda is None:  # builds de CPU y de ROCm
        return "this torch has no CUDA support"
    if not torch.cuda.is_available():
        return "no NVIDIA GPU or driver was found"
    if index >= torch.cuda.device_count():
        return f"there is no GPU number {index}"
    device = torch.device("cuda", index)
    try:
        x = torch.ones(1, 1, 8, 8, device=device)  # conv2d: kernels de torch y cuDNN
        torch.nn.functional.conv2d(x, torch.ones(1, 1, 3, 3, device=device)).sum().item()
    except Exception as exc:
        major, minor = torch.cuda.get_device_capability(index)
        return (
            f"{torch.cuda.get_device_name(index)} (compute capability {major}.{minor}) failed "
            f"with this CUDA {torch.version.cuda} build of torch, made for "
            f"{' '.join(torch.cuda.get_arch_list())}: {type(exc).__name__}: {exc}"
        )
    return None


def resolve_device(requested: str | None = None) -> str:
    # None: la GPU si sirve, si no la CPU. Lo pedido con --device tiene que servir.
    import torch

    if requested is None:
        problem = cuda_problem()
        if problem is not None:
            # avisa sólo si hay una GPU que no corre
            log = logger.warning if torch.cuda.is_available() else logger.info
            log("Using the CPU: %s", problem)
            return "cpu"
        requested = "cuda"
    try:
        device = torch.device(requested)
    except RuntimeError as exc:
        raise DeviceError(f"Unknown device '{requested}': use cpu or cuda.") from exc
    if device.type == "cpu":
        return requested
    if device.type != "cuda":
        raise DeviceError(f"Device '{requested}' is not supported: use cpu or cuda.")
    index = device.index or 0
    if (problem := cuda_problem(index)) is not None:
        raise DeviceError(f"Cannot use '{requested}': {problem}.")
    torch.cuda.set_device(index)
    logger.info("Using the GPU: %s", torch.cuda.get_device_name(index))
    return requested


def batch_size_for(device: str) -> int:
    if not device.startswith("cuda"):
        return CPU_BATCH_SIZE
    import torch

    vram_gib = torch.cuda.get_device_properties(torch.device(device)).total_memory / 2**30
    return next((size for least, size in GPU_BATCH_SIZES if vram_gib >= least), CPU_BATCH_SIZE)


def progress(loader: Iterable[Any], desc: str) -> "tqdm":
    # disable=None: sin TTY no hay barra
    from tqdm.auto import tqdm

    return tqdm(loader, desc=desc, unit="batch", leave=False, disable=None)
