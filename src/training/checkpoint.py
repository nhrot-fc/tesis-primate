import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from data.species import LabelSet
from models.registry import cpu_state_dict

logger = logging.getLogger(__name__)

BEST = "best.pt"
LAST = "last.pt"


def save(
    path: Path,
    architecture: str,
    model: nn.Module,
    hparams: dict[str, Any],
    labels: LabelSet,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, Any],
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    best_score: float = float("-inf"),
) -> None:
    # Sin optimizador queda un checkpoint de inferencia; con él, uno que `resume` retoma.
    checkpoint: dict[str, Any] = {
        "architecture": architecture,
        "hparams": hparams,
        "labels": labels.names,
        "config": config,
        "state_dict": cpu_state_dict(model),
        "epoch": epoch,
        "metrics": metrics,
    }
    if optimizer is not None and scheduler is not None:
        checkpoint |= {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
            "rng": torch.get_rng_state(),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def resume(
    run_dir: Path, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler
) -> tuple[int, float]:
    # -> (próxima época a correr, mejor score). Sin `last.pt`, empieza en 0.
    path = run_dir / LAST
    if not path.is_file():
        return 0, float("-inf")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["rng"])

    epoch, best_score = checkpoint["epoch"] + 1, checkpoint["best_score"]
    logger.info("retomando %s en la época %d (mejor score %.3f)", path, epoch + 1, best_score)
    return epoch, best_score
