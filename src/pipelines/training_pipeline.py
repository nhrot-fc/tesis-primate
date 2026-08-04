import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from pipelines.common import LOSS_KEYS, Losses, Target, to_device


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Target]],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    device: torch.device | str = "cpu",
    clip_grad: float = 0.1,
    desc: str = "train",
) -> Losses:
    model.train()
    totals = dict.fromkeys(LOSS_KEYS, 0.0)

    progress = tqdm(loader, desc=desc, unit="batch", leave=False)
    for step, batch in enumerate(progress, start=1):
        images, targets = to_device(batch, device)
        losses: dict[str, Tensor] = criterion(model(images), targets)

        optimizer.zero_grad()
        losses["loss_total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        for key in LOSS_KEYS:
            totals[key] += losses[f"loss_{key}"].item()
        progress.set_postfix(loss=totals["total"] / step, lr=optimizer.param_groups[0]["lr"])

    return Losses(**{key: value / max(len(loader), 1) for key, value in totals.items()})
