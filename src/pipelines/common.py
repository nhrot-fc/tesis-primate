from typing import NamedTuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

Target = dict[str, Tensor]
Batch = tuple[Tensor, list[Target]]

LOSS_KEYS: tuple[str, ...] = ("total", "cls", "bbox", "iou")


class Losses(NamedTuple):
    total: float
    cls: float
    bbox: float
    iou: float


def to_device(batch: Batch, device: torch.device | str) -> Batch:
    images, targets = batch
    return (
        images.to(device),
        [{key: value.to(device) for key, value in target.items()} for target in targets],
    )


def format_metric(value: float | None, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def make_loader(
    dataset: Dataset, batch_size: int, workers: int = 0, shuffle: bool = False
) -> DataLoader:
    from domain.dataset import collate_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
