from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from core.runtime import share_tensors_by_file
from data import cache
from data.augment import AugmentConfig, Augmenter
from utils.audio import mel_to_gray
from utils.boxes import Target

Batch = tuple[Tensor, list[Target]]

# En una caja cxcywh, x es tiempo y w duración; y es frecuencia y h ancho de banda.
TIME, FREQ = 0, 1


@dataclass(frozen=True)
class BoxJitter:
    time_scale: float = 0.15  # ±% de la duración
    freq_scale: float = 0.15  # ±% del ancho de banda
    time_shift: float = 0.10  # corrimiento, en fracción de la duración
    freq_shift: float = 0.10  # corrimiento, en fracción del ancho de banda
    min_size: float = 0.02  # duración/ancho de banda mínimos, en fracción del clip


def jitter_boxes(boxes: Tensor, jitter: BoxJitter) -> Tensor:
    if not len(boxes):
        return boxes

    centers, sizes = boxes[:, :2], boxes[:, 2:]

    scale = torch.empty_like(sizes)
    scale[:, TIME].uniform_(1 - jitter.time_scale, 1 + jitter.time_scale)
    scale[:, FREQ].uniform_(1 - jitter.freq_scale, 1 + jitter.freq_scale)
    sizes = (sizes * scale).clamp(min=jitter.min_size, max=1.0)

    shift = torch.empty_like(centers)
    shift[:, TIME].uniform_(-jitter.time_shift, jitter.time_shift)
    shift[:, FREQ].uniform_(-jitter.freq_shift, jitter.freq_shift)
    centers = centers + shift * sizes  # el corrimiento es relativo al tamaño de cada eje

    low = (centers - sizes / 2).clamp(0.0, 1.0)
    high = (centers + sizes / 2).clamp(0.0, 1.0)

    # La caja que ya tocaba un borde lo sigue tocando: la llamada se cortó ahí, y moverla
    # hacia adentro inventaría un principio, un final o una banda que no se grabaron.
    original_low = boxes[:, :2] - boxes[:, 2:] / 2
    original_high = boxes[:, :2] + boxes[:, 2:] / 2
    low = torch.where(original_low <= 0.0, torch.zeros_like(low), low)
    high = torch.where(original_high >= 1.0, torch.ones_like(high), high)
    return torch.cat([(low + high) / 2, high - low], dim=1)


class WindowCache(Dataset):
    def __init__(
        self,
        path: Path,
        jitter: BoxJitter | None = None,
        augment: AugmentConfig | None = None,
    ):
        stored = torch.load(path, weights_only=False)
        self.images: Tensor = stored["images"]
        self.boxes: list[Tensor] = stored["boxes"]
        self.labels: list[Tensor] = stored["labels"]
        self.jitter = jitter  # sólo en train: validar contra cajas perturbadas no sirve
        self.augmenter = None
        if augment is not None:
            clips = cache.clips(path.stem, len(self.boxes))
            self.augmenter = Augmenter(self.images, self.boxes, self.labels, clips, augment)

    def __len__(self) -> int:
        return len(self.images)

    def window(self, index: int) -> tuple[Tensor, Target]:
        if self.augmenter is None:
            image = self.images[index]
            target: Target = {"boxes": self.boxes[index], "labels": self.labels[index]}
        else:
            image, target = self.augmenter(index)
        if self.jitter is not None:
            target["boxes"] = jitter_boxes(target["boxes"], self.jitter)
        return image, {"boxes": target["boxes"].clone(), "labels": target["labels"].clone()}


class SpectrogramDataset(WindowCache):
    def __getitem__(self, index: int) -> tuple[Tensor, Target]:
        return self.window(index)


class YOLODataset(WindowCache):
    def __init__(self, path: Path, db_low: float, db_high: float, image_size: int = 512):
        super().__init__(path)
        self.db_low, self.db_high, self.image_size = db_low, db_high, image_size

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[str]]:
        mel, target = self.window(index)
        image = mel_to_gray(mel[0], self.db_low, self.db_high, self.image_size)
        lines = []
        for (cx, cy, w, h), class_id in zip(
            target["boxes"].tolist(), target["labels"].tolist(), strict=True
        ):
            # la imagen va con el grave abajo, así que la frecuencia se invierte con ella
            values = [min(max(v, 0.0), 1.0) for v in (cx, 1.0 - cy, w, h)]
            if values[2] > 0.0 and values[3] > 0.0:
                lines.append(f"{int(class_id)} " + " ".join(f"{v:.6f}" for v in values))
        return image, lines


def collate_fn(batch: list[tuple[Tensor, Target]]) -> Batch:
    images = torch.stack([image for image, _ in batch], dim=0)  # (B, 1, n_mels, T)
    return images, [target for _, target in batch]


def make_loader(
    dataset: Dataset, batch_size: int, workers: int = 0, shuffle: bool = False
) -> DataLoader:
    if workers > 0:
        share_tensors_by_file()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def to_device(batch: Batch, device: torch.device | str) -> Batch:
    images, targets = batch
    return (
        images.to(device),
        [{key: value.to(device) for key, value in target.items()} for target in targets],
    )
