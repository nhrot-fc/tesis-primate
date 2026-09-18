"""Un split del caché en memoria: espectrogramas, cajas, de qué grabación viene cada ventana y
cómo elegir ventanas de ejemplo de una clase."""

from collections.abc import Collection
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert

from data import cache
from data.species import LabelSet

# Una anotación de ejemplo tiene que estar entera en la ventana (lejos de los bordes)
MARGIN = 0.02


class Split:
    def __init__(self, name: str) -> None:
        stored = torch.load(
            cache.split_path(name), map_location="cpu", weights_only=False, mmap=True
        )
        self.name = name
        self.images: Tensor = stored["images"]  # (N, 1, n_mels, n_frames), mel en potencia
        self.boxes: list[Tensor] = stored["boxes"]  # por ventana, (K, 4) cxcywh en [0, 1]
        self.labels: list[Tensor] = stored["labels"]  # por ventana, (K,)
        self.sources = cache.sources(name, len(self.images))
        self.label_set: LabelSet = cache.labels()
        self.db_range = cache.db_range()

    def __len__(self) -> int:
        return len(self.images)

    @property
    def names(self) -> list[str]:
        return self.label_set.names

    def recording_of(self, image_id: int) -> str:
        return Path(self.sources.recordings[self.sources.recording_of_window[image_id]]).stem

    def start_of(self, image_id: int) -> float:
        return self.sources.clip_start_s[image_id]

    @cached_property
    def all_boxes(self) -> tuple[Tensor, Tensor]:
        # Todas las cajas del split apiladas, con su clase
        return torch.cat(self.boxes), torch.cat(self.labels)

    def boxes_of(self, class_id: int) -> Tensor:
        boxes, labels = self.all_boxes
        return boxes[labels == class_id]

    @cached_property
    def whole_box_windows(self) -> dict[int, list[int]]:
        # Por clase, las ventanas (en orden) con una anotación suya entera, lejos de los bordes
        boxes, labels = self.all_boxes
        image_ids = torch.repeat_interleave(
            torch.arange(len(self)), torch.tensor([len(b) for b in self.boxes])
        )
        xyxy = box_convert(boxes, "cxcywh", "xyxy")
        inside = (xyxy[:, 0] > MARGIN) & (xyxy[:, 2] < 1 - MARGIN)
        found: dict[int, list[int]] = {c: [] for c in range(len(self.label_set))}
        pairs = zip(image_ids[inside].tolist(), labels[inside].tolist(), strict=True)
        for image_id, label in sorted(set(pairs)):
            found[label].append(image_id)
        return found

    def examples(
        self, class_id: int, n: int, seed: int, exclude: Collection[str] = ()
    ) -> list[int]:
        # `n` ventanas con una anotación entera de la clase, al azar: de grabaciones distintas
        # mientras alcancen (nunca de las de `exclude`), y después las que queden
        order = np.random.default_rng(seed).permutation(self.whole_box_windows[class_id])
        candidates = [i for i in order.tolist() if self.recording_of(i) not in exclude]
        distinct: list[int] = []
        seen: set[str] = set()
        for image_id in candidates:
            if self.recording_of(image_id) not in seen:
                distinct.append(image_id)
                seen.add(self.recording_of(image_id))
        return (distinct + [i for i in candidates if i not in distinct])[:n]
