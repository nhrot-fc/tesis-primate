from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from core.config import P, Parameters
from domain.species import LabelSet
from utils.audio import MelSpectrogram, hz_to_y, load_clip, window_starts

FloatArray = npt.NDArray[np.float64]

BOX_COORDINATES_SLICE = slice(0, 4)  # cx, cy, w, h
LABEL_INDEX = 4  # id de clase en el `LabelSet`, en 0..N-1
N_BOX_COLS = 5
MIN_BOX_SIZE = 1e-3


class ClipWindow(NamedTuple):
    audio_path: str
    clip_start_s: float
    duration_s: float  # longitud de la ventana; `clip_start_s + duration_s` la cierra
    boxes: FloatArray  # (N, 5): cxcywh normalizado + id de clase


def _boxes_in_window(
    group: pd.DataFrame, class_ids: FloatArray, clip_start_s: float, params: Parameters
) -> tuple[FloatArray, bool]:
    begin = group["begin_time_s"].to_numpy()
    end = group["end_time_s"].to_numpy()

    overlap = np.minimum(end, clip_start_s + params.clip_len_s) - np.maximum(begin, clip_start_s)
    visible = np.minimum(end - begin, params.clip_len_s)
    present = overlap > 0
    keep = present & (overlap >= params.min_overlap * visible)
    if not keep.any():
        return np.empty((0, N_BOX_COLS)), bool(present.any())

    x0 = np.clip((begin[keep] - clip_start_s) / params.clip_len_s, 0.0, 1.0)
    x1 = np.clip((end[keep] - clip_start_s) / params.clip_len_s, 0.0, 1.0)
    y0 = np.clip(hz_to_y(group["low_freq_hz"].to_numpy()[keep], params), 0.0, 1.0)
    y1 = np.clip(hz_to_y(group["high_freq_hz"].to_numpy()[keep], params), 0.0, 1.0)

    boxes = np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, class_ids[keep]], axis=-1)
    usable = (boxes[:, 2] >= MIN_BOX_SIZE) & (boxes[:, 3] >= MIN_BOX_SIZE)
    boxes = boxes[usable]
    return boxes, len(boxes) < int(present.sum())


def build_manifest(
    df: pd.DataFrame,
    labels: LabelSet,
    params: Parameters = P,
    empty_ratio: float = 0.0,
    seed: int = 42,
) -> list[ClipWindow]:
    if not 0.0 <= empty_ratio < 1.0:
        raise ValueError(f"empty_ratio debe estar en [0, 1): {empty_ratio}")

    unknown = {name for name in df["label"].unique() if name not in labels}
    if unknown:
        raise ValueError(f"etiquetas fuera del LabelSet: {sorted(unknown)}")

    positive: list[ClipWindow] = []
    empty: list[ClipWindow] = []
    for audio_path, group in df.groupby("audio_path"):
        try:
            duration_s = sf.info(str(audio_path)).duration
        except (RuntimeError, sf.LibsndfileError):
            continue

        class_ids = group["label"].map(labels.id).to_numpy(dtype=np.float64)
        for clip_start_s in window_starts(duration_s, params):
            boxes, incomplete = _boxes_in_window(group, class_ids, float(clip_start_s), params)
            window = ClipWindow(str(audio_path), float(clip_start_s), params.clip_len_s, boxes)
            if len(boxes):
                positive.append(window)
            elif not incomplete:
                empty.append(window)

    if empty_ratio <= 0.0 or not empty:
        return positive

    n_empty = min(len(empty), round(len(positive) * empty_ratio / (1 - empty_ratio)))
    keep_idx = np.random.default_rng(seed).choice(len(empty), size=n_empty, replace=False)
    return positive + [empty[i] for i in keep_idx]


def split_manifest(
    manifest: list[ClipWindow],
    n_classes: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[ClipWindow], list[ClipWindow], list[ClipWindow]]:
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"los ratios deben sumar 1.0: {ratios} suma {sum(ratios)}")

    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_classes))
    windows_by_file: dict[str, list[ClipWindow]] = defaultdict(list)
    for window in manifest:
        windows_by_file[window.audio_path].append(window)
        for class_id in window.boxes[:, LABEL_INDEX].astype(int):
            counts[window.audio_path][class_id] += 1

    files = sorted(windows_by_file)
    np.random.default_rng(seed).shuffle(files)

    total = np.sum([counts[f] for f in files], axis=0)  # (n_classes,)
    target = np.outer(ratios, total)  # (3, n_classes)
    current = np.zeros((3, n_classes))
    n_windows = np.zeros(3)
    target_windows = np.array(ratios) * len(manifest)

    assigned: dict[str, int] = {}

    # clases de la más rara a la más común
    for class_id in np.argsort(total):
        pending = [f for f in files if f not in assigned and counts[f][class_id] > 0]
        # los archivos con más cajas de esta clase se colocan primero: son los que
        # más mueven la aguja y conviene decidirlos con el máximo de libertad
        pending.sort(key=lambda f: -counts[f][class_id])

        for file in pending:
            deficit = target[:, class_id] - current[:, class_id]
            best = np.flatnonzero(deficit == deficit.max())
            best = best[np.argmax((target_windows - n_windows)[best])] if len(best) > 1 else best[0]

            assigned[file] = int(best)
            current[best] += counts[file]
            n_windows[best] += len(windows_by_file[file])

    for file in files:
        if file not in assigned:
            best = int(np.argmax(target_windows - n_windows))
            assigned[file] = best
            n_windows[best] += len(windows_by_file[file])

    splits: list[list[ClipWindow]] = [[], [], []]
    for file, split_index in assigned.items():
        splits[split_index].extend(windows_by_file[file])
    return splits[0], splits[1], splits[2]


class CallBoxDataset(Dataset):
    def __init__(self, manifest: list[ClipWindow], params: Parameters = P):
        self.manifest = manifest
        self.params = params
        self.mel_spectrogram = MelSpectrogram(params)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        window = self.manifest[index]

        waveform = load_clip(window.audio_path, window.clip_start_s, self.params)
        mel = self.mel_spectrogram(waveform).unsqueeze(0)

        return mel, {
            "boxes": torch.from_numpy(window.boxes[:, BOX_COORDINATES_SLICE].astype(np.float32)),
            "labels": torch.from_numpy(window.boxes[:, LABEL_INDEX].astype(np.int64)),
        }


@dataclass(frozen=True)
class BoxJitter:
    scale: float = 0.15  # ±15% de ancho y alto
    shift: float = 0.10  # corrimiento en el eje temporal, en fracción del ancho de la caja
    min_size: float = 0.02  # ancho/alto mínimo tras el jitter, en fracción del clip


def _uniform(shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
    return torch.empty(shape).uniform_(low, high)


def jitter_boxes(boxes: torch.Tensor, jitter: BoxJitter) -> torch.Tensor:
    if not len(boxes):
        return boxes

    centers, sizes = boxes[:, :2], boxes[:, 2:]
    sizes = (sizes * _uniform(sizes.shape, 1 - jitter.scale, 1 + jitter.scale)).clamp(
        min=jitter.min_size, max=1.0
    )
    shift = _uniform((len(boxes), 1), -jitter.shift, jitter.shift) * sizes[:, :1]
    centers = centers + torch.cat([shift, torch.zeros_like(shift)], dim=1)

    low = (centers - sizes / 2).clamp(0.0, 1.0)
    high = (centers + sizes / 2).clamp(0.0, 1.0)

    edges = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
    low = torch.where(edges[0] <= 0.0, torch.zeros_like(low), low)
    high = torch.where(edges[1] >= 1.0, torch.ones_like(high), high)
    return torch.cat([(low + high) / 2, high - low], dim=1)


class CachedCallBoxDataset(Dataset):
    def __init__(self, path: Path, jitter: BoxJitter | None = None):
        cache = torch.load(path, weights_only=False)
        self.images: torch.Tensor = cache["images"]
        self.boxes: list[torch.Tensor] = cache["boxes"]
        self.labels: list[torch.Tensor] = cache["labels"]
        self.jitter = jitter  # sólo en train: validar contra cajas perturbadas no sirve

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boxes = self.boxes[index]
        if self.jitter is not None:
            boxes = jitter_boxes(boxes, self.jitter)
        return self.images[index], {"boxes": boxes, "labels": self.labels[index]}


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = torch.stack([image for image, _ in batch], dim=0)  # (B, 1, n_mels, T)
    return images, [target for _, target in batch]
