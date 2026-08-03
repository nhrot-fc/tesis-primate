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
from utils.audio import LogMelSpectrogram, hz_to_y, load_clip, window_starts

FloatArray = npt.NDArray[np.float64]

BOX_COORDINATES_SLICE = slice(0, 4)  # cx, cy, w, h
LABEL_INDEX = 4  # id de clase en el `LabelSet`, en 0..N-1
N_BOX_COLS = 5
MIN_BOX_SIZE = 1e-3  # ancho/alto normalizado mínimo para que una caja sea utilizable


class ClipWindow(NamedTuple):
    audio_path: str
    clip_start_s: float
    duration_s: float  # longitud de la ventana; `clip_start_s + duration_s` la cierra
    boxes: FloatArray  # (N, 5): cxcywh normalizado + id de clase


def _boxes_in_window(
    group: pd.DataFrame, class_ids: FloatArray, clip_start_s: float, params: Parameters
) -> FloatArray:
    """Cajas normalizadas de los eventos que solapan lo suficiente con la ventana."""
    begin = group["begin_time_s"].to_numpy()
    end = group["end_time_s"].to_numpy()

    overlap = np.minimum(end, clip_start_s + params.clip_len_s) - np.maximum(begin, clip_start_s)
    keep = (overlap > 0) & (overlap >= params.min_overlap * (end - begin))
    if not keep.any():
        return np.empty((0, N_BOX_COLS))

    x0 = np.clip((begin[keep] - clip_start_s) / params.clip_len_s, 0.0, 1.0)
    x1 = np.clip((end[keep] - clip_start_s) / params.clip_len_s, 0.0, 1.0)
    y0 = np.clip(hz_to_y(group["low_freq_hz"].to_numpy()[keep], params), 0.0, 1.0)
    y1 = np.clip(hz_to_y(group["high_freq_hz"].to_numpy()[keep], params), 0.0, 1.0)

    boxes = np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, class_ids[keep]], axis=-1)
    # Una anotación enteramente por encima de `f_max` (o al ras del borde del clip) se
    # recorta a una caja de área cero: IoU 0 para siempre y un target de tamaño 0 que
    # el modelo no puede alcanzar. Mejor descartarla que entrenar contra ella.
    usable = (boxes[:, 2] >= MIN_BOX_SIZE) & (boxes[:, 3] >= MIN_BOX_SIZE)
    return boxes[usable]


def build_manifest(
    df: pd.DataFrame,
    labels: LabelSet,
    params: Parameters = P,
    keep_empty: bool = False,
) -> list[ClipWindow]:
    unknown = {name for name in df["label"].unique() if name not in labels}
    if unknown:
        raise ValueError(f"etiquetas fuera del LabelSet: {sorted(unknown)}")

    windows: list[ClipWindow] = []
    for audio_path, group in df.groupby("audio_path"):
        try:
            duration_s = sf.info(str(audio_path)).duration
        except (RuntimeError, sf.LibsndfileError):
            continue

        class_ids = group["label"].map(labels.id).to_numpy(dtype=np.float64)
        for clip_start_s in window_starts(duration_s, params):
            boxes = _boxes_in_window(group, class_ids, float(clip_start_s), params)
            if len(boxes) or keep_empty:
                windows.append(
                    ClipWindow(str(audio_path), float(clip_start_s), params.clip_len_s, boxes)
                )

    return windows


def split_manifest(
    manifest: list[ClipWindow],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> tuple[list[ClipWindow], list[ClipWindow], list[ClipWindow]]:
    """Reparto **por grabación**, para que ventanas solapadas no crucen de split."""
    paths = sorted({w.audio_path for w in manifest})
    np.random.default_rng(seed).shuffle(paths)

    n_train = int(ratios[0] * len(paths))
    n_val = int(ratios[1] * len(paths))

    def subset(keep: list[str]) -> list[ClipWindow]:
        return [w for w in manifest if w.audio_path in set(keep)]

    return (
        subset(paths[:n_train]),
        subset(paths[n_train : n_train + n_val]),
        subset(paths[n_train + n_val :]),
    )


class CallBoxDataset(Dataset):
    """Ventana del manifiesto -> (log-mel, {boxes, labels})."""

    def __init__(self, manifest: list[ClipWindow], params: Parameters = P):
        self.manifest = manifest
        self.params = params
        self.log_mel_spectrogram = LogMelSpectrogram(params)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        window = self.manifest[index]

        waveform = load_clip(window.audio_path, window.clip_start_s, self.params)
        log_mel = self.log_mel_spectrogram(waveform).unsqueeze(0)

        return log_mel, {
            "boxes": torch.from_numpy(window.boxes[:, BOX_COORDINATES_SLICE].astype(np.float32)),
            "labels": torch.from_numpy(window.boxes[:, LABEL_INDEX].astype(np.int64)),
        }


@dataclass(frozen=True)
class BoxJitter:
    """Ruido sobre las cajas del target, sin tocar el espectrograma.

    Las anotaciones no son exactas: los bordes de una vocalización son difusos y cada
    anotador los corta distinto. Perturbar la caja en cada `__getitem__` (o sea, un
    jitter distinto por época) evita que el modelo memorice las coordenadas exactas de
    cada ventana y lo empuja a aprender el evento, no la anotación.
    """

    scale: float = 0.15  # ±15% de ancho y alto
    shift: float = 0.10  # corrimiento en el eje temporal, en fracción del ancho de la caja
    min_size: float = 0.02  # ancho/alto mínimo tras el jitter, en fracción del clip


def _uniform(shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
    return torch.empty(shape).uniform_(low, high)


def jitter_boxes(boxes: torch.Tensor, jitter: BoxJitter) -> torch.Tensor:
    """Cajas cxcywh normalizadas -> cajas perturbadas, siempre dentro del clip [0, 1]."""
    if not len(boxes):
        return boxes

    centers, sizes = boxes[:, :2], boxes[:, 2:]
    sizes = (sizes * _uniform(sizes.shape, 1 - jitter.scale, 1 + jitter.scale)).clamp(
        min=jitter.min_size, max=1.0
    )
    shift = _uniform((len(boxes), 1), -jitter.shift, jitter.shift) * sizes[:, :1]
    centers = centers + torch.cat([shift, torch.zeros_like(shift)], dim=1)

    # recorte en xyxy y no en el centro: una llamada que ya venía cortada por el borde
    # del clip se queda pegada al borde en vez de moverse hacia adentro.
    low = (centers - sizes / 2).clamp(0.0, 1.0)
    high = (centers + sizes / 2).clamp(0.0, 1.0)
    return torch.cat([(low + high) / 2, high - low], dim=1)


class CachedCallBoxDataset(Dataset):
    """Split materializado por `create_dataset.py`: tensores ya listos en `path`,
    sin I/O de audio ni cómputo de espectrograma en cada `__getitem__`."""

    def __init__(self, path: Path, jitter: BoxJitter | None = None):
        cache = torch.load(path, weights_only=False)
        self.images: torch.Tensor = cache["images"]
        self.boxes: list[torch.Tensor] = cache["boxes"]
        self.labels: list[torch.Tensor] = cache["labels"]
        self.jitter = jitter  # sólo en train: validar contra cajas perturbadas no sirve

    def use_precomputed(self, features: torch.Tensor) -> None:
        """Sustituye los espectrogramas por features ya calculadas del backbone congelado.

        Todo lo de abajo (collate, jitter, targets) sigue igual: lo único que cambia es
        qué tensor viaja como "imagen" hasta el modelo.
        """
        if len(features) != len(self.images):
            raise ValueError(f"{len(features)} features para {len(self.images)} ventanas")
        self.images = features

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
