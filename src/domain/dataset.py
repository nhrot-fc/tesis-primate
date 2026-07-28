from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from core.config import P, Parameters
from domain.species import LabelSet
from utils.audio import LogMelSpectrogram, load_clip

FloatArray = npt.NDArray[np.float64]

BOX_COORDINATES_SLICE = slice(0, 4)  # cx, cy, w, h
LABEL_INDEX = 4  # id de clase en el `LabelSet`, en 0..N-1
N_BOX_COLS = 5


class ClipWindow(NamedTuple):
    audio_path: str
    clip_start_s: float
    duration_s: float  # longitud de la ventana; `clip_start_s + duration_s` la cierra
    boxes: FloatArray  # (N, 5): cxcywh normalizado + id de clase


def _hz_to_y(freq_hz: FloatArray, params: Parameters) -> FloatArray:
    """Hz -> posición vertical normalizada en el eje mel HTK (y=0 = graves)."""

    def mel(hz):
        return params.mel_scale_q * np.log10(
            1.0 + np.asarray(hz, dtype=np.float64) / params.mel_break_hz
        )

    mel_lo, mel_hi = mel(params.f_min), mel(params.f_max)
    return (mel(freq_hz) - mel_lo) / (mel_hi - mel_lo)


def _window_starts(duration_s: float, params: Parameters) -> FloatArray:
    last_start = max(duration_s - params.clip_len_s, 0.0)
    if last_start == 0.0:
        return np.zeros(1)
    return np.arange(0.0, last_start + params.clip_hop_s / 2, params.clip_hop_s)


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
    y0 = np.clip(_hz_to_y(group["low_freq_hz"].to_numpy()[keep], params), 0.0, 1.0)
    y1 = np.clip(_hz_to_y(group["high_freq_hz"].to_numpy()[keep], params), 0.0, 1.0)

    return np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, class_ids[keep]], axis=-1)


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
        for clip_start_s in _window_starts(duration_s, params):
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


def collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    images = torch.stack([image for image, _ in batch], dim=0)  # (B, 1, n_mels, T)
    return images, [target for _, target in batch]
