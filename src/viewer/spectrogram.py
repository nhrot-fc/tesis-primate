from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor

EPS = 1e-10


def load_audio(path: Path, target_sr: int) -> Tensor:
    waveform, source_sr = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0)
    if source_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sr)
    return waveform


def pcm16(waveform: Tensor) -> bytes:
    """Serializa la forma de onda mono como PCM entero de 16 bits."""
    return (waveform.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()


def stft_db(waveform: Tensor, n_fft: int, hop_length: int) -> np.ndarray:
    if waveform.numel() < n_fft:
        waveform = F.pad(waveform, (0, n_fft - waveform.numel()))
    spec = (
        torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            return_complex=True,
        )
        .abs()
        .square()
    )
    return (10 * torch.log10(spec + EPS)).numpy()


def db_baseline(waveform: Tensor, sr: int, n_fft: int) -> tuple[float, float]:
    spec = stft_db(waveform, n_fft, max(sr // 10, waveform.numel() // 3000, 1))
    lo, hi = np.percentile(spec, (5.0, 99.5))
    return float(lo), float(hi)


def db_levels(
    baseline: tuple[float, float], brightness: float, contrast: float
) -> tuple[float, float]:
    lo, hi = baseline
    center = 0.5 * (lo + hi) - brightness
    half_range = 0.5 * (hi - lo) / max(contrast, 1e-3)
    return center - half_range, center + half_range
