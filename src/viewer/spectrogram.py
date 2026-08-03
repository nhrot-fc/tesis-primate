from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile
import soxr

EPS = 1e-10

Waveform = npt.NDArray[np.float32]


def load_audio(path: Path, target_sr: int) -> Waveform:
    frames, source_sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = frames.mean(axis=1)
    if source_sr != target_sr:
        waveform = soxr.resample(waveform, source_sr, target_sr)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def pcm16(waveform: Waveform) -> bytes:
    return (np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def _frames(waveform: Waveform, n_fft: int, hop_length: int) -> npt.NDArray[np.float32]:
    padded = np.pad(waveform, n_fft // 2, mode="reflect")
    count = (padded.size - n_fft) // hop_length + 1
    stride = padded.strides[0]
    return np.lib.stride_tricks.as_strided(
        padded, shape=(count, n_fft), strides=(stride * hop_length, stride)
    )


def stft_db(waveform: Waveform, n_fft: int, hop_length: int) -> npt.NDArray[np.float32]:
    if waveform.size < n_fft:
        waveform = np.pad(waveform, (0, n_fft - waveform.size))
    window = np.hanning(n_fft + 1)[:-1]  # periodica, como torch.hann_window
    spectrum = np.fft.rfft(_frames(waveform, n_fft, hop_length) * window, axis=-1)
    power = spectrum.real**2 + spectrum.imag**2
    return (10.0 * np.log10(power.T + EPS)).astype(np.float32)


def db_baseline(waveform: Waveform, sr: int, n_fft: int) -> tuple[float, float]:
    spec = stft_db(waveform, n_fft, max(sr // 10, waveform.size // 3000, 1))
    lo, hi = np.percentile(spec, (5.0, 99.5))
    return float(lo), float(hi)


def db_levels(
    baseline: tuple[float, float], brightness: float, contrast: float
) -> tuple[float, float]:
    lo, hi = baseline
    center = 0.5 * (lo + hi) - brightness
    half_range = 0.5 * (hi - lo) / max(contrast, 1e-3)
    return center - half_range, center + half_range
