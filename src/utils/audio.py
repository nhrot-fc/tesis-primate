import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn

from core.config import P, Parameters

FloatArray = npt.NDArray[np.float64]


def waveform_padding(waveform: Tensor, params: Parameters) -> Tensor:
    missing = params.clip_len_samples - waveform.numel()
    if missing <= 0:
        return waveform[: params.clip_len_samples]
    if params.pad_mode == "zeros":
        return F.pad(waveform, (0, missing))

    noise_floor = float(waveform.abs().quantile(0.1)) if waveform.numel() else 0.0
    generator = torch.Generator().manual_seed(params.pad_seed)
    return torch.cat([waveform, torch.randn(missing, generator=generator) * noise_floor])


def read_clip(
    audio_file: soundfile.SoundFile, clip_start_s: float, params: Parameters = P
) -> Tensor:
    source_sample_rate = audio_file.samplerate
    audio_file.seek(int(clip_start_s * source_sample_rate))
    frames = audio_file.read(
        int(params.clip_len_s * source_sample_rate), dtype="float32", always_2d=True
    )

    waveform = torch.from_numpy(frames.mean(axis=1))
    if source_sample_rate != params.target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, params.target_sr)
    return waveform_padding(waveform, params)


def load_clip(audio_path: Path | str, clip_start_s: float, params: Parameters = P) -> Tensor:
    with soundfile.SoundFile(audio_path) as audio_file:
        return read_clip(audio_file, clip_start_s, params)


def load_clips(
    audio_path: Path | str, clip_starts: Iterable[float], params: Parameters = P
) -> list[Tensor]:
    with soundfile.SoundFile(audio_path) as audio_file:
        return [read_clip(audio_file, float(start), params) for start in clip_starts]


def window_starts(duration_s: float, params: Parameters) -> FloatArray:
    if duration_s <= params.clip_len_s:
        return np.zeros(1)
    # -1e-9: si el audio cierra justo en un múltiplo del hop no agrega una ventana de puro relleno
    n_hops = math.ceil((duration_s - params.clip_len_s) / params.clip_hop_s - 1e-9)
    return np.arange(n_hops + 1, dtype=np.float64) * params.clip_hop_s


def _mel(hz: FloatArray | float, params: Parameters) -> FloatArray:
    return params.mel_scale_q * np.log10(
        1.0 + np.asarray(hz, dtype=np.float64) / params.mel_break_hz
    )


def hz_to_y(freq_hz: FloatArray, params: Parameters) -> FloatArray:
    mel_lo, mel_hi = _mel(params.f_min, params), _mel(params.f_max, params)
    return (_mel(freq_hz, params) - mel_lo) / (mel_hi - mel_lo)


def y_to_hz(y: FloatArray, params: Parameters) -> FloatArray:
    mel_lo, mel_hi = _mel(params.f_min, params), _mel(params.f_max, params)
    mel_value = mel_lo + np.clip(y, 0.0, 1.0) * (mel_hi - mel_lo)
    return params.mel_break_hz * (10.0 ** (mel_value / params.mel_scale_q) - 1.0)


class MelSpectrogram(nn.Module):
    def __init__(self, params: Parameters = P) -> None:
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=params.target_sr,
            n_fft=params.n_fft,
            win_length=params.win_length,
            hop_length=params.hop_length,
            n_mels=params.n_mels,
            f_min=params.f_min,
            f_max=params.f_max,
            power=2.0,
            mel_scale=params.mel_scale,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        return self.mel_spectrogram(waveform)
