import numpy as np
import numpy.typing as npt
import soundfile
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn

from core.config import P, Parameters

FloatArray = npt.NDArray[np.float64]


def load_clip(audio_path: str, clip_start_s: float, params: Parameters = P) -> Tensor:
    with soundfile.SoundFile(audio_path) as audio_file:
        source_sample_rate = audio_file.samplerate
        audio_file.seek(int(clip_start_s * source_sample_rate))
        frames = audio_file.read(
            int(params.clip_len_s * source_sample_rate), dtype="float32", always_2d=True
        )

    waveform = torch.from_numpy(frames.mean(axis=1))
    if source_sample_rate != params.target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, params.target_sr)

    num_samples = params.clip_len_samples
    if waveform.numel() < num_samples:
        return F.pad(waveform, (0, num_samples - waveform.numel()))
    return waveform[:num_samples]


def window_starts(duration_s: float, params: Parameters) -> FloatArray:
    last_start = max(duration_s - params.clip_len_s, 0.0)
    if last_start == 0.0:
        return np.zeros(1)
    return np.arange(0.0, last_start + params.clip_hop_s / 2, params.clip_hop_s)


def _mel(hz: FloatArray | float, params: Parameters) -> FloatArray:
    return params.mel_scale_q * np.log10(
        1.0 + np.asarray(hz, dtype=np.float64) / params.mel_break_hz
    )


def hz_to_y(freq_hz: FloatArray, params: Parameters) -> FloatArray:
    """Hz -> posición vertical normalizada en el eje mel HTK (y=0 = graves)."""
    mel_lo, mel_hi = _mel(params.f_min, params), _mel(params.f_max, params)
    return (_mel(freq_hz, params) - mel_lo) / (mel_hi - mel_lo)


def y_to_hz(y: FloatArray, params: Parameters) -> FloatArray:
    """Inversa de `hz_to_y`: posición normalizada [0,1] -> Hz."""
    mel_lo, mel_hi = _mel(params.f_min, params), _mel(params.f_max, params)
    mel_value = mel_lo + np.clip(y, 0.0, 1.0) * (mel_hi - mel_lo)
    return params.mel_break_hz * (10.0 ** (mel_value / params.mel_scale_q) - 1.0)


class LogMelSpectrogram(nn.Module):
    def __init__(self, params: Parameters = P) -> None:
        super().__init__()
        self.eps = params.eps
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
        log_mel = torch.log(self.mel_spectrogram(waveform) + self.eps)
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True)
        return (log_mel - mean) / (std + self.eps)
