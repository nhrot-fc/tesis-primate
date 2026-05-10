from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
import torch
import torchaudio
from torch.nn.functional import pad

# ---------------------------------------------------------------------------
# Type conversions
# ---------------------------------------------------------------------------


def wav_np_to_tensor(wav: npt.NDArray[np.float32]) -> torch.Tensor:
    assert wav.ndim == 1, "Expected 1D audio array"
    return torch.from_numpy(wav).float()


def wav_tensor_to_np(wav: torch.Tensor) -> npt.NDArray[np.float32]:
    assert wav.dim() == 1, "Expected 1D audio tensor"
    return wav.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_audio_librosa(file_path: Path, sample_rate: int) -> torch.Tensor:
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return wav_np_to_tensor(audio)


def load_audio_torchaudio(file_path: Path, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    return waveform.squeeze(dim=0)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def slice_audio(
    wav: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    duration_sec: float,
) -> torch.Tensor:
    assert wav.dim() == 1, "Expected 1D audio tensor"

    total: int = wav.shape[0]
    start_sec = max(0.0, start_sec)
    start: int = min(max(int(round(start_sec * sample_rate)), 0), total)
    end: int = min(max(int(round((start_sec + duration_sec) * sample_rate)), start), total)

    return wav[start:end].clone()


# ---------------------------------------------------------------------------
# Audio augmentations
# ---------------------------------------------------------------------------


def stretch_time(wav: torch.Tensor, stretch_factor: float) -> torch.Tensor:
    assert stretch_factor > 0, "stretch_factor must be positive"
    assert wav.dim() == 1, "Expected 1D audio tensor"

    stretched = librosa.effects.time_stretch(y=wav_tensor_to_np(wav), rate=stretch_factor)
    return wav_np_to_tensor(stretched.astype(np.float32, copy=False))


def speed_change(audio: torch.Tensor, sample_rate: int, speed_factor: float) -> torch.Tensor:
    assert speed_factor > 0, "speed_factor must be positive"
    assert audio.dim() == 1, "Expected 1D audio tensor"

    resampled = torchaudio.transforms.Resample(
        orig_freq=int(sample_rate * speed_factor), new_freq=sample_rate
    )(audio.unsqueeze(0))
    return resampled.squeeze(0)


def pitch_shift(audio: torch.Tensor, sample_rate: int, semitones: float) -> torch.Tensor:
    assert sample_rate > 0, "sample_rate must be positive"
    assert audio.dim() == 1, "Expected 1D audio tensor"

    shifted = librosa.effects.pitch_shift(
        y=wav_tensor_to_np(audio), sr=sample_rate, n_steps=semitones
    )
    return wav_np_to_tensor(shifted.astype(np.float32, copy=False))


def mix_audio(base: torch.Tensor, mix: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    assert base.dim() == 1, "Expected 1D audio tensor"
    assert mix.dim() == 1, "Expected 1D audio tensor"

    if mix.shape[0] > base.shape[0]:
        mix = mix[: base.shape[0]]
    elif mix.shape[0] < base.shape[0]:
        mix = pad(mix, (0, base.shape[0] - mix.shape[0]))
    signal_power = base.pow(2).mean().clamp(min=1e-10)
    noise_power = mix.pow(2).mean().clamp(min=1e-10)
    scale = (signal_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
    return base + scale * mix


def add_gaussian_noise(audio: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    assert audio.dim() == 1, "Expected 1D audio tensor"

    signal_power = audio.pow(2).mean().clamp(min=1e-10)
    noise_power = signal_power / 10 ** (snr_db / 10)
    return audio + torch.randn_like(audio) * noise_power.sqrt()
