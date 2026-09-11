import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile
import torch
import torchaudio
from torch import Tensor, nn

from core.config import SEED, P, Parameters

FloatArray = npt.NDArray[np.float64]

# Rango de dB del mel para normalizar la entrada de los modelos: recorta el 1 % más callado y
# el 0.1 % más fuerte de 2000 ventanas de train. (El visor calibra su propio rango sobre el
# STFT de cada grabación; ver `viewer.spectrogram.DISPLAY_PERCENTILES`.)
DB_PERCENTILES = (1.0, 99.9)
DB_RANGE_WINDOWS = 2000
# Cuantil de |onda| que se toma como piso de ruido al rellenar un clip corto
PAD_FLOOR_QUANTILE = 0.1


def pad_to_clip(waveform: Tensor, params: Parameters) -> Tensor:
    missing = params.clip_len_samples - waveform.numel()
    if missing <= 0:
        return waveform[: params.clip_len_samples]
    # Ruido al nivel del piso de la grabación: el silencio digital sería un escalón.
    noise_floor = float(waveform.abs().quantile(PAD_FLOOR_QUANTILE)) if waveform.numel() else 0.0
    generator = torch.Generator().manual_seed(params.pad_seed)
    return torch.cat([waveform, torch.randn(missing, generator=generator) * noise_floor])


# Lo que oye el modelo: mono y resampleado con torchaudio, igual que al construir el caché.
# El visor tiene su propio lector sin torch (`viewer.spectrogram.load_audio`) porque tiene que
# abrir audio aunque el motor de detección no cargue.
def read_clip(
    audio_file: soundfile.SoundFile, clip_start_s: float, params: Parameters = P
) -> Tensor:
    source_sr = audio_file.samplerate
    audio_file.seek(int(clip_start_s * source_sr))
    frames = audio_file.read(int(params.clip_len_s * source_sr), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(frames.mean(axis=1))
    if source_sr != params.target_sr:
        waveform = torchaudio.functional.resample(waveform, source_sr, params.target_sr)
    return pad_to_clip(waveform, params)


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
    # -1e-9: un audio que cierra justo en un múltiplo del hop no suma una ventana de relleno.
    n_hops = math.ceil((duration_s - params.clip_len_s) / params.clip_hop_s - 1e-9)
    return np.arange(n_hops + 1, dtype=np.float64) * params.clip_hop_s


def hz_to_mel(hz: FloatArray | float, params: Parameters) -> FloatArray:
    # Escala HTK: m(f) = q·log10(1 + f/f_break), la misma del banco de filtros.
    return params.mel_scale_q * np.log10(
        1.0 + np.asarray(hz, dtype=np.float64) / params.mel_break_hz
    )


def hz_to_y(freq_hz: FloatArray, params: Parameters) -> FloatArray:
    # Fila del espectrograma en [0, 1].
    low, high = hz_to_mel(params.f_min, params), hz_to_mel(params.f_max, params)
    return (hz_to_mel(freq_hz, params) - low) / (high - low)


def y_to_hz(y: FloatArray, params: Parameters) -> FloatArray:
    low, high = hz_to_mel(params.f_min, params), hz_to_mel(params.f_max, params)
    mel_value = low + np.clip(y, 0.0, 1.0) * (high - low)
    return params.mel_break_hz * (10.0 ** (mel_value / params.mel_scale_q) - 1.0)


def mel_spectrogram(params: Parameters = P) -> nn.Module:
    return torchaudio.transforms.MelSpectrogram(
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


def mel_to_db(mel: Tensor, params: Parameters = P) -> Tensor:
    # 10 y no 20: el mel va en potencia. El visor repite la fórmula en numpy sobre su STFT
    # (`viewer.spectrogram.stft_db`) con un piso más bajo, porque no pasa por el banco mel.
    return 10.0 * torch.log10(mel + params.eps)


def mel_db_range(mels: Tensor) -> tuple[float, float]:
    index = np.random.default_rng(SEED).choice(
        len(mels), size=min(DB_RANGE_WINDOWS, len(mels)), replace=False
    )
    db = mel_to_db(mels[torch.from_numpy(index)].float()).numpy()  # `torch.quantile` no escala
    low, high = np.percentile(db, DB_PERCENTILES)
    return float(low), float(high)


def mel_to_unit(mel: Tensor, low: float, high: float) -> Tensor:
    return ((mel_to_db(mel) - low) / (high - low)).clamp(0.0, 1.0)


def mel_to_gray(mel: Tensor, low: float, high: float, size: int | None = None) -> np.ndarray:
    import cv2

    gray = np.flipud((mel_to_unit(mel, low, high) * 255).to(torch.uint8).numpy())  # grave abajo
    if size is None:
        return np.ascontiguousarray(gray)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
