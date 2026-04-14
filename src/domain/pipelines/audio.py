from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, overload

import librosa
import numpy as np
import numpy.typing as npt
import torch
import torchaudio

from domain.pipelines.types import AnnotationBox


@overload
def convert_audio(audio: torch.Tensor, target: Literal["tensor"]) -> torch.Tensor: ...


@overload
def convert_audio(audio: torch.Tensor, target: Literal["array"]) -> npt.NDArray[np.float32]: ...


@overload
def convert_audio(audio: npt.NDArray[np.float32], target: Literal["tensor"]) -> torch.Tensor: ...


@overload
def convert_audio(
    audio: npt.NDArray[np.float32], target: Literal["array"]
) -> npt.NDArray[np.float32]: ...


def convert_audio(
    audio: torch.Tensor | npt.NDArray[np.float32], target: Literal["tensor", "array"]
) -> torch.Tensor | npt.NDArray[np.float32]:
    if target == "tensor":
        if isinstance(audio, torch.Tensor):
            return audio.detach().to(torch.float32).cpu()
        return torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(torch.float32)

    if isinstance(audio, torch.Tensor):
        return audio.detach().to(torch.float32).cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def librosa_load_audio_file(file_path: Path, sample_rate: int) -> torch.Tensor:
    audio_array, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    return convert_audio(audio_array, target="tensor")


def torchaudio_load_audio_file(file_path: Path, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return convert_audio(waveform.squeeze(0), target="tensor")


def slice_audio_window(
    audio: torch.Tensor,
    annotations: Sequence[AnnotationBox],
    sample_rate: int,
    start_time_sec: float,
    window_duration_sec: float,
) -> tuple[torch.Tensor, list[AnnotationBox]]:
    audio_1d = convert_audio(audio, target="tensor")
    if audio_1d.ndim != 1:
        raise ValueError("audio must be a 1D mono signal")

    total_samples = int(audio_1d.shape[0])
    window_start_sec = max(0.0, start_time_sec)
    window_end_sec = window_start_sec + window_duration_sec

    start_sample = int(round(window_start_sec * sample_rate))
    end_sample = int(round(window_end_sec * sample_rate))
    start_sample = min(max(start_sample, 0), total_samples)
    end_sample = min(max(end_sample, start_sample), total_samples)

    sliced_audio = audio_1d[start_sample:end_sample].clone()

    adjusted_annotations: list[AnnotationBox] = []
    for box in annotations:
        overlap_start = max(box.begin_time, window_start_sec)
        overlap_end = min(box.end_time, window_end_sec)

        if overlap_end <= overlap_start:
            continue

        adjusted_annotations.append(
            replace(
                box,
                begin_time=overlap_start - window_start_sec,
                end_time=overlap_end - window_start_sec,
            )
        )

    return sliced_audio, adjusted_annotations


def apply_time_stretch(
    audio: torch.Tensor, annotations: Sequence[AnnotationBox], stretch_factor: float
) -> tuple[torch.Tensor, list[AnnotationBox]]:
    if stretch_factor <= 0:
        raise ValueError("stretch_factor must be greater than zero")

    audio_1d = convert_audio(audio, target="tensor")
    if audio_1d.ndim != 1:
        raise ValueError("audio must be a 1D mono signal")

    stretched_audio = librosa.effects.time_stretch(
        y=convert_audio(audio_1d, target="array"), rate=stretch_factor
    )

    transformed_annotations = [
        replace(
            box,
            begin_time=box.begin_time / stretch_factor,
            end_time=box.end_time / stretch_factor,
        )
        for box in annotations
    ]

    return (
        convert_audio(stretched_audio.astype(np.float32, copy=False), target="tensor"),
        transformed_annotations,
    )


def apply_pitch_shift(
    audio: torch.Tensor,
    annotations: Sequence[AnnotationBox],
    sample_rate: int,
    semitones: float,
) -> tuple[torch.Tensor, list[AnnotationBox]]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")

    audio_1d = convert_audio(audio, target="tensor")
    if audio_1d.ndim != 1:
        raise ValueError("audio must be a 1D mono signal")

    shifted_audio = librosa.effects.pitch_shift(
        y=convert_audio(audio_1d, target="array"),
        sr=sample_rate,
        n_steps=semitones,
    )

    pitch_ratio = 2.0 ** (semitones / 12.0)
    transformed_annotations = [
        replace(
            box,
            low_freq=max(0.0, box.low_freq * pitch_ratio),
            high_freq=max(0.0, box.high_freq * pitch_ratio),
        )
        for box in annotations
    ]

    return (
        convert_audio(shifted_audio.astype(np.float32, copy=False), target="tensor"),
        transformed_annotations,
    )
