from dataclasses import replace
from pathlib import Path
from typing import Literal

import torch

from domain.pipelines.annotations import (
    df_to_annotations,
    get_annotation_file,
    load_annotation_file,
    normalize_header,
    normalize_species_and_calls,
    parse_numerics,
    pitch_shift_annotations,
    stretch_annotations,
)
from domain.pipelines.audio import (
    add_gaussian_noise,
    load_audio_librosa,
    load_audio_torchaudio,
    mix_audio,
    pitch_shift,
    slice_audio,
    speed_change,
    stretch_time,
)
from domain.pipelines.types import Annotation, AudioRecord

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_recording(
    audio_file: Path,
    annotations_dir: Path | None,
    sample_rate: int,
    loader: Literal["librosa", "torchaudio"] = "torchaudio",
) -> AudioRecord:
    if loader == "librosa":
        wav = load_audio_librosa(audio_file, sample_rate)
    elif loader == "torchaudio":
        wav = load_audio_torchaudio(audio_file, sample_rate)
    else:
        raise ValueError(f"Unsupported audio loader: {loader}")

    annotations: list[Annotation] = []
    if annotations_dir is not None:
        annotation_file = get_annotation_file(annotations_dir, audio_file)
        if annotation_file is not None:
            df = load_annotation_file(annotation_file)
            df = normalize_header(df)
            df = normalize_species_and_calls(df)
            df = parse_numerics(df)
            annotations = df_to_annotations(df)

    return AudioRecord(
        wav=wav,
        sample_rate=sample_rate,
        annotations=annotations,
    )


# ---------------------------------------------------------------------------
# Coupled transforms (audio + annotations)
# ---------------------------------------------------------------------------


def apply_pitch_shift(record: AudioRecord, semitones: float) -> AudioRecord:
    if semitones == 0.0:
        return record
    wav = pitch_shift(record.wav, record.sample_rate, semitones)
    annotations = pitch_shift_annotations(record.annotations, semitones)
    return AudioRecord(
        wav=wav,
        sample_rate=record.sample_rate,
        annotations=annotations,
    )


def apply_time_stretch(record: AudioRecord, stretch_factor: float) -> AudioRecord:
    assert stretch_factor > 0, "stretch_factor must be positive"
    if stretch_factor == 1.0:
        return record
    wav = stretch_time(record.wav, stretch_factor)
    annotations = stretch_annotations(record.annotations, stretch_factor)
    return AudioRecord(
        wav=wav,
        sample_rate=record.sample_rate,
        annotations=annotations,
    )


def apply_speed_change(record: AudioRecord, speed_factor: float) -> AudioRecord:
    """Change playback speed. Unlike time_stretch, this also shifts pitch (speed_factor * freq)."""
    assert speed_factor > 0, "speed_factor must be positive"
    if speed_factor == 1.0:
        return record
    wav = speed_change(record.wav, record.sample_rate, speed_factor)
    # Speed change compresses/stretches time and raises/lowers pitch proportionally.
    annotations = [
        replace(
            ann,
            begin_time=ann.begin_time / speed_factor,
            end_time=ann.end_time / speed_factor,
            low_freq=ann.low_freq * speed_factor,
            high_freq=ann.high_freq * speed_factor,
        )
        for ann in record.annotations
    ]
    return AudioRecord(
        wav=wav,
        sample_rate=record.sample_rate,
        annotations=annotations,
    )


def apply_clip(
    record: AudioRecord,
    start_sec: float,
    duration_sec: float,
    overlap_threshold: float = 0.8,
) -> AudioRecord:
    wav = slice_audio(record.wav, record.sample_rate, start_sec, duration_sec)
    annotations = _clip_annotations(record.annotations, start_sec, duration_sec, overlap_threshold)
    return AudioRecord(
        wav=wav,
        sample_rate=record.sample_rate,
        annotations=annotations,
    )


def _clip_annotations(
    annotations: list[Annotation],
    start_sec: float,
    duration_sec: float,
    overlap_threshold: float,
) -> list[Annotation]:
    end_sec = start_sec + duration_sec
    result: list[Annotation] = []
    for ann in annotations:
        overlap_start = max(ann.begin_time, start_sec)
        overlap_end = min(ann.end_time, end_sec)
        if overlap_end <= overlap_start:
            continue
        ann_duration = ann.end_time - ann.begin_time
        if ann_duration <= 0:
            continue
        if (overlap_end - overlap_start) / ann_duration >= overlap_threshold:
            result.append(
                replace(
                    ann,
                    begin_time=overlap_start - start_sec,
                    end_time=overlap_end - start_sec,
                )
            )
    return result


# ---------------------------------------------------------------------------
# Audio-only transforms
# ---------------------------------------------------------------------------


def apply_noise_mix(
    record: AudioRecord, noise_wav: torch.Tensor, snr_db: float = 10.0
) -> AudioRecord:
    wav = mix_audio(record.wav, noise_wav, snr_db)
    return AudioRecord(
        wav=wav,
        sample_rate=record.sample_rate,
        annotations=record.annotations,
    )


def apply_gaussian_noise(record: AudioRecord, snr_db: float = 20.0) -> AudioRecord:
    wav = add_gaussian_noise(record.wav, snr_db)
    return AudioRecord(
        wav=wav,
        sample_rate=record.sample_rate,
        annotations=record.annotations,
    )
