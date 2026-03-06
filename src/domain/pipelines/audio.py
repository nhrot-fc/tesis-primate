import numpy as np
import numpy.typing as npt
import random
from typing import TypedDict
from collections.abc import Sequence
import pandas as pd

type AudioArray = npt.NDArray[np.float32]


class AnnotationBox(TypedDict):
    specie: str
    call_type: str
    begin_time: float
    end_time: float
    low_freq: float
    high_freq: float


class YoloCoord(TypedDict):
    class_id: int
    xc_rel: float
    yc_rel: float
    w_rel: float
    h_rel: float


def annotations_to_boxes(annotations: pd.DataFrame) -> list[AnnotationBox]:
    boxes: list[AnnotationBox] = []
    for _, row in annotations.iterrows():
        boxes.append(
            AnnotationBox(
                specie=str(row["specie"]),
                call_type=str(row["call_type"]),
                begin_time=float(row["begin_time"]),
                end_time=float(row["end_time"]),
                low_freq=float(row["low_freq"]),
                high_freq=float(row["high_freq"]),
            )
        )
    boxes.sort(key=lambda box: box["begin_time"])
    return boxes


def get_logical_windows(
    total_duration_sec: float, clip_duration_sec: float = 10.0, overlap: float = 0.1
) -> list[float]:
    """Retorna una lista de offsets (en segundos) calculados matemáticamente."""
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")

    if total_duration_sec <= clip_duration_sec:
        return [0.0]

    step = clip_duration_sec * (1.0 - overlap)
    offsets = np.arange(
        0, max(0.1, total_duration_sec - clip_duration_sec + step), step
    )
    return [float(o) for o in offsets]


def fit_audio_length(audio: AudioArray, target_length: int) -> AudioArray:
    if target_length <= 0:
        raise ValueError("target_length must be greater than zero")

    if len(audio) == target_length:
        return audio.astype(np.float32)

    if len(audio) > target_length:
        return audio[:target_length].astype(np.float32)

    pad_width = target_length - len(audio)
    return np.pad(audio, (0, pad_width), mode="constant", constant_values=0.0).astype(
        np.float32
    )


def recalculate_annotations(
    annotations: Sequence[AnnotationBox],
    offset_sec: float,
    clip_duration: float,
    iou_threshold: float = 0.3,
) -> list[AnnotationBox]:
    valid_annotations: list[AnnotationBox] = []

    for ann in annotations:
        t_start = ann["begin_time"]
        t_end = ann["end_time"]

        t_prime_start = t_start - offset_sec
        t_prime_end = t_end - offset_sec

        t_trunc_start = max(0.0, min(t_prime_start, clip_duration))
        t_trunc_end = max(0.0, min(t_prime_end, clip_duration))

        orig_dur = t_end - t_start
        new_dur = t_trunc_end - t_trunc_start

        if orig_dur > 0 and (new_dur / orig_dur) >= iou_threshold:
            new_ann = ann.copy()
            new_ann["begin_time"] = float(t_trunc_start)
            new_ann["end_time"] = float(t_trunc_end)
            valid_annotations.append(new_ann)

    return valid_annotations


def extract_clip(
    waveform: AudioArray,
    sample_rate: int | float,
    offset_sec: float,
    duration_sec: float = 5.0,
) -> AudioArray:
    target_samples = int(duration_sec * sample_rate)
    start_sample = int(offset_sec * sample_rate)
    end_sample = start_sample + target_samples

    total_samples = len(waveform)

    valid_start = max(0, start_sample)
    valid_end = min(total_samples, end_sample)

    clip = waveform[valid_start:valid_end]

    if len(clip) < target_samples:
        pad_width = target_samples - len(clip)
        clip = np.pad(clip, (0, pad_width), mode="constant", constant_values=0.0)

    return clip


def apply_acoustic_mixup(
    audio_a: AudioArray,
    ann_a: list[AnnotationBox],
    audio_b: AudioArray,
    ann_b: list[AnnotationBox],
    alpha: float | None = None,
) -> tuple[AudioArray, list[AnnotationBox]]:
    if len(audio_a) != len(audio_b):
        raise ValueError(
            "Los tensores de audio deben tener la misma longitud exacta para MixUp."
        )

    if alpha is None:
        alpha = float(np.random.beta(0.4, 0.4))
        alpha = max(0.0, min(1.0, alpha))

    mixed_audio = (alpha * audio_a) + ((1.0 - alpha) * audio_b)
    mixed_anns = ann_a + ann_b

    return mixed_audio.astype(np.float32), mixed_anns


def add_background_noise(
    signal_audio: AudioArray,
    signal_anns: list[AnnotationBox],
    noise_audio: AudioArray,
    snr_db: float = 10.0,
) -> tuple[AudioArray, list[AnnotationBox]]:
    if len(signal_audio) != len(noise_audio):
        raise ValueError("La señal y el ruido deben tener la misma longitud.")

    signal_power = np.mean(signal_audio**2)
    noise_power = np.mean(noise_audio**2)

    if noise_power == 0:
        return signal_audio, signal_anns

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale_factor = np.sqrt(target_noise_power / noise_power)

    mixed_audio = signal_audio + (noise_audio * scale_factor)

    max_amp = np.max(np.abs(mixed_audio))
    if max_amp > 1.0:
        mixed_audio = mixed_audio / max_amp

    return mixed_audio.astype(np.float32), signal_anns
