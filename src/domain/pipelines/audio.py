import numpy as np
import numpy.typing as npt
import random
from typing import TypedDict
from collections.abc import Sequence

type AudioArray = npt.NDArray[np.float32]


class BBoxAnnotation(TypedDict):
    specie: str
    call_type: str
    begin_time: float
    end_time: float
    low_freq: float
    high_freq: float


def recalculate_annotations(
    annotations: Sequence[BBoxAnnotation],
    offset_sec: float,
    clip_duration: float,
    iou_threshold: float = 0.3,
) -> list[BBoxAnnotation]:
    valid_annotations: list[BBoxAnnotation] = []

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
    clip_duration: float = 5.0,
) -> AudioArray:
    target_samples = int(clip_duration * sample_rate)
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


def extract_anchored_clip(
    waveform: AudioArray,
    annotations: Sequence[BBoxAnnotation],
    target_event: BBoxAnnotation,
    sample_rate: int | float,
    clip_duration: float = 5.0,
    iou_threshold: float = 0.3,
) -> tuple[AudioArray, list[BBoxAnnotation]]:
    t_start = target_event["begin_time"]
    t_end = target_event["end_time"]

    min_offset = t_end - clip_duration
    max_offset = t_start

    if min_offset > max_offset:
        min_offset, max_offset = max_offset, min_offset

    min_offset = max(0.0, min_offset)
    max_offset = max(0.0, max_offset)

    offset_sec = random.uniform(min_offset, max_offset)

    clip_audio = extract_clip(waveform, sample_rate, offset_sec, clip_duration)

    clip_annotations = recalculate_annotations(
        annotations, offset_sec, clip_duration, iou_threshold
    )

    return clip_audio, clip_annotations


def apply_acoustic_mixup(
    audio_a: AudioArray,
    ann_a: list[BBoxAnnotation],
    audio_b: AudioArray,
    ann_b: list[BBoxAnnotation],
    alpha: float | None = None,
) -> tuple[AudioArray, list[BBoxAnnotation]]:
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
    signal_anns: list[BBoxAnnotation],
    noise_audio: AudioArray,
    snr_db: float = 10.0,
) -> tuple[AudioArray, list[BBoxAnnotation]]:
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
