from collections.abc import Callable, Sequence
from typing import Literal

import albumentations
import numpy as np
import torch
import torchaudio

from domain.pipelines.types import AnnotationBox, YoloLabel


def wav_to_spec(
    waveform: torch.Tensor,
    n_fft: int,
    hop_length: int,
    filter_static_noise: bool = False,
    scale_method: (Literal["min_max", "z_score", "z_score_per_band", "percentile"] | None) = None,
) -> torch.Tensor:

    spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    db_transform = torchaudio.transforms.AmplitudeToDB()
    spec = db_transform(spec_transform(waveform))

    if filter_static_noise:
        median_profile, _ = torch.median(spec, dim=-1, keepdim=True)
        spec = spec - median_profile

        spec = torch.clamp(spec, min=0.0)

    if scale_method == "min_max":
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max - spec_min > 1e-6:
            spec = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec = torch.zeros_like(spec)

    elif scale_method == "z_score":
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)

    elif scale_method == "z_score_per_band":
        mean = spec.mean(dim=-1, keepdim=True)
        std = spec.std(dim=-1, keepdim=True)
        spec = (spec - mean) / (std + 1e-8)

    elif scale_method == "percentile":
        p95 = torch.quantile(spec, 0.95)
        spec = torch.clamp(spec, min=torch.tensor(0.0), max=p95)
        spec = spec / p95 if p95 > 1e-06 else torch.zeros_like(spec)
    spec = torch.flip(spec, dims=[-2])
    if spec.dim() == 2:
        spec = spec.unsqueeze(0)

    return spec


def spec_to_image(spec_tensor: torch.Tensor) -> np.ndarray:
    spec_np = spec_tensor.squeeze(0).permute(1, 2, 0).numpy()
    spec_min = np.min(spec_np)
    spec_max = np.max(spec_np)
    if spec_max - spec_min > 0:
        spec_norm = (spec_np - spec_min) / (spec_max - spec_min)
    else:
        spec_norm = np.zeros_like(spec_np)
    spec_img = (spec_norm * 255).astype(np.uint8)
    return spec_img


def annotations_to_yolo(
    annotations: Sequence[AnnotationBox],
    window_duration_sec: float,
    sample_rate: int,
    class_mapping_fn: Callable[[str, str], int],
) -> list[YoloLabel]:
    max_freq = sample_rate / 2.0
    yolo_labels = []

    for ann in annotations:
        center_time = ann.begin_time + (ann.end_time - ann.begin_time) / 2.0
        xc_rel = center_time / window_duration_sec
        w_rel = (ann.end_time - ann.begin_time) / window_duration_sec

        center_freq = ann.low_freq + (ann.high_freq - ann.low_freq) / 2.0
        yc_rel = 1.0 - (center_freq / max_freq)
        h_rel = (ann.high_freq - ann.low_freq) / max_freq

        yolo_labels.append(
            YoloLabel(
                class_id=class_mapping_fn(ann.specie, ann.call_type),
                xc_rel=np.clip(xc_rel, 0.0, 1.0),
                yc_rel=np.clip(yc_rel, 0.0, 1.0),
                w_rel=np.clip(w_rel, 0.0, 1.0),
                h_rel=np.clip(h_rel, 0.0, 1.0),
            )
        )
    return yolo_labels


def apply_visual_augmentations(
    spec_tensor: torch.Tensor, yolo_labels: list[YoloLabel], transform: albumentations.Compose
) -> tuple[torch.Tensor, list[YoloLabel]]:
    image_np = spec_tensor.permute(1, 2, 0).numpy()

    bboxes = [[lbl.xc_rel, lbl.yc_rel, lbl.w_rel, lbl.h_rel, lbl.class_id] for lbl in yolo_labels]

    augmented = transform(image=image_np, bboxes=bboxes)

    aug_image_tensor = torch.from_numpy(augmented["image"]).permute(2, 0, 1)

    aug_labels = [
        YoloLabel(
            class_id=int(box[4]),
            xc_rel=box[0],
            yc_rel=box[1],
            w_rel=box[2],
            h_rel=box[3],
        )
        for box in augmented["bboxes"]
    ]

    return aug_image_tensor, aug_labels
