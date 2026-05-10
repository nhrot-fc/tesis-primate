import random
from collections.abc import Callable
from typing import Literal

import albumentations
import numpy as np
import torch
import torchaudio
from torch.nn.functional import conv2d

from domain.pipelines.types import YoloBox

ScaleMethod = Literal["min_max", "z_score", "z_score_per_band", "percentile"]

# ---------------------------------------------------------------------------
# Spectrogram computation
# ---------------------------------------------------------------------------


def compute_spectrogram(waveform: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
    spec = torchaudio.transforms.AmplitudeToDB()(
        torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)(waveform)
    )
    if spec.dim() == 2:
        spec = spec.unsqueeze(0)
    return spec


def filter_static_noise(spec: torch.Tensor) -> torch.Tensor:
    median, _ = torch.median(spec, dim=-1, keepdim=True)
    return torch.clamp(spec - median, min=0.0)


# ---------------------------------------------------------------------------
# Spectrogram normalization  (one function per method for single responsibility)
# ---------------------------------------------------------------------------


def scale_min_max(spec: torch.Tensor) -> torch.Tensor:
    s_min, s_max = spec.min(), spec.max()
    if s_max - s_min > 1e-6:
        return (spec - s_min) / (s_max - s_min)
    return torch.zeros_like(spec)


def scale_z_score(spec: torch.Tensor) -> torch.Tensor:
    return (spec - spec.mean()) / (spec.std() + 1e-8)


def scale_z_score_per_band(spec: torch.Tensor) -> torch.Tensor:
    mean = spec.mean(dim=-1, keepdim=True)
    std = spec.std(dim=-1, keepdim=True)
    return (spec - mean) / (std + 1e-8)


def scale_percentile(spec: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    p = float(torch.quantile(spec, q).item())
    p = max(p, 1e-6)
    return torch.clamp(spec, min=0.0, max=p) / p


_SCALE_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "min_max": scale_min_max,
    "z_score": scale_z_score,
    "z_score_per_band": scale_z_score_per_band,
    "percentile": scale_percentile,
}


def normalize_spectrogram(spec: torch.Tensor, method: ScaleMethod) -> torch.Tensor:
    """Dispatch to the named scaling function."""
    return _SCALE_FNS[method](spec)


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def spec_to_image(spec: torch.Tensor) -> np.ndarray:
    """Convert [C, freq, time] spectrogram tensor to [freq, time, C] uint8 numpy image."""
    spec_np = spec.permute(1, 2, 0).numpy()
    s_min, s_max = spec_np.min(), spec_np.max()
    norm = (spec_np - s_min) / (s_max - s_min) if s_max - s_min > 0 else np.zeros_like(spec_np)
    return (norm * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Spectrogram augmentations
# ---------------------------------------------------------------------------


def apply_freq_mask(
    spec: torch.Tensor, max_mask_frac: float = 0.15, fill_value: float = 0.0
) -> torch.Tensor:
    """Zero out a random contiguous band of frequency bins (SpecAugment-style)."""
    n_freq = spec.shape[-2]
    mask_size = random.randint(0, int(n_freq * max_mask_frac))
    start = random.randint(0, max(0, n_freq - mask_size))
    out = spec.clone()
    out[..., start : start + mask_size, :] = fill_value
    return out


def apply_time_mask(
    spec: torch.Tensor, max_mask_frac: float = 0.15, fill_value: float = 0.0
) -> torch.Tensor:
    """Zero out a random contiguous time window (SpecAugment-style)."""
    n_time = spec.shape[-1]
    mask_size = random.randint(0, int(n_time * max_mask_frac))
    start = random.randint(0, max(0, n_time - mask_size))
    out = spec.clone()
    out[..., :, start : start + mask_size] = fill_value
    return out


def apply_random_erasing(
    spec: torch.Tensor,
    max_area_frac: float = 0.08,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Erase a random dark rectangle from the spectrogram."""
    n_freq, n_time = spec.shape[-2], spec.shape[-1]
    max_area = max(1, int(n_freq * n_time * max_area_frac))
    area = random.randint(1, max_area)
    aspect = random.uniform(0.3, 3.0)
    h = min(n_freq, max(1, int((area * aspect) ** 0.5)))
    w = min(n_time, max(1, area // h))
    r0 = random.randint(0, n_freq - h)
    c0 = random.randint(0, n_time - w)
    out = spec.clone()
    out[..., r0 : r0 + h, c0 : c0 + w] = fill_value
    return out


def apply_gaussian_blur(
    spec: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5
) -> torch.Tensor:
    """Smooth the spectrogram with a separable Gaussian kernel."""
    k = kernel_size
    coords = torch.arange(k, dtype=torch.float32, device=spec.device) - k // 2
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d /= g1d.sum()
    kernel = (g1d.unsqueeze(0) * g1d.unsqueeze(1)).view(1, 1, k, k)

    x = spec.unsqueeze(0)  # [1, C, freq, time]
    channels = x.shape[1]
    kernel = kernel.expand(channels, 1, k, k)
    blurred = conv2d(x, kernel, padding=k // 2, groups=channels)
    return blurred.squeeze(0)


# ---------------------------------------------------------------------------
# Visual augmentation pipeline (albumentations)
# ---------------------------------------------------------------------------


def apply_visual_augmentations(
    spec: torch.Tensor,
    yolo_labels: list[YoloBox],
    transform: albumentations.Compose,
) -> tuple[torch.Tensor, list[YoloBox]]:
    """Apply an albumentations transform to a [C, freq, time] spectrogram and its YOLO labels."""
    image_np = spec.permute(1, 2, 0).numpy()
    bboxes = [[lbl.xc_rel, lbl.yc_rel, lbl.w_rel, lbl.h_rel, lbl.class_id] for lbl in yolo_labels]
    augmented = transform(image=image_np, bboxes=bboxes)
    aug_labels = [
        YoloBox(class_id=int(b[4]), xc_rel=b[0], yc_rel=b[1], w_rel=b[2], h_rel=b[3])
        for b in augmented["bboxes"]
    ]
    return torch.from_numpy(augmented["image"]).permute(2, 0, 1), aug_labels
