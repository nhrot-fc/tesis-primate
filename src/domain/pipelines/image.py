import random
from collections.abc import Callable

import albumentations
import numpy as np
import torch
import torchaudio

from src.domain.pipelines.types import AnnotationBox, PixelBBox, YoloLabel


def calc_spec_shape(
    sample_length: int,
    nfft: int,
    hop_length: int,
) -> tuple[int, int]:
    n_freq_bins = int(np.floor(nfft / 2)) + 1
    n_time_frames = int(np.floor(max(sample_length - nfft, 0) / hop_length)) + 1

    return n_freq_bins, n_time_frames


def wav_to_spec(
    waveform: torch.Tensor, sample_rate: int, n_fft: int, hop_length: int
) -> torch.Tensor:
    mono_waveform = waveform.detach().to(torch.float32).cpu()
    if mono_waveform.ndim == 2:
        mono_waveform = mono_waveform.mean(dim=0)
    if mono_waveform.ndim != 1:
        raise ValueError("waveform must be a 1D mono signal or a 2D tensor")

    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        center=False,
    )(mono_waveform)
    spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=80)(spectrogram)

    spec_min = torch.min(spectrogram)
    spec_max = torch.max(spectrogram)
    if torch.isclose(spec_max, spec_min):
        return torch.zeros_like(spectrogram, dtype=torch.float32)

    normalized = (spectrogram - spec_min) / (spec_max - spec_min)
    return normalized.to(torch.float32)


def map_annotations_to_pixels(
    annotations: list[AnnotationBox],
    sample_length: int,
    sample_rate: int,
    nfft: int,
    hop_length: int,
    class_mapping_fn: Callable[[str, str], int],
) -> list[PixelBBox]:
    spec_height, spec_width = calc_spec_shape(sample_length, nfft, hop_length)
    bounding_boxes: list[PixelBBox] = []
    for ann in annotations:
        class_id = class_mapping_fn(ann.specie, ann.call_type)

        x_min = int(np.floor(ann.begin_time * sample_rate / hop_length))
        x_max = int(np.ceil(ann.end_time * sample_rate / hop_length))
        y_min = int(np.floor(ann.low_freq * spec_height / (sample_rate / 2)))
        y_max = int(np.ceil(ann.high_freq * spec_height / (sample_rate / 2)))

        x_min = int(np.clip(x_min, 0, max(spec_width - 1, 0)))
        x_max = int(np.clip(x_max, x_min + 1, spec_width))
        y_min = int(np.clip(y_min, 0, max(spec_height - 1, 0)))
        y_max = int(np.clip(y_max, y_min + 1, spec_height))

        bounding_boxes.append(PixelBBox(class_id, x_min, y_min, x_max, y_max))

    return bounding_boxes


def _to_pascal_voc(
    box: PixelBBox, width: int, height: int
) -> tuple[float, float, float, float]:
    x_min = float(np.clip(box.x_min, 0, max(width - 1, 0)))
    y_min = float(np.clip(box.y_min, 0, max(height - 1, 0)))
    x_max = float(np.clip(box.x_max, x_min + 1.0, width))
    y_max = float(np.clip(box.y_max, y_min + 1.0, height))
    return x_min, y_min, x_max, y_max


def _to_yolo_label(
    box: tuple[float, float, float, float], class_id: int, width: int, height: int
) -> YoloLabel:
    x_min, y_min, x_max, y_max = box
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + box_width / 2.0
    y_center = y_min + box_height / 2.0

    return YoloLabel(
        class_id=class_id,
        xc_rel=x_center / width,
        yc_rel=y_center / height,
        w_rel=box_width / width,
        h_rel=box_height / height,
    )


def apply_visual_augmentations(
    spectrogram_img: torch.Tensor,
    pixel_bboxes: list[PixelBBox],
    blur: float = 0.0,
    dropout: float = 0.0,
    seed: int = 42,
) -> tuple[torch.Tensor, list[YoloLabel]]:
    image_tensor = spectrogram_img.detach().to(torch.float32).cpu()
    if image_tensor.ndim == 3 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
    if image_tensor.ndim != 2:
        raise ValueError("spectrogram_img must be a 2D tensor or have shape [1, H, W]")

    height, width = map(int, image_tensor.shape)
    if height <= 0 or width <= 0:
        raise ValueError("spectrogram_img must have positive spatial dimensions")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transforms = []
    if blur > 0:
        blur_limit = max(3, int(round(blur)) | 1)
        transforms.append(
            albumentations.GaussianBlur(blur_limit=(3, blur_limit), p=1.0)
        )
    if dropout > 0:
        transforms.append(
            albumentations.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill=0.0,
                p=min(1.0, dropout),
            )
        )

    transform = albumentations.Compose(
        transforms,
        bbox_params=albumentations.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_area=1.0,
            min_visibility=0.0,
        ),
    )

    image_np = image_tensor.numpy()
    input_bboxes = [
        _to_pascal_voc(box, width=width, height=height) for box in pixel_bboxes
    ]
    class_labels = [box.class_id for box in pixel_bboxes]

    transformed = transform(
        image=image_np,
        bboxes=input_bboxes,
        class_labels=class_labels,
    )

    augmented_tensor = torch.from_numpy(
        np.asarray(transformed["image"], dtype=np.float32)
    )
    augmented_labels: list[YoloLabel] = []
    for bbox, class_id in zip(
        transformed["bboxes"], transformed["class_labels"], strict=False
    ):
        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            continue
        augmented_labels.append(
            _to_yolo_label(
                (x_min, y_min, x_max, y_max), int(class_id), width=width, height=height
            )
        )

    return augmented_tensor, augmented_labels
