from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import Any

import cv2
import librosa
import numpy as np

from src.domain.pipelines.audio import AnnotationBox, AudioArray, YoloCoord

type PixelBBox = tuple[int, float, float, float, float]


def _get_albumentations_module() -> ModuleType:
    try:
        return import_module("albumentations")
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Albumentations no está instalado. Ejecuta la sincronización de dependencias del proyecto."
        ) from exc


def audio_to_db_spectrogram(
    waveform: AudioArray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: float | None = None,
    ref_power: float = 1.0,
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> np.ndarray:
    if db_max <= db_min:
        raise ValueError("db_max must be greater than db_min")

    power_spectrogram = (
        np.abs(
            librosa.stft(
                waveform.astype(np.float32), n_fft=n_fft, hop_length=hop_length
            )
        )
        ** 2
    )

    if fmax is not None:
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        frequency_mask = frequencies <= fmax
        if np.any(frequency_mask):
            power_spectrogram = power_spectrogram[frequency_mask, :]

    db_spectrogram = librosa.power_to_db(power_spectrogram, ref=ref_power)
    return np.clip(db_spectrogram, db_min, db_max).astype(np.float32)


def apply_specaugment(
    db_spectrogram: np.ndarray,
    num_time_masks: int = 2,
    max_time_mask_width: int = 32,
    num_freq_masks: int = 2,
    max_freq_mask_height: int = 32,
    background_db: float | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    if db_spectrogram.ndim != 2:
        raise ValueError("db_spectrogram must be a 2D array")

    augmented = db_spectrogram.copy()
    height, width = augmented.shape
    rng = np.random.default_rng(random_seed)

    if background_db is None:
        background_db = float(np.percentile(augmented, 20))

    max_time_width = max(1, min(max_time_mask_width, width))
    for _ in range(max(0, num_time_masks)):
        mask_width = int(rng.integers(1, max_time_width + 1))
        start_x = int(rng.integers(0, max(1, width - mask_width + 1)))
        augmented[:, start_x : start_x + mask_width] = background_db

    max_freq_height = max(1, min(max_freq_mask_height, height))
    for _ in range(max(0, num_freq_masks)):
        mask_height = int(rng.integers(1, max_freq_height + 1))
        start_y = int(rng.integers(0, max(1, height - mask_height + 1)))
        augmented[start_y : start_y + mask_height, :] = background_db

    return augmented


def db_spectrogram_to_rgb(
    db_spectrogram: np.ndarray,
    db_min: float = -80.0,
    db_max: float = 0.0,
) -> np.ndarray:
    if db_max <= db_min:
        raise ValueError("db_max must be greater than db_min")

    normalized = (db_spectrogram - db_min) * (255.0 / (db_max - db_min))
    spectrogram_uint8 = np.clip(normalized, 0, 255).astype(np.uint8)
    spectrogram_uint8 = cv2.flip(spectrogram_uint8, 0)
    return cv2.applyColorMap(spectrogram_uint8, cv2.COLORMAP_VIRIDIS)


def get_expected_spectrogram_shape(
    duration_sec: float,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: float | None = None,
) -> tuple[int, int]:
    """Retorna geométricamente las dimensiones esperadas (Alto, Ancho) del espectrograma."""
    total_samples = int(duration_sec * sample_rate)
    width = (total_samples // hop_length) + 1

    if fmax is not None:
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        height = int(np.sum(frequencies <= fmax))
    else:
        height = (n_fft // 2) + 1

    return height, width


def get_crop_window(
    target_event_bbox_px: PixelBBox,
    img_w: int,
    img_h: int,
    crop_size: int = 640,
) -> tuple[int, int]:
    """Calcula matemáticamente el punto (x_start, y_start) óptimo para centrar el evento."""
    _, event_x1, event_y1, event_x2, event_y2 = target_event_bbox_px
    event_cx = (event_x1 + event_x2) / 2.0
    event_cy = (event_y1 + event_y2) / 2.0

    x_start = int(max(0, min(event_cx - crop_size / 2.0, img_w - crop_size)))
    y_start = int(max(0, min(event_cy - crop_size / 2.0, img_h - crop_size)))

    return max(0, x_start), max(0, y_start)


def translate_boxes_to_yolo(
    global_bboxes_px: list[PixelBBox],
    x_start: int,
    y_start: int,
    crop_size: int = 640,
    retained_area_threshold: float = 0.3,
    min_box_side_px: float = 5.0,
) -> list[YoloCoord]:
    """Traslada las coordenadas absolutas al marco local del Crop y las formatea para YOLO."""
    if not 0.0 <= retained_area_threshold <= 1.0:
        raise ValueError("retained_area_threshold must be in [0, 1]")

    valid_labels: list[YoloCoord] = []

    for class_id, x1, y1, x2, y2 in global_bboxes_px:
        orig_width = max(0.0, x2 - x1)
        orig_height = max(0.0, y2 - y1)
        orig_area = orig_width * orig_height
        if orig_area <= 0.0:
            continue

        new_x1 = max(0.0, min(x1 - x_start, float(crop_size)))
        new_y1 = max(0.0, min(y1 - y_start, float(crop_size)))
        new_x2 = max(0.0, min(x2 - x_start, float(crop_size)))
        new_y2 = max(0.0, min(y2 - y_start, float(crop_size)))

        width_pixels = new_x2 - new_x1
        height_pixels = new_y2 - new_y1
        clipped_area = max(0.0, width_pixels) * max(0.0, height_pixels)
        retained_fraction = clipped_area / orig_area

        if (
            width_pixels > min_box_side_px
            and height_pixels > min_box_side_px
            and retained_fraction >= retained_area_threshold
        ):
            valid_labels.append(
                YoloCoord(
                    class_id=int(class_id),
                    xc_rel=float((new_x1 + new_x2) / 2.0 / crop_size),
                    yc_rel=float((new_y1 + new_y2) / 2.0 / crop_size),
                    w_rel=float(width_pixels / crop_size),
                    h_rel=float(height_pixels / crop_size),
                )
            )

    return valid_labels


def audio_to_rgb_spectrogram(
    waveform: AudioArray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: float | None = None,
    db_min: float = -80.0,
    db_max: float = 0.0,
    apply_specaugment_masks: bool = False,
    num_time_masks: int = 2,
    max_time_mask_width: int = 32,
    num_freq_masks: int = 2,
    max_freq_mask_height: int = 32,
    random_seed: int | None = None,
) -> np.ndarray:
    db_spectrogram = audio_to_db_spectrogram(
        waveform=waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=fmax,
        ref_power=1.0,
        db_min=db_min,
        db_max=db_max,
    )

    if apply_specaugment_masks:
        db_spectrogram = apply_specaugment(
            db_spectrogram=db_spectrogram,
            num_time_masks=num_time_masks,
            max_time_mask_width=max_time_mask_width,
            num_freq_masks=num_freq_masks,
            max_freq_mask_height=max_freq_mask_height,
            random_seed=random_seed,
        )

    return db_spectrogram_to_rgb(
        db_spectrogram=db_spectrogram,
        db_min=db_min,
        db_max=db_max,
    )


def build_visual_augmentation_pipeline(
    blur_probability: float = 0.3,
    brightness_contrast_probability: float = 0.3,
) -> Any:
    albumentations = _get_albumentations_module()
    return albumentations.Compose(
        [
            albumentations.OneOf(
                [
                    albumentations.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    albumentations.MotionBlur(blur_limit=(3, 7), p=1.0),
                ],
                p=blur_probability,
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=brightness_contrast_probability,
            ),
        ],
        bbox_params=albumentations.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.3,
        ),
    )


def apply_visual_augmentations(
    image_rgb: np.ndarray,
    yolo_labels: list[YoloCoord],
    augmenter: Any | None = None,
) -> tuple[np.ndarray, list[YoloCoord]]:
    if augmenter is None:
        augmenter = build_visual_augmentation_pipeline()
    if augmenter is None:
        raise ValueError("augmenter no puede ser None")

    bboxes = [
        [label["xc_rel"], label["yc_rel"], label["w_rel"], label["h_rel"]]
        for label in yolo_labels
    ]
    class_ids = [label["class_id"] for label in yolo_labels]

    transformed = augmenter(image=image_rgb, bboxes=bboxes, class_ids=class_ids)

    transformed_labels: list[YoloCoord] = []
    for bbox, class_id in zip(transformed["bboxes"], transformed["class_ids"]):
        transformed_labels.append(
            YoloCoord(
                class_id=int(class_id),
                xc_rel=float(bbox[0]),
                yc_rel=float(bbox[1]),
                w_rel=float(bbox[2]),
                h_rel=float(bbox[3]),
            )
        )

    return transformed["image"], transformed_labels


def annotations_to_pixel_coords(
    annotations: list[AnnotationBox],
    duration: float,
    sample_rate: int,
    img_w: int,
    img_h: int,
    class_mapping: Callable[[str, str], int],
    fmax: float | None = None,
) -> list[PixelBBox]:
    max_frequency = fmax if fmax is not None else sample_rate / 2.0
    pixel_bboxes: list[PixelBBox] = []

    for annotation in annotations:
        class_id = class_mapping(annotation["specie"], annotation["call_type"])
        if class_id == -1:
            continue

        start_time = max(0.0, annotation["begin_time"])
        end_time = min(duration, annotation["end_time"])
        if end_time <= start_time:
            continue

        low_frequency = max(0.0, min(annotation["low_freq"], max_frequency))
        high_frequency = max(0.0, min(annotation["high_freq"], max_frequency))
        if high_frequency <= low_frequency:
            continue

        x1 = (start_time / duration) * img_w
        x2 = (end_time / duration) * img_w

        y1 = (1.0 - (high_frequency / max_frequency)) * img_h
        y2 = (1.0 - (low_frequency / max_frequency)) * img_h

        pixel_bboxes.append((int(class_id), x1, y1, x2, y2))

    return pixel_bboxes


def pad_to_min_size(image: np.ndarray, min_size: int = 640) -> np.ndarray:
    """Compatibilidad retroactiva: el padding preferente ahora se realiza en audio."""
    height, width = image.shape[:2]
    target_height = max(height, min_size)
    target_width = max(width, min_size)

    if height == target_height and width == target_width:
        return image

    bottom = target_height - height
    right = target_width - width
    return cv2.copyMakeBorder(
        image,
        top=0,
        bottom=bottom,
        left=0,
        right=right,
        borderType=cv2.BORDER_REPLICATE,
    )
