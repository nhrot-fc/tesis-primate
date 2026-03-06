from collections.abc import Callable

import cv2
import librosa
import numpy as np

from src.domain.pipelines.audio import AnnotationBox, AudioArray, YoloCoord

type PixelBBox = tuple[int, float, float, float, float]


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
) -> list[YoloCoord]:
    """Traslada las coordenadas absolutas al marco local del Crop y las formatea para YOLO."""
    valid_labels: list[YoloCoord] = []

    for class_id, x1, y1, x2, y2 in global_bboxes_px:
        new_x1 = max(0.0, min(x1 - x_start, float(crop_size)))
        new_y1 = max(0.0, min(y1 - y_start, float(crop_size)))
        new_x2 = max(0.0, min(x2 - x_start, float(crop_size)))
        new_y2 = max(0.0, min(y2 - y_start, float(crop_size)))

        width_pixels = new_x2 - new_x1
        height_pixels = new_y2 - new_y1

        if width_pixels > 5 and height_pixels > 5:
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
) -> np.ndarray:
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

    db_spectrogram = librosa.power_to_db(power_spectrogram, ref=1.0)

    min_value = float(np.min(db_spectrogram))
    max_value = float(np.max(db_spectrogram))
    if max_value > min_value:
        normalized = (db_spectrogram - min_value) * (255.0 / (max_value - min_value))
    else:
        normalized = np.zeros_like(db_spectrogram, dtype=np.float32)

    spectrogram_uint8 = np.clip(normalized, 0, 255).astype(np.uint8)
    spectrogram_uint8 = cv2.flip(spectrogram_uint8, 0)

    return cv2.applyColorMap(spectrogram_uint8, cv2.COLORMAP_VIRIDIS)


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
    """Aplica padding en la parte inferior y derecha para no alterar el origen (0,0)."""
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
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
