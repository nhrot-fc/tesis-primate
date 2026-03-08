from collections.abc import Callable
import albumentations

import numpy as np
import torch
import torchaudio

from src.domain.pipelines.types import (
    AnnotationBox,
    BoundingBox,
    YoloLabel,
)

import torch
import numpy as np
from src.domain.pipelines.types import AnnotationBox, BoundingBox, YoloLabel


def calc_spec_shape(
    sample_length: int,
    nfft: int,
    hop_length: int,
) -> tuple[int, int]:
    n_freq_bins = int(np.floor(nfft / 2)) + 1
    n_time_frames = int(np.ceil((sample_length - nfft) / hop_length)) + 1

    return n_freq_bins, n_time_frames


def audio_to_spectrogram_tensor(
    waveform: torch.Tensor, sample_rate: int, n_fft: int, hop_length: int
) -> torch.Tensor:
    """
    Convierte un tensor de audio a un espectrograma usando torchaudio.

    Retorno:
        Un torch.Tensor 2D (o 3D de un canal) representando la imagen acústica.
    """
    raise NotImplementedError("audio_to_spectrogram_tensor is not implemented yet")


def map_annotations_to_pixels(
    annotations: list[AnnotationBox],
    sample_length: int,
    sample_rate: int,
    nfft: int,
    hop_length: int,
    class_mapping_fn: Callable[[str, str], int],
) -> list[BoundingBox]:
    """
    Convierte las coordenadas físicas (segundos y Hz) a píxeles exactos
    basándose en las dimensiones finales del espectrograma generado.
    """
    spec_shape = calc_spec_shape(sample_length, nfft, hop_length)
    bounding_boxes: list[BoundingBox] = []
    for ann in annotations:
        class_id = class_mapping_fn(ann.specie, ann.call_type)

        x_min = int(np.floor(ann.begin_time * sample_rate / hop_length))
        x_max = int(np.ceil(ann.end_time * sample_rate / hop_length))
        y_min = int(np.floor(ann.low_freq * spec_shape[0] / (sample_rate / 2)))
        y_max = int(np.ceil(ann.high_freq * spec_shape[0] / (sample_rate / 2)))

        bounding_boxes.append(BoundingBox(class_id, x_min, y_min, x_max, y_max))

    return bounding_boxes


def apply_visual_augmentations(
    spectrogram_img: np.ndarray,
    pixel_bboxes: list[BoundingBox],
    min_visibility: float = 0.3,
) -> tuple[np.ndarray, list[YoloLabel]]:
    """
    Aplica aumentos visuales (Blur, Cutout/CoarseDropout, ruido de píxeles)
    utilizando Albumentations. Convierte las PixelBBox resultantes a formato YOLO.

    Elimina automáticamente las cajas si su visibilidad cae por debajo
    de `min_visibility` (ej. si un Cutout tapa el llamado del primate).
    """
    raise NotImplementedError("apply_visual_augmentations is not implemented yet")
