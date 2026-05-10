from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class Annotation:
    species: str
    call_type: str
    begin_time: float
    end_time: float
    low_freq: float
    high_freq: float


@dataclass(frozen=True, slots=True)
class ImageBoundingBox:
    class_id: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def to_yolo(self, img_width: int, img_height: int) -> "YoloBox":
        w: int = self.x_max - self.x_min
        h: int = self.y_max - self.y_min

        x_center: float = self.x_min + (w / 2.0)
        y_center: float = self.y_min + (h / 2.0)

        return YoloBox(
            class_id=self.class_id,
            xc_rel=x_center / img_width,
            yc_rel=y_center / img_height,
            w_rel=w / img_width,
            h_rel=h / img_height,
        )


@dataclass(frozen=True, slots=True)
class YoloBox:
    class_id: int
    xc_rel: float
    yc_rel: float
    w_rel: float
    h_rel: float

    def to_bbox(self, img_width: int, img_height: int) -> ImageBoundingBox:
        w: float = self.w_rel * img_width
        h: float = self.h_rel * img_height

        x_center: float = self.xc_rel * img_width
        y_center: float = self.yc_rel * img_height

        x_min = int(x_center - (w / 2.0))
        y_min = int(y_center - (h / 2.0))
        x_max = int(x_center + (w / 2.0))
        y_max = int(y_center + (h / 2.0))

        return ImageBoundingBox(
            class_id=self.class_id,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )


@dataclass(frozen=True, slots=True)
class AudioRecord:
    wav: torch.Tensor
    sample_rate: int
    annotations: list[Annotation]

    def __repr__(self) -> str:
        return f"AudioRecord(wav.shape={self.wav.shape}, sample_rate={self.sample_rate}, num_annotations={len(self.annotations)})"


TransformFn = Callable[[AudioRecord], AudioRecord]


def annotations_to_yolo(
    annotations: Sequence[Annotation],
    window_duration_sec: float,
    sample_rate: int,
    class_mapping_fn: Callable[[str, str], int],
) -> list[YoloBox]:
    max_freq = sample_rate / 2.0
    return [
        YoloBox(
            class_id=class_mapping_fn(ann.species, ann.call_type),
            xc_rel=float(
                np.clip((ann.begin_time + ann.end_time) / 2.0 / window_duration_sec, 0.0, 1.0)
            ),
            yc_rel=float(np.clip(1.0 - (ann.low_freq + ann.high_freq) / 2.0 / max_freq, 0.0, 1.0)),
            w_rel=float(np.clip((ann.end_time - ann.begin_time) / window_duration_sec, 0.0, 1.0)),
            h_rel=float(np.clip((ann.high_freq - ann.low_freq) / max_freq, 0.0, 1.0)),
        )
        for ann in annotations
    ]
