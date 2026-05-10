from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from domain.pipelines.image import (
    ScaleMethod,
    apply_visual_augmentations,
    compute_spectrogram,
    normalize_spectrogram,
    spec_to_image,
)
from domain.pipelines.recordings import apply_clip
from domain.pipelines.types import Annotation, AudioRecord, annotations_to_yolo

ClassMappingFn = Callable[[str, str], int]


@dataclass(frozen=True, slots=True)
class WindowConfig:
    duration_sec: float
    hop_sec: float
    n_fft: int
    hop_length: int
    sample_rate: int
    scale_method: ScaleMethod = "min_max"
    overlap_threshold: float = 0.5
    img_size: int = 640
    n_channels: int = 3  # repeat single-channel spec to match YOLO input (1→3)


def _sliding_windows(rec: AudioRecord, cfg: WindowConfig) -> list[tuple[AudioRecord, float]]:
    total_sec = rec.wav.shape[0] / rec.sample_rate
    start = 0.0
    windows = []
    while start + cfg.duration_sec <= total_sec:
        windows.append((rec, start))
        start += cfg.hop_sec
    return windows


def _to_spec(wav: torch.Tensor, cfg: WindowConfig) -> torch.Tensor:
    spec = compute_spectrogram(wav, cfg.n_fft, cfg.hop_length)
    spec = spec.flip(dims=[-2])  # high-freq at row 0, consistent with annotations_to_yolo
    spec = normalize_spectrogram(spec, cfg.scale_method)
    spec = F.interpolate(
        spec.unsqueeze(0), size=(cfg.img_size, cfg.img_size), mode="bilinear", align_corners=False
    ).squeeze(0)
    if cfg.n_channels > 1 and spec.shape[0] == 1:
        spec = spec.repeat(cfg.n_channels, 1, 1)
    return spec


def _to_label_tensor(
    annotations: list[Annotation], cfg: WindowConfig, class_mapping_fn: ClassMappingFn
) -> torch.Tensor:
    boxes = annotations_to_yolo(annotations, cfg.duration_sec, cfg.sample_rate, class_mapping_fn)
    if not boxes:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(
        [[b.class_id, b.xc_rel, b.yc_rel, b.w_rel, b.h_rel] for b in boxes],
        dtype=torch.float32,
    )


class SpectrogramDataset(Dataset):
    """
    Sliding-window spectrogram dataset for YOLO-style detection.

    Each item: (image [C, H, W] float32, labels [N, 5] float32).
    Labels columns: [class_id, xc_rel, yc_rel, w_rel, h_rel] in [0, 1].
    Negative windows (no annotations) return labels of shape [0, 5].
    """

    def __init__(
        self,
        recordings: list[AudioRecord],
        cfg: WindowConfig,
        class_mapping_fn: ClassMappingFn,
        transform=None,
    ) -> None:
        self.cfg = cfg
        self.class_mapping_fn = class_mapping_fn
        self.transform = transform
        self._windows = [w for rec in recordings for w in _sliding_windows(rec, cfg)]

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec, start_sec = self._windows[index]
        clip = apply_clip(rec, start_sec, self.cfg.duration_sec, self.cfg.overlap_threshold)
        spec = _to_spec(clip.wav, self.cfg)
        labels = _to_label_tensor(clip.annotations, self.cfg, self.class_mapping_fn)
        if self.transform is not None:
            boxes = annotations_to_yolo(
                clip.annotations, self.cfg.duration_sec, clip.sample_rate, self.class_mapping_fn
            )
            spec, aug_boxes = apply_visual_augmentations(spec, boxes, self.transform)
            labels = (
                torch.tensor(
                    [[b.class_id, b.xc_rel, b.yc_rel, b.w_rel, b.h_rel] for b in aug_boxes],
                    dtype=torch.float32,
                )
                if aug_boxes
                else torch.zeros((0, 5), dtype=torch.float32)
            )
        return spec, labels


def yolo_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stack images and prepend batch index to labels.

    Returns:
        images:  [B, C, H, W] float32
        targets: [total_boxes, 6] float32 — columns: [batch_idx, cls, xc, yc, w, h]
    """
    images, label_list = zip(*batch, strict=False)
    images_t = torch.stack(list(images))

    parts = [
        torch.cat([labels.new_full((labels.shape[0], 1), float(i)), labels], dim=1)
        for i, labels in enumerate(label_list)
        if labels.shape[0]
    ]
    targets = torch.cat(parts) if parts else torch.zeros((0, 6), dtype=torch.float32)
    return images_t, targets


def export_to_yolo(
    dataset: SpectrogramDataset,
    output_path: Path | str,
    split: str = "train",
) -> None:
    """
    Write a SpectrogramDataset to disk in YOLO flat-file format.

    output_path/
        {split}/images/spec_00000.png
        {split}/labels/spec_00000.txt   # one row per box: cls xc yc w h

    Build the dataset without a transform for clean (un-augmented) exports.
    """
    output_path = Path(output_path)
    img_dir = output_path / split / "images"
    lbl_dir = output_path / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(dataset)), desc=f"Exporting {split}"):
        spec, labels = dataset[i]
        img_bgr = cv2.cvtColor(spec_to_image(spec), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_dir / f"spec_{i:05d}.png"), img_bgr)
        with open(lbl_dir / f"spec_{i:05d}.txt", "w") as f:
            for cls, xc, yc, w, h in labels.tolist():
                f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def write_data_yaml(
    output_path: Path | str,
    names: dict[int, str],
    train_split: str = "train",
    val_split: str = "val",
    test_split: str | None = None,
) -> None:
    """Write data.yaml for Ultralytics YOLO training."""
    output_path = Path(output_path)
    lines = [
        f"path: {output_path.resolve()}",
        f"train: {train_split}/images",
        f"val: {val_split}/images",
    ]
    if test_split is not None:
        lines.append(f"test: {test_split}/images")
    lines += ["names:"] + [f"  {idx}: {names[idx]}" for idx in sorted(names)]
    (output_path / "data.yaml").write_text("\n".join(lines) + "\n")
