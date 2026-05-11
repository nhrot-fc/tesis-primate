import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from domain.pipelines.image import compute_spectrogram, normalize_spectrogram, spec_to_image
from domain.pipelines.recordings import apply_clip
from domain.pipelines.types import ScaleMethod, annotations_to_yolo


@dataclass(frozen=True, slots=True)
class WindowConfig:
    duration_sec: float
    hop_sec: float
    n_fft: int
    hop_length: int
    sample_rate: int
    img_size: int = 640
    channel_methods: tuple[ScaleMethod, ...] = ("min_max", "z_score_per_band", "noise_filtered")


class SpectrogramDataset(Dataset):
    def __init__(self, recordings, cfg, class_mapping_fn) -> None:
        self.cfg = cfg
        self.class_mapping_fn = class_mapping_fn
        self._windows = []
        for rec in recordings:
            total_sec = rec.wav.shape[0] / rec.sample_rate
            start = 0.0
            while start + cfg.duration_sec <= total_sec:
                self._windows.append((rec, start))
                start += cfg.hop_sec

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        rec, start_sec = self._windows[index]
        clip = apply_clip(rec, start_sec, self.cfg.duration_sec)

        spec_raw = compute_spectrogram(clip.wav, self.cfg.n_fft, self.cfg.hop_length)
        spec_raw = spec_raw.flip(dims=[-2])  # Frecuencias bajas abajo

        channels = []
        for method in self.cfg.channel_methods:
            channels.append(normalize_spectrogram(spec_raw, method))

        spec = torch.cat(channels, dim=0)
        spec = F.interpolate(
            spec.unsqueeze(0), size=(self.cfg.img_size, self.cfg.img_size)
        ).squeeze(0)

        boxes = annotations_to_yolo(
            annotations=clip.annotations,
            window_duration_sec=self.cfg.duration_sec,
            sample_rate=self.cfg.sample_rate,
            class_mapping_fn=self.class_mapping_fn,
        )
        labels = (
            torch.tensor([[b.class_id, b.xc_rel, b.yc_rel, b.w_rel, b.h_rel] for b in boxes])
            if boxes
            else torch.zeros((0, 5))
        )

        return spec, labels


def export_to_yolo(
    dataset: SpectrogramDataset, output_path: Path, split: str, bg_keep_ratio=0.1
) -> None:
    img_dir = output_path / split / "images"
    lbl_dir = output_path / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(dataset)), desc=f"Exporting {split}"):
        spec, labels = dataset[i]

        if len(labels) == 0 and random.random() > bg_keep_ratio:
            continue

        img_bgr = cv2.cvtColor(spec_to_image(spec), cv2.COLOR_RGB2BGR)
        img_path = img_dir / f"spec_{i:05d}.png"
        cv2.imwrite(str(img_path), img_bgr)

        with open(lbl_dir / f"spec_{i:05d}.txt", "w") as f:
            for cls, xc, yc, w, h in labels.tolist():
                f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def write_data_yaml(
    output_path: Path,
    names: dict[int, str],
    train_split: str = "train",
    val_split: str = "val",
    test_split: str | None = None,
) -> None:
    lines = [
        f"path: {output_path.resolve()}",
        f"train: {train_split}/images",
        f"val: {val_split}/images",
    ]
    if test_split is not None:
        lines.append(f"test: {test_split}/images")
    lines += ["names:"] + [f"  {idx}: {names[idx]}" for idx in sorted(names)]
    (output_path / "data.yaml").write_text("\n".join(lines) + "\n")
