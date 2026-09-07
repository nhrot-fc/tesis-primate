import argparse
import logging
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle

from core.config import SEED, settings
from core.runtime import set_seed, setup_logging
from data import cache
from data.augment import AugmentConfig, Augmenter
from data.datasets import SpectrogramDataset
from utils.audio import mel_to_unit
from utils.boxes import to_pixel_xyxy

logger = logging.getLogger("show_augment")

RARE_CLASSES = 8
SILENT = AugmentConfig(p_gain=0.0, p_background=0.0, p_shift=0.0, p_paste=0.0)
ROWS = (
    ("original", None),
    ("ganancia", replace(SILENT, p_gain=1.0)),
    ("fondo", replace(SILENT, p_background=1.0)),
    ("corrimiento", replace(SILENT, p_shift=1.0)),
    ("copy-paste", replace(SILENT, p_paste=1.0)),
    ("completo", AugmentConfig()),
)


def draw(axis, mel, boxes, labels, names, db_range) -> None:
    axis.imshow(mel_to_unit(mel[0], *db_range).numpy(), origin="lower", aspect="auto", cmap="magma")
    for (t0, f0, t1, f1), label in zip(to_pixel_xyxy(boxes).tolist(), labels.tolist(), strict=True):
        axis.add_patch(
            Rectangle((t0, f0), t1 - t0, f1 - f0, fill=False, edgecolor="#4de1ff", linewidth=0.8)
        )
        axis.text(t0, f1 + 2, names[label], color="#4de1ff", fontsize=5)
    axis.set_xticks([])
    axis.set_yticks([])


def rare_windows(windows: SpectrogramDataset, n_classes: int, wanted: int) -> list[int]:
    totals = torch.zeros(n_classes)
    for labels in windows.labels:
        totals += labels.bincount(minlength=n_classes)
    rare = set(totals.argsort()[:RARE_CLASSES].tolist())
    found = [
        index
        for index, labels in enumerate(windows.labels)
        if 0 < len(labels) <= 4 and rare & set(labels.tolist())
    ]
    return found[:: max(1, len(found) // wanted)][:wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dibuja qué le hace la aumentación al mel.")
    parser.add_argument("--split", default="train", choices=cache.SPLITS)
    parser.add_argument("--windows", type=int, nargs="+", default=None)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    setup_logging()
    labels = cache.labels()
    db_range = cache.db_range()
    windows = SpectrogramDataset(cache.split_path(args.split))
    augmenter = Augmenter(
        windows.images,
        windows.boxes,
        windows.labels,
        cache.clips(args.split, len(windows.boxes)),
        AugmentConfig(),
    )
    indices = args.windows or rare_windows(windows, len(labels), args.columns)

    figure, axes = plt.subplots(
        len(ROWS), len(indices), figsize=(3.2 * len(indices), 1.9 * len(ROWS)), squeeze=False
    )
    for row, (title, config) in enumerate(ROWS):
        for column, index in enumerate(indices):
            set_seed(args.seed + column)
            if config is None:
                mel = windows.images[index]
                target = {"boxes": windows.boxes[index], "labels": windows.labels[index]}
            else:
                augmenter.config = config
                mel, target = augmenter(index)
            draw(axes[row][column], mel, target["boxes"], target["labels"], labels.names, db_range)
            axes[row][column].set_ylabel(title, fontsize=8)
            if row == 0:
                axes[row][column].set_title(f"ventana {index}", fontsize=8)

    figure.tight_layout()
    output = settings.runs_dir / "augment" / f"{args.split}_operaciones.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    logger.info("%s", output)


if __name__ == "__main__":
    main()
