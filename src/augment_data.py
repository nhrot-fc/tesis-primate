import argparse
import json
import logging
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from core.config import SEED, settings
from core.runtime import set_seed, setup_logging
from data import cache
from data.augment import AugmentConfig, Augmenter
from prepare_data import compute_mel_statistics
from utils.audio import mel_db_range

logger = logging.getLogger("augment_data")


def augment_train(config: AugmentConfig, passes: int) -> dict:
    stored = torch.load(cache.split_path("train"), weights_only=False, mmap=True)
    images, boxes, labels = stored["images"], stored["boxes"], stored["labels"]
    augmenter = Augmenter(images, boxes, labels, cache.clips("train", len(boxes)), config)

    augmented = torch.empty(len(images) * passes, *images.shape[1:])
    new_boxes: list[torch.Tensor] = []
    new_labels: list[torch.Tensor] = []
    for position in tqdm(range(len(augmented)), desc="aumentando train"):
        mel, target = augmenter(position % len(images))
        augmented[position] = mel
        new_boxes.append(target["boxes"])
        new_labels.append(target["labels"])

    logger.info(
        "train: %d ventanas y %d cajas -> %d ventanas y %d cajas",
        len(images),
        sum(len(window) for window in boxes),
        len(augmented),
        sum(len(window) for window in new_boxes),
    )
    return {"images": augmented, "boxes": new_boxes, "labels": new_labels}


def copy_split(split: str, out: Path) -> None:
    shutil.copyfile(cache.split_path(split), out / f"{split}.pt")
    shutil.copyfile(cache.sources_path(split), out / f"{split}_sources.json")


def write_train_sources(out: Path, passes: int, n_windows: int) -> None:
    found = cache.sources("train", n_windows)
    tiled = found._replace(
        recording_of_window=found.recording_of_window * passes,
        clip_start_s=found.clip_start_s * passes,
    )
    (out / "train_sources.json").write_text(json.dumps(tiled._asdict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Escribe un caché con train aumentado; val y test van sin tocar."
    )
    parser.add_argument("--out", type=Path, default=settings.data_dir / "processed_aug")
    parser.add_argument("--passes", type=int, default=1, help="copias aumentadas de train")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    config = AugmentConfig()
    args.out.mkdir(parents=True, exist_ok=True)

    windows = augment_train(config, args.passes)
    normalization = compute_mel_statistics(windows["images"])
    if not math.isfinite(normalization["mean"]):
        raise RuntimeError(f"la aumentación produjo valores no finitos: {normalization}")

    torch.save(windows, args.out / "train.pt")
    write_train_sources(args.out, args.passes, len(windows["boxes"]) // args.passes)
    for split in ("val", "test"):
        copy_split(split, args.out)
    shutil.copyfile(settings.processed_dir / "labels.json", args.out / "labels.json")

    images = windows["images"]
    meta = cache.meta() | {
        "augment": asdict(config) | {"passes": args.passes, "seed": args.seed},
        "normalization": normalization,
        "db_range": list(mel_db_range(images[:, 0])),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.info(
        "%s (%.2f GB)", args.out, sum(f.stat().st_size for f in args.out.iterdir()) / 1024**3
    )


if __name__ == "__main__":
    main()
