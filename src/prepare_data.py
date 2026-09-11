import argparse
import json
import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.config import PROCESSED_DIR, SEED, P, Parameters
from core.runtime import setup_logging
from data import cache
from data.annotations import load_annotations
from data.manifest import ClipWindow, build_manifest, event_windows, sample_windows, split_manifest
from data.species import BACKGROUND_SPECIES, LabelSet
from utils.audio import load_clip, mel_db_range, mel_spectrogram

logger = logging.getLogger("prepare_data")

# Mínimo de anotaciones de una etiqueta species/call_type para ser clase
MIN_PAIR_COUNT = 100
# Fracción de ventanas sin anotaciones, sobre el total
EMPTY_RATIO = 0.25
# Fracción de ventanas con canto de ave (PteroSet), sobre las positivas
BACKGROUND_RATIO = 0.25
# Splits train/val/test por grabación
SPLIT_RATIOS = (0.6, 0.225, 0.175)
EXCLUDED_LABELS = ("lw/cc", "sm/fc", "sb/pcs")
JOINED_LABELS = {
    "lw/tr": "lw/trino",
    "lw/tj": "lw/trino",
    "lw/tt": "lw/trino",
    "lw/tf": "lw/trino",
    "sb/lpc": "sb/ppc",
}


def select_experiment(annotations: pd.DataFrame) -> tuple[pd.DataFrame, LabelSet]:
    raven_df = annotations.loc[~annotations["species"].isin(BACKGROUND_SPECIES)].copy()
    raven_df["low_freq_hz"] = raven_df["low_freq_hz"].clip(lower=P.f_min)
    raven_df["label"] = (raven_df["species"] + "/" + raven_df["call_type"]).replace(JOINED_LABELS)
    raven_df = raven_df.loc[~raven_df["label"].isin(EXCLUDED_LABELS)]

    counts = raven_df["label"].value_counts()
    frequent = counts[counts >= MIN_PAIR_COUNT].index.tolist()
    experiment = raven_df.loc[raven_df["label"].isin(frequent)]
    labels = LabelSet(experiment["label"])
    logger.info(
        "%d anotaciones | %d/%d etiquetas con >= %d: %s",
        len(experiment),
        len(frequent),
        len(counts),
        MIN_PAIR_COUNT,
        ", ".join(labels.names),
    )
    return experiment, labels


def build_dataset(manifest: list[ClipWindow], params: Parameters = P) -> dict[str, Any]:
    to_mel = mel_spectrogram(params)
    images = torch.empty(len(manifest), 1, params.n_mels, params.n_frames, dtype=torch.float32)
    boxes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for index, window in enumerate(tqdm(manifest, desc="materializando", disable=None)):
        images[index] = to_mel(load_clip(window.audio_path, window.clip_start_s, params))
        boxes.append(torch.from_numpy(window.boxes.astype(np.float32)))
        labels.append(torch.from_numpy(window.labels))
    return {"images": images, "boxes": boxes, "labels": labels}


def write_meta(db_range: tuple[float, float]) -> None:
    meta = {
        "seed": SEED,
        "min_pair_count": MIN_PAIR_COUNT,
        "empty_ratio": EMPTY_RATIO,
        "background_species": sorted(BACKGROUND_SPECIES),
        "background_ratio": BACKGROUND_RATIO,
        "split_ratios": SPLIT_RATIOS,
        "excluded_labels": EXCLUDED_LABELS,
        "joined_labels": JOINED_LABELS,
        "db_range": db_range,
        "params": asdict(P),
    }
    (PROCESSED_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Materializa el caché de ventanas.")
    parser.add_argument("--force", action="store_true", help="regenera el caché existente")
    parser.add_argument(
        "--sources-only", action="store_true", help="reescribe sólo el mapa ventana -> grabación"
    )
    args = parser.parse_args()
    setup_logging()
    if (PROCESSED_DIR / "meta.json").exists() and not (args.force or args.sources_only):
        raise SystemExit(f"ya hay un caché en {PROCESSED_DIR}; pasá --force para regenerarlo.")

    annotations = load_annotations()
    experiment, labels = select_experiment(annotations)
    manifest = build_manifest(experiment, labels, empty_ratio=EMPTY_RATIO, seed=SEED)
    birds = annotations.loc[annotations["species"].isin(BACKGROUND_SPECIES)]
    if len(birds):
        n_positive = sum(len(window.boxes) > 0 for window in manifest)
        bird_windows = sample_windows(event_windows(birds), round(n_positive * BACKGROUND_RATIO))
        manifest += bird_windows
        logger.info("%d ventanas positivas | %d de fondo (aves)", n_positive, len(bird_windows))
    splits = split_manifest(manifest, n_classes=len(labels), seed=SEED, ratios=SPLIT_RATIOS)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in zip(cache.SPLITS, splits, strict=True):
        cache.write_sources(name, split)
    if args.sources_only:
        return

    (PROCESSED_DIR / "meta.json").unlink(missing_ok=True)
    (PROCESSED_DIR / "labels.json").write_text(
        json.dumps(dict(enumerate(labels.names)), indent=2, ensure_ascii=False)
    )
    db_range = (0.0, 0.0)
    for name, split in zip(cache.SPLITS, splits, strict=True):
        logger.info("%s: %d ventanas", name, len(split))
        windows = build_dataset(split)
        if name == "train":
            db_range = mel_db_range(windows["images"][:, 0])
        path = cache.split_path(name)
        torch.save(windows, path)
        logger.info("%s -> %s (%.2f GB)", name, path, path.stat().st_size / 1024**3)
    write_meta(db_range)


if __name__ == "__main__":
    main()
