import argparse
import json
import logging
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.config import SEED, P, Parameters, settings
from core.runtime import setup_logging
from data import cache
from data.annotations import load_annotations
from data.manifest import ClipWindow, build_manifest, split_manifest
from data.species import LabelSet
from utils.audio import load_clip, mel_db_range, mel_spectrogram

logger = logging.getLogger("prepare_data")

MIN_PAIR_COUNT = 100
# Se queda con las N clases más frecuentes de las que sobreviven a `EXCLUDED_PAIRS`,
# `JOINED_PAIRS` y `MIN_PAIR_COUNT`. `None` las conserva todas.
MAX_CLASSES: int | None = None
EMPTY_RATIO = 0.25
SPLIT_RATIOS = (0.6, 0.225, 0.175)  # train / val / test, por archivo de audio
LABEL_BY = "species/call_type"
LABEL_COLUMN = {
    "call": lambda df: "call",
    "species": lambda df: df["species"],
    "species/call_type": lambda df: df["species"] + "/" + df["call_type"],
}
EXCLUDED_PAIRS: set[tuple[str, str]] = {("lw", "cc"), ("sm", "fc"), ("sb", "pcs")}
JOINED_PAIRS: dict[tuple[tuple[str, str], ...], tuple[str, str]] = {
    (("lw", "tr"), ("lw", "tj"), ("lw", "tt"), ("lw", "tf")): ("lw", "trino"),
}


def select_experiment() -> tuple[pd.DataFrame, LabelSet]:
    annotations = load_annotations(settings.data_dir / "cleaned")
    excluded = annotations[["species", "call_type"]].apply(tuple, axis=1).isin(EXCLUDED_PAIRS)
    annotations = annotations[~excluded]
    annotations["low_freq_hz"] = annotations["low_freq_hz"].clip(lower=P.f_min)
    for old_pairs, new_pair in JOINED_PAIRS.items():
        for old_pair in old_pairs:
            joined = (annotations["species"] == old_pair[0]) & (
                annotations["call_type"] == old_pair[1]
            )
            annotations.loc[joined, ["species", "call_type"]] = new_pair
    logger.info("%d anotaciones | %d especies", len(annotations), annotations.species.nunique())

    pairs = annotations[["species", "call_type"]].apply(tuple, axis=1)
    pair_counts = pairs.value_counts()  # ya ordenado de mayor a menor
    frequent = pair_counts[pair_counts >= MIN_PAIR_COUNT]
    logger.info(
        "%d/%d pares species/call_type con >= %d anotaciones",
        len(frequent),
        len(pair_counts),
        MIN_PAIR_COUNT,
    )
    if MAX_CLASSES is not None and len(frequent) > MAX_CLASSES:
        dropped = frequent.iloc[MAX_CLASSES:]
        frequent = frequent.iloc[:MAX_CLASSES]
        logger.info(
            "recorte a las %d más frecuentes; quedan fuera %s",
            MAX_CLASSES,
            ", ".join(f"{sp}/{ct} ({n})" for (sp, ct), n in dropped.items()),
        )

    valid_pairs = frequent.index
    logger.info(
        "pares seleccionados: %s",
        ", ".join(f"{species}/{call_type}" for species, call_type in valid_pairs),
    )

    experiment_df = annotations[pairs.isin(valid_pairs)].copy()
    experiment_df["label"] = LABEL_COLUMN[LABEL_BY](experiment_df)

    labels = LabelSet(experiment_df["label"])
    logger.info(
        "%d anotaciones del experimento | %d clases: %s",
        len(experiment_df),
        len(labels),
        ", ".join(labels.names),
    )
    return experiment_df, labels


def compute_mel_statistics(images: torch.Tensor, chunk: int = 256) -> dict[str, float]:
    total = torch.zeros((), dtype=torch.float64)
    total_sq = torch.zeros((), dtype=torch.float64)
    for start in range(0, len(images), chunk):
        block = images[start : start + chunk].double()
        total += block.sum()
        total_sq += block.pow(2).sum()

    n = images.numel()
    mean = float(total / n)
    variance = max(float(total_sq / n) - mean**2, 0.0)
    return {"mean": mean, "std": variance**0.5}


def build_dataset(manifest: list[ClipWindow], params: Parameters = P) -> dict[str, Any]:
    to_mel = mel_spectrogram(params)
    images = torch.empty(len(manifest), 1, params.n_mels, params.n_frames, dtype=torch.float32)
    boxes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for index, window in enumerate(tqdm(manifest, desc="materializando")):
        waveform = load_clip(window.audio_path, window.clip_start_s, params)
        images[index] = to_mel(waveform)
        boxes.append(torch.from_numpy(window.boxes.astype(np.float32)))
        labels.append(torch.from_numpy(window.labels))
    return {"images": images, "boxes": boxes, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materializa el caché de ventanas que consumen los tres detectores."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="regenera aunque ya haya un caché (la reconstrucción es destructiva)",
    )
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="reescribe sólo el mapa ventana -> grabación, sin recalcular un solo mel",
    )
    args = parser.parse_args()

    setup_logging()
    cache_dir = settings.processed_dir
    if (cache_dir / "meta.json").exists() and not (args.force or args.sources_only):
        raise SystemExit(f"ya hay un caché en {cache_dir}; pasá --force para regenerarlo.")

    experiment_df, labels = select_experiment()
    manifest = build_manifest(experiment_df, labels, empty_ratio=EMPTY_RATIO, seed=SEED)
    train_m, val_m, test_m = split_manifest(
        manifest, n_classes=len(labels), seed=SEED, ratios=SPLIT_RATIOS
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.sources_only:
        for name, split in [("train", train_m), ("val", val_m), ("test", test_m)]:
            cache.write_sources(name, split)
            logger.info("%s: %d ventanas -> %s", name, len(split), cache.sources_path(name))
        return

    (cache_dir / "meta.json").unlink(missing_ok=True)
    (cache_dir / "labels.json").write_text(
        json.dumps(dict(enumerate(labels.names)), indent=2, ensure_ascii=False)
    )

    normalization: dict[str, float] = {}
    db_range: list[float] = []
    for name, split in [("train", train_m), ("val", val_m), ("test", test_m)]:
        logger.info("%s: materializando %d ventanas...", name, len(split))
        cache.write_sources(name, split)  # ventana -> grabación, para el IC por grabación
        windows = build_dataset(split)
        images: torch.Tensor = windows["images"]
        if name == "train":
            normalization = compute_mel_statistics(images)
            db_range = list(mel_db_range(images[:, 0]))
            logger.info("mel de train -> normalización %s | dB %s", normalization, db_range)
        path = cache_dir / f"{name}.pt"
        torch.save(windows, path)
        logger.info("%s -> %s (%.2f GB)", name, path, path.stat().st_size / 1024**3)

    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "min_pair_count": MIN_PAIR_COUNT,
                "max_classes": MAX_CLASSES,
                "empty_ratio": EMPTY_RATIO,
                "label_by": LABEL_BY,
                "excluded_pairs": sorted(EXCLUDED_PAIRS),
                "normalization": normalization,
                "db_range": db_range,
                "params": asdict(P),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
