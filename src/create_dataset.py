import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from core.config import P, settings
from core.setup import setup_logging, setup_project_path
from domain.annotations import load_annotations
from domain.dataset import CallBoxDataset, ClipWindow, build_manifest, split_manifest
from domain.species import LabelSet

logger = logging.getLogger("create_dataset")

PROJECT_DIR = Path.cwd()
CACHE_DIR = PROJECT_DIR / "data" / "processed"

SEED = 42
MIN_PAIR_COUNT = 100
LABEL_BY = "species/call_type"
LABEL_COLUMN = {
    "call": lambda df: "call",
    "species": lambda df: df["species"],
    "species/call_type": lambda df: df["species"] + "/" + df["call_type"],
}
EXCLUDED_PAIRS: set[tuple[str, str]] = {("lw", "cc"), ("sm", "fc"), ("sb", "pcs")}
OTHER_LABEL = "other"


def select_experiment() -> tuple[pd.DataFrame, LabelSet]:
    annotations = load_annotations(settings.data_dir / "cleaned")
    excluded = annotations[["species", "call_type"]].apply(tuple, axis=1).isin(EXCLUDED_PAIRS)
    annotations = annotations[~excluded]
    annotations["low_freq_hz"] = annotations["low_freq_hz"].clip(lower=P.f_min)
    logger.info("%d anotaciones | %d especies", len(annotations), annotations.species.nunique())

    pairs = annotations[["species", "call_type"]].apply(tuple, axis=1)
    pair_counts = pairs.value_counts()
    valid_pairs = pair_counts[pair_counts >= MIN_PAIR_COUNT].index
    logger.info(
        "%d/%d pares species/call_type con >= %d anotaciones: %s",
        len(valid_pairs),
        len(pair_counts),
        MIN_PAIR_COUNT,
        ", ".join(f"{species}/{call_type}" for species, call_type in valid_pairs),
    )

    # Los pares que no llegan al umbral no se descartan: entrenar contra una ventana con un
    # evento real invisible (porque su anotación se tiró) enseña al modelo a no detectarlo.
    # Se agrupan en `OTHER_LABEL` para seguir dando señal de "hay una llamada acá" sin pedirle
    # al modelo que la clasifique.
    experiment_df = annotations.copy()
    experiment_df["label"] = LABEL_COLUMN[LABEL_BY](experiment_df)
    if LABEL_BY == "species/call_type":
        experiment_df.loc[~pairs.isin(valid_pairs), "label"] = OTHER_LABEL

    labels = LabelSet(experiment_df["label"])
    logger.info(
        "%d anotaciones del experimento | %d clases: %s",
        len(experiment_df),
        len(labels),
        ", ".join(labels.names),
    )
    return experiment_df, labels


def log_mel_stats(images: torch.Tensor, chunk: int = 256) -> dict[str, float]:
    """Media y desvío globales del log-mel, acumulados en float64 y por trozos para no
    duplicar en memoria un tensor de varios GB."""
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


def materialize(manifest: list[ClipWindow], params=P) -> dict[str, Any]:
    """Corre `CallBoxDataset.__getitem__` una vez por ventana y junta todo en tensores."""
    dataset = CallBoxDataset(manifest, params)
    images = torch.empty(len(dataset), 1, params.n_mels, params.n_frames, dtype=torch.float32)
    boxes: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for index in tqdm(range(len(dataset)), desc="materializando"):
        image, target = dataset[index]
        images[index] = image
        boxes.append(target["boxes"])
        labels.append(target["labels"])
    return {"images": images, "boxes": boxes, "labels": labels}


def main() -> None:
    setup_logging(settings.LOG_LEVEL)
    setup_project_path(PROJECT_DIR)

    experiment_df, labels = select_experiment()
    manifest = build_manifest(experiment_df, labels)
    train_m, val_m, test_m = split_manifest(manifest, seed=SEED)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # `meta.json` se escribe al final (necesita las estadísticas de train). Borrarlo
    # ahora evita que, si esto se corta a mitad, queden `.pt` nuevos con un meta viejo:
    # `train.py` normalizaría con estadísticas de otra corrida sin poder darse cuenta.
    (CACHE_DIR / "meta.json").unlink(missing_ok=True)
    (CACHE_DIR / "labels.json").write_text(
        json.dumps(dict(enumerate(labels.names)), indent=2, ensure_ascii=False)
    )

    normalization: dict[str, float] = {}
    for name, split in [("train", train_m), ("val", val_m), ("test", test_m)]:
        logger.info("%s: materializando %d ventanas...", name, len(split))
        cache = materialize(split)
        images: torch.Tensor = cache["images"]
        if name == "train":
            # Estadísticas del log-mel crudo sobre train (nunca sobre val/test: serían
            # una fuga). El modelo las usa para normalizar, y viajan en el checkpoint.
            normalization = log_mel_stats(images)
            logger.info("normalización (log-mel de train) -> %s", normalization)
        path = CACHE_DIR / f"{name}.pt"
        torch.save(cache, path)
        logger.info("%s -> %s (%.2f GB)", name, path, path.stat().st_size / 1024**3)

    (CACHE_DIR / "meta.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "min_pair_count": MIN_PAIR_COUNT,
                "label_by": LABEL_BY,
                "excluded_pairs": sorted(EXCLUDED_PAIRS),
                "normalization": normalization,
                "params": asdict(P),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
