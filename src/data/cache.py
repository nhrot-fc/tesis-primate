import json
from pathlib import Path

from core.config import settings
from data.species import LabelSet

SPLITS = ("train", "val", "test")


def split_path(split: str) -> Path:
    return settings.processed_dir / f"{split}.pt"


def read_json(name: str) -> dict:
    path = settings.processed_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"falta {path}. Corré `python src/prepare_data.py` para generar el caché."
        )
    return json.loads(path.read_text())


def meta() -> dict:
    return read_json("meta.json")


def labels() -> LabelSet:
    return LabelSet(read_json("labels.json").values())


def db_range() -> tuple[float, float]:
    # Percentiles de dB del mel de train, con los que Faster R-CNN y YOLO pasan a gris.
    low, high = meta()["db_range"]
    return float(low), float(high)
