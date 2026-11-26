import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from models.registry import LoadedModel

# Piso de la inferencia; el slider filtra encima sin volver a correr el modelo
DETECT_THRESHOLD = 0.05

LOADED: dict[Path, "LoadedModel"] = {}


def preload() -> None:
    for module in ("inference.batch", "inference.predictor", "models.registry"):
        importlib.import_module(module)


def load(checkpoint_path: Path) -> "LoadedModel":
    from models.registry import load_checkpoint

    if checkpoint_path not in LOADED:
        LOADED.clear()  # uno a la vez: el anterior se suelta antes de cargar el siguiente
        LOADED[checkpoint_path] = load_checkpoint(checkpoint_path)
    return LOADED[checkpoint_path]


def classes(checkpoint_path: Path) -> list[str]:
    from data.species import LabelSet
    from models.registry import read_checkpoint

    return LabelSet(read_checkpoint(checkpoint_path)["labels"]).names


def detect(
    audio_path: Path,
    checkpoint_path: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    from inference.predictor import predict

    loaded = load(checkpoint_path)
    table = predict(loaded, audio_path, DETECT_THRESHOLD, on_progress)
    table.attrs["operating_point"] = loaded.operating_point
    table.attrs["device"] = loaded.device
    return table
