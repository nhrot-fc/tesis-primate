import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from models.registry import LoadedModel

# El visor pide al modelo todo lo que supere esto y el slider de score filtra de acá para
# arriba: bajar el slider no vuelve a correr el modelo.
DETECT_THRESHOLD = 0.05

# Cache en memoria de los checkpoints cargados
# Esto es posible porque los checkpoints son ligeros <= 400 MB
LOADED: dict[Path, "LoadedModel | Any"] = {}


def preload() -> None:
    # Carga torch, transformers y los modelos fuera del hilo de la interfaz.
    for module in ("inference.predictor", "models.registry"):
        importlib.import_module(module)


def detect(
    audio_path: Path,
    checkpoint_path: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    from core.runtime import resolve_device
    from inference.predictor import predict
    from models.registry import load_checkpoint

    device = resolve_device()
    if checkpoint_path not in LOADED:
        LOADED.clear()
        LOADED[checkpoint_path] = load_checkpoint(checkpoint_path, device)
    loaded = LOADED[checkpoint_path]

    table = predict(
        loaded, audio_path, device, score_threshold=DETECT_THRESHOLD, on_progress=on_progress
    )
    table.attrs["operating_point"] = loaded.operating_point
    return table
