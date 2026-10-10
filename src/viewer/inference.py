import importlib
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from viewer.tasks import UnreadableError

if TYPE_CHECKING:
    from models.registry import LoadedModel

# El visor pide al modelo todo lo que supere esto y el slider de score filtra de acá para
# arriba: bajar el slider no vuelve a correr el modelo.
DETECT_THRESHOLD = 0.05

# Cache en memoria del checkpoint cargado, uno solo: el visor y la vista Batch comparten el
# modelo, y con <= 400 MB por checkpoint cabe recargarlo al cambiar de modelo.
LOADED: dict[Path, "LoadedModel"] = {}


def preload() -> None:
    # Carga torch, transformers y los modelos fuera del hilo de la interfaz.
    for module in ("inference.batch", "inference.predictor", "models.registry"):
        importlib.import_module(module)


def not_a_model(path: Path) -> UnreadableError:
    return UnreadableError(
        f"'{path.name}' in {path.parent} is not a model of this program, or the file is "
        "damaged. Unzip the model again (detector-*-model-*.zip)."
    )


# torch guarda los checkpoints como zip: lo que no lo es (otro archivo .pt, una copia cortada)
# no llega a torch, que lo diría con un error de pickle que no le dice nada a nadie.
def check(checkpoint_path: Path) -> None:
    if not checkpoint_path.is_file():
        raise UnreadableError(f"'{checkpoint_path}' is not there any more.")
    if not zipfile.is_zipfile(checkpoint_path):
        raise not_a_model(checkpoint_path)


def load(checkpoint_path: Path) -> "tuple[LoadedModel, str]":
    from core.runtime import resolve_device
    from models.registry import load_checkpoint

    device = resolve_device()
    if checkpoint_path not in LOADED:
        check(checkpoint_path)
        LOADED.clear()
        LOADED[checkpoint_path] = load_checkpoint(checkpoint_path, device)
    return LOADED[checkpoint_path], device


# Las clases de un checkpoint sin armar el modelo: con mmap los pesos no se leen.
def classes(checkpoint_path: Path) -> list[str]:
    if checkpoint_path in LOADED:
        return list(LOADED[checkpoint_path].labels.names)
    import torch

    check(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(checkpoint, dict) or "labels" not in checkpoint:
        raise not_a_model(checkpoint_path)
    return sorted(str(name) for name in checkpoint["labels"])


def detect(
    audio_path: Path,
    checkpoint_path: Path,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    from inference.batch import batch_size_for
    from inference.predictor import predict

    loaded, device = load(checkpoint_path)
    table = predict(
        loaded,
        audio_path,
        device,
        score_threshold=DETECT_THRESHOLD,
        batch_size=batch_size_for(device),
        on_progress=on_progress,
    )
    table.attrs["operating_point"] = loaded.operating_point
    return table
