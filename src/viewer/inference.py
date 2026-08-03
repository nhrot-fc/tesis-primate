"""Inferencia dentro del propio proceso del visor: sin servidor ni contenedor."""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from infer import LoadedModel

# Un checkpoint cargado ocupa cientos de MB, asi que se reutiliza mientras no
# cambie y solo se guarda uno.
LOADED: dict[Path, "LoadedModel | Any"] = {}


def detect(
    audio_path: Path,
    checkpoint_path: Path,
    score_threshold: float = 0.05,
    nms_iou: float | None = None,
    batch_size: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Corre el modelo sobre un audio. Va en el hilo de trabajo: bloquea el suyo.

    Detecta con un umbral bajo a propósito, para que el slider de la ventana pueda
    explorar hacia abajo; el punto de operación con el que se eligió el checkpoint
    viaja en `table.attrs` y es donde arranca ese slider.
    """
    # Imports diferidos: torch tarda segundos en cargarse y la ventana debe abrir ya.
    import torch

    from infer import load_model
    from pipelines.inference_pipeline import predict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path not in LOADED:
        LOADED.clear()
        LOADED[checkpoint_path] = load_model(checkpoint_path, device)
    loaded = LOADED[checkpoint_path]

    table = predict(
        loaded.model,
        audio_path,
        loaded.labels,
        device,
        score_threshold=min(score_threshold, loaded.score_threshold),
        nms_iou=loaded.nms_iou if nms_iou is None else nms_iou,
        batch_size=batch_size,
        on_progress=on_progress,
    )
    table.attrs["operating_score_threshold"] = loaded.score_threshold
    return table
