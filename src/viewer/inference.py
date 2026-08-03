"""Inferencia dentro del propio proceso del visor: sin servidor ni contenedor."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

# Un checkpoint cargado ocupa cientos de MB, asi que se reutiliza mientras no
# cambie y solo se guarda uno.
LOADED: dict[Path, tuple] = {}


def detect(
    audio_path: Path,
    checkpoint_path: Path,
    score_threshold: float = 0.05,
    nms_iou: float = 0.3,
    batch_size: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Corre el modelo sobre un audio. Va en el hilo de trabajo: bloquea el suyo."""
    # Imports diferidos: torch tarda segundos en cargarse y la ventana debe abrir ya.
    import torch

    from infer import load_model
    from pipelines.inference_pipeline import predict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path not in LOADED:
        LOADED.clear()
        LOADED[checkpoint_path] = load_model(checkpoint_path, device)
    model, labels = LOADED[checkpoint_path]

    return predict(
        model,
        audio_path,
        labels,
        device,
        score_threshold=score_threshold,
        nms_iou=nms_iou,
        batch_size=batch_size,
        on_progress=on_progress,
    )
