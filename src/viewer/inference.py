from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from models.registry import LoadedModel

# Un checkpoint cargado ocupa cientos de MB, asi que se reutiliza mientras no
# cambie y solo se guarda uno.
LOADED: dict[Path, "LoadedModel | Any"] = {}


def preload() -> None:
    import inference.predictor  # noqa: F401
    from models.registry import ARCHITECTURES, architecture

    for name in ARCHITECTURES:
        import_module(architecture(name).module)


def detect(
    audio_path: Path,
    checkpoint_path: Path,
    score_threshold: float = 0.05,
    nms_iou: float | None = None,
    batch_size: int = 2,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    import torch

    from inference.predictor import predict
    from models.registry import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path not in LOADED:
        LOADED.clear()
        LOADED[checkpoint_path] = load_checkpoint(checkpoint_path, device)
    loaded = LOADED[checkpoint_path]

    table = predict(
        loaded,
        audio_path,
        device,
        score_threshold=min(score_threshold, loaded.score_threshold),
        nms_iou=loaded.nms_iou if nms_iou is None else nms_iou,
        batch_size=batch_size,
        on_progress=on_progress,
    )
    table.attrs["operating_score_threshold"] = loaded.score_threshold
    return table
