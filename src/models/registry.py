import logging
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from data.species import LabelSet
from evaluation.protocol import read_operating_point
from models.base import Detector
from models.coco_deformable_detr import CocoDeformableDETR
from models.deformable_detr import ASTDeformableDETR
from models.faster_rcnn import SpectrogramFasterRCNN
from models.yolo import SpectrogramYOLO

logger = logging.getLogger(__name__)

# Cada una se rearma con `n_classes` más los `hparams` del checkpoint.
ARCHITECTURES: dict[str, type[Detector]] = {
    "ast_deformable_detr": ASTDeformableDETR,
    "coco_deformable_detr": CocoDeformableDETR,
    "faster_rcnn": SpectrogramFasterRCNN,
    "yolo": SpectrogramYOLO,
}
# Extra de `pyproject.toml` que trae las librerías de cada arquitectura; el Faster R-CNN va de base.
EXTRAS = {"ast_deformable_detr": "detr", "coco_deformable_detr": "detr", "yolo": "yolo"}


def build_model(name: str, n_classes: int, hparams: dict[str, Any]) -> Detector:
    if name not in ARCHITECTURES:
        raise ValueError(f"arquitectura desconocida: {name!r}; hay {sorted(ARCHITECTURES)}")
    try:
        return ARCHITECTURES[name](n_classes=n_classes, **hparams)
    except ModuleNotFoundError as exc:
        if name not in EXTRAS:
            raise
        raise ModuleNotFoundError(
            f"{name!r} necesita el extra `{EXTRAS[name]}` (falta {exc.name}): "
            f"uv sync --extra {EXTRAS[name]}"
        ) from exc


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class LoadedModel(NamedTuple):
    model: Detector
    architecture: str
    labels: LabelSet
    config: dict[str, Any]
    # Umbral elegido en val por `compare_models.py`; None si la corrida no se comparó
    operating_point: float | None


def load_checkpoint(path: Path | str, device: str | torch.device = "cpu") -> LoadedModel:
    path, device = Path(path), str(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "labels" not in checkpoint:
        raise ValueError(
            f"{path} no es un checkpoint del proyecto (los válidos son runs/*/best.pt)"
        )

    name = checkpoint["architecture"]
    labels = LabelSet(checkpoint["labels"])
    model = build_model(name, len(labels), checkpoint["hparams"]).to(device)
    # Estricto: otra versión del grafo no carga a medias
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    operating_point = read_operating_point(path.parent)
    logger.info(
        "%s | %d clases | época %s | umbral %s | %s",
        name,
        len(labels),
        checkpoint.get("epoch"),
        "sin comparar" if operating_point is None else f"{operating_point:.2f}",
        path,
    )
    return LoadedModel(model, name, labels, checkpoint.get("config", {}), operating_point)
