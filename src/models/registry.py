import logging
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from data.species import LabelSet
from inference.catalog import read_operating_point
from models.base import Detector
from models.coco_deformable_detr import ResNetDeformableDETR
from models.deformable_detr import ASTDeformableDETR
from models.faster_rcnn import SpectrogramFasterRCNN
from models.yolo import SpectrogramYOLO

logger = logging.getLogger(__name__)

ARCHITECTURES: dict[str, type[Detector]] = {
    "ast_deformable_detr": ASTDeformableDETR,
    "resnet_deformable_detr": ResNetDeformableDETR,
    "faster_rcnn": SpectrogramFasterRCNN,
    "yolo": SpectrogramYOLO,
}
EXTRAS = {"ast_deformable_detr": "detr", "resnet_deformable_detr": "detr", "yolo": "yolo"}


def build_model(name: str, n_classes: int, hparams: dict[str, Any]) -> Detector:
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; known: {sorted(ARCHITECTURES)}")
    try:
        return ARCHITECTURES[name](n_classes=n_classes, **hparams)
    except ModuleNotFoundError as exc:
        if name not in EXTRAS:
            raise
        raise ModuleNotFoundError(
            f"{name!r} needs the `{EXTRAS[name]}` extra ({exc.name} is missing): "
            f"uv sync --extra {EXTRAS[name]}"
        ) from exc


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class LoadedModel(NamedTuple):
    model: Detector
    architecture: str
    labels: LabelSet
    config: dict[str, Any]
    operating_point: float | None  # None: corrida sin comparar


def load_checkpoint(path: Path | str, device: str | torch.device = "cpu") -> LoadedModel:
    path, device = Path(path), str(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "labels" not in checkpoint:
        raise ValueError(
            f"{path} is not a checkpoint of this project (use a best.pt from models/ or runs/)"
        )

    name = checkpoint["architecture"]
    labels = LabelSet(checkpoint["labels"])
    model = build_model(name, len(labels), checkpoint["hparams"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    operating_point = read_operating_point(path.parent)
    logger.info(
        "%s | %d classes | epoch %s | operating point %s | %s",
        name,
        len(labels),
        checkpoint.get("epoch"),
        "none" if operating_point is None else f"{operating_point:.2f}",
        path,
    )
    return LoadedModel(model, name, labels, checkpoint.get("config", {}), operating_point)
