import logging
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from data.species import LabelSet
from models.base import Detector
from models.deformable_detr import ASTDeformableDETR
from models.faster_rcnn import SpectrogramFasterRCNN
from models.yolo import SpectrogramYOLO

logger = logging.getLogger(__name__)

# Cada una se rearma con `n_classes` más los `hparams` del checkpoint.
ARCHITECTURES: dict[str, type[Detector]] = {
    "ast_deformable_detr": ASTDeformableDETR,
    "faster_rcnn": SpectrogramFasterRCNN,
    "yolo": SpectrogramYOLO,
}


def build_model(name: str, n_classes: int, hparams: dict[str, Any]) -> Detector:
    if name not in ARCHITECTURES:
        raise ValueError(f"arquitectura desconocida: {name!r}; hay {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name](n_classes=n_classes, **hparams)


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class LoadedModel(NamedTuple):
    model: Detector
    architecture: str
    labels: LabelSet
    config: dict[str, Any]

    @property
    def score_threshold(self) -> float:
        return float(self.config.get("score_threshold", 0.5))


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
    logger.info("%s | %d clases | época %s | %s", name, len(labels), checkpoint.get("epoch"), path)
    return LoadedModel(model, name, labels, checkpoint.get("config", {}))
