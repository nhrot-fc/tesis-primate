import logging
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from data.datasets import FasterRCNNDataset, SpectrogramDataset
from data.species import LabelSet
from utils.boxes import Detections

Detect = Callable[[nn.Module, Tensor, float], list[Detections]]

logger = logging.getLogger(__name__)


class Architecture(NamedTuple):
    module: str  # expone la clase `model` y una función `detect(modelo, mel, umbral)`
    model: str
    dataset: type[SpectrogramDataset] = SpectrogramDataset  # formato de entrada que come
    clip_grad: float = 1.0
    provided: tuple[str, ...] = ()  # claves que el modelo repone solo al construirse


ARCHITECTURES: dict[str, Architecture] = {
    "ast_deformable_detr": Architecture(
        "models.deformable_detr",
        "ASTDeformableDETR",
        clip_grad=0.1,
        # el AST congelado se lee de la copia local y el criterio se rearma solo
        provided=("backbone.model.", "criterion."),
    ),
    "eat_dino": Architecture(
        "models.dino",
        "EATDINO",
        clip_grad=0.1,
        # el EAT congelado se lee de la copia local y el criterio se rearma solo
        provided=("backbone.model.", "criterion."),
    ),
    "faster_rcnn": Architecture(
        "models.faster_rcnn", "SpectrogramFasterRCNN", FasterRCNNDataset, clip_grad=10.0
    ),
    "yolo": Architecture("models.yolo", "SpectrogramYOLO"),
}


def architecture(name: str) -> Architecture:
    if name not in ARCHITECTURES:
        raise ValueError(f"arquitectura desconocida: {name!r}; hay {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name]


def detector(name: str) -> Detect:
    return import_module(architecture(name).module).detect


def build_model(name: str, n_classes: int, hparams: dict[str, Any]) -> nn.Module:
    spec = architecture(name)
    return getattr(import_module(spec.module), spec.model)(n_classes=n_classes, **hparams)


def savable_state_dict(model: nn.Module, name: str) -> dict[str, Tensor]:
    """Todos los pesos menos los que la arquitectura repone sola al construirse.

    No sirve filtrar sólo por `requires_grad`: las capas congeladas de Faster R-CNN son
    pesos de COCO que nadie vuelve a poner, y sin ellas el checkpoint carga medio ResNet
    al azar. La excepción va al revés: dentro de un prefijo `provided`, lo que sí recibió
    gradiente ya no es el preentrenado y hay que guardarlo --si no, una corrida con el
    AST descongelado guardaría un checkpoint que al cargarse vuelve a los pesos de
    AudioSet y pierde el fine-tuning entero.
    """
    provided = architecture(name).provided
    trained = {key for key, param in model.named_parameters() if param.requires_grad}
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(provided) or key in trained
    }


def load_state_dict(model: nn.Module, name: str, state_dict: dict[str, Tensor]) -> None:
    """Faltar claves de `provided` es normal; cualquier otra ausencia o sobra es un
    checkpoint incompatible, y cargarlo a medias da un modelo que no protesta."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    provided = architecture(name).provided
    broken = list(unexpected) + [key for key in missing if not key.startswith(provided)]
    if broken:
        raise RuntimeError(f"checkpoint incompatible con {name}: {sorted(broken)}")


class LoadedModel(NamedTuple):
    model: nn.Module
    architecture: str
    labels: LabelSet
    score_threshold: float  # punto de operación con el que se eligió este checkpoint
    nms_iou: float
    config: dict

    def detect(self, images: Tensor, score_threshold: float | None = None) -> list[Detections]:
        threshold = self.score_threshold if score_threshold is None else score_threshold
        return detector(self.architecture)(self.model, images, threshold)


def operating_point(config: dict[str, Any]) -> tuple[float, float]:
    return float(config.get("score_threshold", 0.5)), float(config.get("nms_iou", 0.3))


def _load_ultralytics(path: Path, checkpoint: Any, device: str) -> LoadedModel:
    """El `best.pt` que deja Ultralytics, que no salió de acá pero es el resultado natural
    de `src/train_yolo.py`; cualquier otro se rechaza diciendo qué archivo hace falta."""
    yolo = import_module("models.yolo")
    if not yolo.is_ultralytics_checkpoint(checkpoint):
        raise ValueError(
            f"{path} no es un checkpoint de este proyecto: no tiene 'state_dict' ni 'labels'. "
            "Los válidos son los `best.pt` que dejan las corridas en runs/<corrida>/."
        )
    adapted = yolo.load_ultralytics_checkpoint(path, checkpoint, device)
    adapted.model.eval()
    logger.info("yolo (Ultralytics) | %d clases | %s", len(adapted.labels), path)
    return LoadedModel(
        adapted.model, "yolo", adapted.labels, *operating_point(adapted.config), adapted.config
    )


def load_checkpoint(path: Path | str, device: str | torch.device = "cpu") -> LoadedModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "labels" not in checkpoint:
        return _load_ultralytics(Path(path), checkpoint, str(device))

    name = checkpoint["architecture"]
    labels = LabelSet(checkpoint["labels"])
    config = checkpoint.get("config", {})

    model = build_model(name, len(labels), checkpoint["hparams"]).to(device)
    load_state_dict(model, name, checkpoint["state_dict"])
    model.eval()

    logger.info("%s | %d clases | época %s | %s", name, len(labels), checkpoint.get("epoch"), path)
    return LoadedModel(model, name, labels, *operating_point(config), config)
