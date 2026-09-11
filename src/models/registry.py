import logging
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from data.species import LabelSet
from utils.boxes import Detections

# Corre el modelo en el modo en que esté y postprocesa. Ponerlo en `eval()` es de quien
# llama: `load_checkpoint` lo hace al cargar y `Trainer` alrededor de la validación.
Detect = Callable[[nn.Module, Tensor, float], list[Detections]]

logger = logging.getLogger(__name__)


class Architecture(NamedTuple):
    module: str  # expone la clase `model` y una función `detect(modelo, mel, umbral)`
    model: str
    clip_grad: float = 1.0
    # NMS que hay que correrle encima al salir de `detect`. `None` para el DETR: el matching
    # húngaro es uno a uno y el costo ya castiga los duplicados durante el entrenamiento, así
    # que suprimir después sólo puede borrar cajas legítimas —dos vocalizaciones simultáneas
    # se solapan de verdad en el espectrograma—. Faster R-CNN y YOLO sí traen duplicados:
    # sus cabezas son densas.
    nms_iou: float | None = 0.3


ARCHITECTURES: dict[str, Architecture] = {
    "ast_deformable_detr": Architecture(
        "models.deformable_detr", "ASTDeformableDETR", clip_grad=0.1, nms_iou=None
    ),
    "faster_rcnn": Architecture("models.faster_rcnn", "SpectrogramFasterRCNN", clip_grad=10.0),
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


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


class LoadedModel(NamedTuple):
    model: nn.Module
    architecture: str
    labels: LabelSet
    config: dict[str, Any]

    @property
    def score_threshold(self) -> float:  # punto de operación con el que se eligió el checkpoint
        return float(self.config.get("score_threshold", 0.5))

    @property
    def nms_iou(self) -> float | None:  # lo fija la arquitectura, no la corrida
        return architecture(self.architecture).nms_iou

    def detect(self, images: Tensor, score_threshold: float | None = None) -> list[Detections]:
        threshold = self.score_threshold if score_threshold is None else score_threshold
        return detector(self.architecture)(self.model, images, threshold)


def load_checkpoint(path: Path | str, device: str | torch.device = "cpu") -> LoadedModel:
    path, device = Path(path), str(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "labels" not in checkpoint:
        raise ValueError(
            f"{path} no es un checkpoint de este proyecto: no tiene 'state_dict' ni 'labels'. "
            "Los válidos son los `best.pt` que dejan las corridas en runs/<corrida>/ "
            "(`train.py --arch yolo` exporta el de Ultralytics a ese formato al terminar)."
        )

    name = checkpoint["architecture"]
    labels = LabelSet(checkpoint["labels"])
    model = build_model(name, len(labels), checkpoint["hparams"]).to(device)
    # Estricto: un checkpoint con claves de más o de menos es de otra versión del grafo, y
    # cargarlo a medias da un modelo que no protesta.
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    logger.info("%s | %d clases | época %s | %s", name, len(labels), checkpoint.get("epoch"), path)
    return LoadedModel(model, name, labels, checkpoint.get("config", {}))
