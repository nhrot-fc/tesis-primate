import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from architectures.deformable_detr import Detections
from domain.species import LabelSet

logger = logging.getLogger(__name__)


class Architecture(NamedTuple):
    module: str  # expone la clase `model` y una función `detect(modelo, mel, umbral)`
    model: str
    provided: tuple[str, ...] = ()  # claves que el modelo repone solo al construirse


ARCHITECTURES: dict[str, Architecture] = {
    "ast_deformable_detr": Architecture(
        "architectures.deformable_detr",
        "ASTDeformableDETR",
        provided=("backbone.model.",),  # el AST congelado se lee de la copia local
    ),
    "faster_rcnn": Architecture("architectures.faster_rcnn", "SpectrogramFasterRCNN"),
    "yolo": Architecture("architectures.yolo", "SpectrogramYOLO"),
}

LEGACY_ARCHITECTURE = "ast_deformable_detr"
LEGACY_HPARAMS = ("dim", "n_queries", "n_levels", "n_frames", "time_stride")


def architecture(name: str) -> Architecture:
    if name not in ARCHITECTURES:
        raise ValueError(f"arquitectura desconocida: {name!r}; hay {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name]


class LoadedModel(NamedTuple):
    model: nn.Module
    architecture: str
    labels: LabelSet
    score_threshold: float  # punto de operación con el que se eligió este checkpoint
    nms_iou: float
    config: dict

    def detect(self, images: Tensor, score_threshold: float | None = None) -> list[Detections]:
        threshold = self.score_threshold if score_threshold is None else score_threshold
        module = import_module(architecture(self.architecture).module)
        return module.detect(self.model, images, threshold)


def build_model(name: str, n_classes: int, hparams: dict[str, Any], **extra: Any) -> nn.Module:
    spec = architecture(name)
    return getattr(import_module(spec.module), spec.model)(n_classes=n_classes, **hparams, **extra)


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


def save_checkpoint(
    path: Path,
    architecture: str,
    model: nn.Module,
    hparams: dict[str, Any],
    labels: LabelSet,
    config: dict[str, Any],
    **extra: Any,
) -> None:
    torch.save(
        {
            "architecture": architecture,
            "hparams": hparams,
            "state_dict": savable_state_dict(model, architecture),
            "labels": labels.names,
            "config": config,
            **extra,
        },
        path,
    )


def operating_point(config: dict[str, Any]) -> tuple[float, float]:
    return (
        float(config.get("operating_score_threshold", 0.5)),
        float(config.get("nms_iou", 0.3)),
    )


def load_foreign_checkpoint(path: Path, checkpoint: Any, device: str) -> LoadedModel:
    """Un `.pt` que no salió de `save_checkpoint`.

    El único que sabemos adaptar es el `best.pt` de Ultralytics, que es el archivo que
    `train_yolo.py` deja como resultado natural de una corrida; cualquier otro se
    rechaza diciendo qué archivo hace falta, en vez de reventar con un `KeyError`.
    """
    yolo = import_module("architectures.yolo")
    if yolo.is_ultralytics_checkpoint(checkpoint):
        adapted = yolo.load_ultralytics_checkpoint(Path(path), checkpoint, str(device))
        adapted.model.eval()
        score_threshold, nms_iou = operating_point(adapted.config)
        logger.info("yolo (Ultralytics) | %d clases | %s", len(adapted.labels), path)
        return LoadedModel(
            model=adapted.model,
            architecture="yolo",
            labels=adapted.labels,
            score_threshold=score_threshold,
            nms_iou=nms_iou,
            config=adapted.config,
        )
    raise ValueError(
        f"{path} no es un checkpoint de este proyecto: no tiene 'state_dict' ni 'labels'. "
        f"Los checkpoints válidos son los que dejan `train.py` y `train_frcnn.py` en "
        f"checkpoints/, y para YOLO el `best.pt` o el `best_spectrogram.pth` de "
        f"runs/yolo/<corrida>/weights/."
    )


def load_checkpoint(path: Path | str, device: str | torch.device = "cpu") -> LoadedModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "labels" not in checkpoint:
        return load_foreign_checkpoint(Path(path), checkpoint, str(device))
    name = checkpoint.get("architecture", LEGACY_ARCHITECTURE)
    # antes del registro los hiperparámetros iban sueltos en la raíz del checkpoint
    hparams = checkpoint.get("hparams") or {
        key: checkpoint[key] for key in LEGACY_HPARAMS if checkpoint.get(key) is not None
    }
    labels = LabelSet(checkpoint["labels"])
    config = checkpoint.get("config", {})

    model = build_model(name, len(labels), hparams).to(device)

    # Faltar claves de `provided` es normal; cualquier otra ausencia o sobra es un
    # checkpoint incompatible, y cargarlo a medias da un modelo que no protesta.
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    provided = architecture(name).provided
    broken = list(unexpected) + [key for key in missing if not key.startswith(provided)]
    if broken:
        raise RuntimeError(f"checkpoint incompatible con {name}: {sorted(broken)}")

    model.eval()
    score_threshold, nms_iou = operating_point(config)
    logger.info("%s | %d clases | %s", name, len(labels), path)
    return LoadedModel(
        model=model,
        architecture=name,
        labels=labels,
        score_threshold=score_threshold,
        nms_iou=nms_iou,
        config=config,
    )


def save_labels_json(labels: LabelSet, path: Path) -> None:
    path.write_text(json.dumps(dict(enumerate(labels.names)), indent=2, ensure_ascii=False))
