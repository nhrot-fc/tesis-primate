"""Adaptador de un YOLO de Ultralytics al mismo contrato que el resto de detectores.

Ultralytics entrena con PNG en disco, así que acá se rehace en memoria la conversión de
`create_yolo_dataset.py`: mel de potencia -> dB con el rango global -> gris de 8 bits ->
cuadrado de `imgsz`. Ese exportador deja el grave abajo, de modo que las detecciones
vuelven con `cy -> 1 - cy` al espacio mel normalizado que usan las otras arquitecturas.
"""

import json
import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import Tensor, nn

from architectures.deformable_detr import Detections
from core.config import settings
from domain.species import LabelSet
from utils.audio import mel_to_unit

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo26s"
DEFAULT_IMAGE_SIZE = 512


class SpectrogramYOLO(nn.Module):
    _yolo: Any

    def __init__(
        self,
        n_classes: int,
        db_low: float,
        db_high: float,
        model: str = DEFAULT_MODEL,
        imgsz: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel

        self.db_low, self.db_high = db_low, db_high
        self.imgsz = imgsz
        self.detector = DetectionModel(f"{model}.yaml", nc=n_classes, verbose=False)

        # El `YOLO` de Ultralytics hereda de `nn.Module` pero su `train()` no es el de
        # PyTorch: lanza un entrenamiento. Registrado como submódulo, un `model.eval()`
        # --que llama a `train(False)` en cada hijo-- dispara una corrida sobre COCO8 y
        # pisa los pesos. Por eso vive fuera del árbol de módulos y sólo sirve para
        # `predict`; el submódulo entrenable es `self.detector`.
        wrapper = YOLO(f"{model}.yaml")
        wrapper.model = self.detector
        self.__dict__["_yolo"] = wrapper

    def to_images(self, mel: Tensor) -> list[np.ndarray]:
        import cv2

        unit = mel_to_unit(mel.detach().cpu(), self.db_low, self.db_high)
        images = []
        for gray in (unit[:, 0] * 255).to(torch.uint8).numpy():
            resized = cv2.resize(
                np.flipud(gray), (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
            )
            images.append(cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR))
        return images

    @torch.no_grad()
    def detect(self, mel: Tensor, score_threshold: float = 0.5) -> list[Detections]:
        results = self._yolo.predict(
            self.to_images(mel),
            imgsz=self.imgsz,
            conf=max(score_threshold, 1e-4),
            device=mel.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = torch.as_tensor(result.boxes.xywhn).float().reshape(-1, 4).cpu()
            boxes[:, 1] = 1.0 - boxes[:, 1]
            scores = torch.as_tensor(result.boxes.conf).float().reshape(-1).cpu()
            labels = torch.as_tensor(result.boxes.cls).long().reshape(-1).cpu()
            order = scores.argsort(descending=True)
            detections.append(Detections(boxes[order], scores[order], labels[order]))
        return detections


def load_ultralytics_state(model: SpectrogramYOLO, checkpoint: dict) -> None:
    trained = checkpoint["model"]
    model.detector.load_state_dict({k: v.float() for k, v in trained.state_dict().items()})


def load_ultralytics_weights(model: SpectrogramYOLO, weights: Path) -> None:
    load_ultralytics_state(model, torch.load(weights, weights_only=False))


# --- Cargar el `best.pt` de Ultralytics tal cual ---------------------------------
# El registro guarda arquitectura, hiperparámetros y `state_dict`; Ultralytics guarda
# el modelo pickleado y sus `train_args`. Lo que le falta para entrar al viewer no está
# en el `.pt` --el rango de dB y los nombres con barra-- sino en el `meta.json` del
# export, así que se lo busca ahí en vez de pedirle al usuario que corra la reexportación.
ULTRALYTICS_MARKERS = ("model", "train_args")


class AdaptedYOLO(NamedTuple):
    model: SpectrogramYOLO
    labels: LabelSet
    hparams: dict[str, Any]
    config: dict[str, Any]


def is_ultralytics_checkpoint(checkpoint: object) -> bool:
    return isinstance(checkpoint, dict) and all(key in checkpoint for key in ULTRALYTICS_MARKERS)


def _dataset_meta(checkpoint: dict, weights: Path) -> tuple[Path, dict]:
    """`meta.json` del export: primero donde se entrenó, si no, el del proyecto."""
    candidates = []
    data = (checkpoint.get("train_args") or {}).get("data")
    if data:
        candidates.append(Path(data).parent / "meta.json")
    candidates.append(settings.data_dir / "yolo" / "meta.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text())
    raise FileNotFoundError(
        f"{weights} es un checkpoint de Ultralytics, pero no encuentro el `meta.json` del "
        f"export de YOLO (busqué en {', '.join(str(c) for c in candidates)}). Sin él no se "
        "sabe con qué rango de dB se generaron las imágenes ni cómo se llamaban las clases. "
        "Corré `python src/create_yolo_dataset.py`, o cargá el `best_spectrogram.pth` que "
        "`train_yolo.py` deja al lado del `best.pt`."
    )


def load_ultralytics_checkpoint(weights: Path, checkpoint: dict, device: str) -> AdaptedYOLO:
    meta_path, meta = _dataset_meta(checkpoint, weights)
    labels = LabelSet(meta["names_original"])

    trained_classes = int(getattr(checkpoint["model"], "nc", len(labels)))
    if trained_classes != len(labels):
        raise ValueError(
            f"{weights} se entrenó con {trained_classes} clases y {meta_path} describe "
            f"{len(labels)}: el export se regeneró después de entrenar. Volvé a correr "
            "`python src/create_yolo_dataset.py` y `python src/train_yolo.py`."
        )

    train_args = checkpoint.get("train_args") or {}
    hparams: dict[str, Any] = {
        "model": Path(str(train_args.get("model", DEFAULT_MODEL))).stem,
        "imgsz": meta["image_size"],
        "db_low": meta["db_range"]["low"],
        "db_high": meta["db_range"]["high"],
    }
    model = SpectrogramYOLO(n_classes=len(labels), **hparams)
    load_ultralytics_state(model, checkpoint)

    # `train_yolo.py` deja el punto de operación en `spectrogram_config.json`, un nivel
    # arriba de `weights/`. Si no está, valen los mismos valores por defecto del registro.
    config_path = weights.parent.parent / "spectrogram_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    logger.info(
        "checkpoint de Ultralytics adaptado con %s%s",
        meta_path,
        "" if config else " (sin spectrogram_config.json: punto de operación por defecto)",
    )
    return AdaptedYOLO(model.to(device), labels, hparams, config)


def detect(
    model: SpectrogramYOLO, images: Tensor, score_threshold: float = 0.5
) -> list[Detections]:
    return model.detect(images, score_threshold)
