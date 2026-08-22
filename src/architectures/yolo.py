"""Adaptador de un YOLO de Ultralytics al mismo contrato que el resto de detectores.

Ultralytics entrena con PNG en disco, así que acá se rehace en memoria la conversión de
`create_yolo_dataset.py`: mel de potencia -> dB con el rango global -> gris de 8 bits ->
cuadrado de `imgsz`. Ese exportador deja el grave abajo, de modo que las detecciones
vuelven con `cy -> 1 - cy` al espacio mel normalizado que usan las otras arquitecturas.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from architectures.deformable_detr import Detections
from utils.audio import mel_to_unit

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


def load_ultralytics_weights(model: SpectrogramYOLO, weights: Path) -> None:
    trained = torch.load(weights, weights_only=False)["model"]
    model.detector.load_state_dict({k: v.float() for k, v in trained.state_dict().items()})


def detect(
    model: SpectrogramYOLO, images: Tensor, score_threshold: float = 0.5
) -> list[Detections]:
    return model.detect(images, score_threshold)
