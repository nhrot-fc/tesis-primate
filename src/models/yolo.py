from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from core.config import NMS_IOU, SCORE_FLOOR, SCORE_THRESHOLD
from models.base import Detector
from utils.audio import mel_to_gray
from utils.boxes import Detections

DEFAULT_MODEL = "yolo26s"
# Lado de la imagen cuadrada: el del export (`export_yolo.py`) y el de `imgsz` al entrenar
IMAGE_SIZE = 512
# RT-DETR usa otra tarea de Ultralytics (decodificador de conjuntos, no cabeza densa) y por
# tanto otra clase de modelo y otro envoltorio; lo demás del flujo es idéntico al de YOLO.
RTDETR_PREFIX = "rtdetr"
RTDETR_MODEL = "rtdetr-l"


class SpectrogramYOLO(Detector):
    # YOLO26 sale sin NMS; el pos-proceso de anidadas se aplica igual que al resto
    nms_iou = NMS_IOU
    clip_grad = 1.0
    needs_db_range = True
    ultralytics_predictor: Any

    def __init__(
        self,
        n_classes: int,
        db_low: float,
        db_high: float,
        model: str = DEFAULT_MODEL,
        imgsz: int = IMAGE_SIZE,
    ) -> None:
        super().__init__()
        from ultralytics import RTDETR, YOLO
        from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel

        is_rtdetr = model.startswith(RTDETR_PREFIX)
        task, api = (
            (RTDETRDetectionModel, RTDETR) if is_rtdetr else (DetectionModel, YOLO)
        )

        self.db_low, self.db_high = db_low, db_high
        self.imgsz = imgsz
        self.detector = task(f"{model}.yaml", nc=n_classes, verbose=False)

        # `YOLO.train()` entrena en vez de cambiar de modo: se lo deja fuera del árbol de módulos.
        wrapper = api(f"{model}.yaml")
        wrapper.model = self.detector
        self.__dict__["ultralytics_predictor"] = wrapper

    def to_images(self, mel: Tensor) -> list[np.ndarray]:
        import cv2

        mel = mel.detach().cpu()
        return [
            cv2.cvtColor(
                mel_to_gray(mel[index, 0], self.db_low, self.db_high, self.imgsz),
                cv2.COLOR_GRAY2BGR,
            )
            for index in range(len(mel))
        ]

    @torch.no_grad()
    def detect(self, mel: Tensor, score_threshold: float = SCORE_THRESHOLD) -> list[Detections]:
        results = self.ultralytics_predictor.predict(
            self.to_images(mel),
            imgsz=self.imgsz,
            conf=max(score_threshold, SCORE_FLOOR),  # Ultralytics no acepta 0
            device=mel.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = torch.as_tensor(result.boxes.xywhn).float().reshape(-1, 4).cpu()
            boxes[:, 1] = 1.0 - boxes[:, 1]  # la imagen va con el grave abajo
            scores = torch.as_tensor(result.boxes.conf).float().reshape(-1).cpu()
            labels = torch.as_tensor(result.boxes.cls).long().reshape(-1).cpu()
            by_score = scores.argsort(descending=True)
            detections.append(Detections(boxes[by_score], scores[by_score], labels[by_score]))
        return detections


def load_ultralytics_weights(model: SpectrogramYOLO, weights: Path) -> None:
    # Ultralytics guarda el modelo pickleado (en fp16); acá sólo interesan sus pesos.
    trained = torch.load(weights, weights_only=False)["model"]
    model.detector.load_state_dict({k: v.float() for k, v in trained.state_dict().items()})
