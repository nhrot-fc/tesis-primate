from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from tqdm.auto import tqdm

from evaluation.metrics import (
    BETA,
    Boxes,
    DetectionMetrics,
    concat,
    detection_metrics,
    sort_by_score,
)
from utils.boxes import Detections, suppress_nested

# El `detect` del registro con el modelo ya atado: sólo quedan imágenes y umbral.
BoundDetect = Callable[[Tensor, float], list[Detections]]

# Umbral bajo al detectar y filtro después: así el AP ve toda la cola de la curva y el
# punto de operación se puede mover sin volver a correr el modelo.
MIN_SCORE = 0.001
# El DETR no puede emitir más cajas que sus queries (64), mientras torchvision corta en
# 100 y Ultralytics en 300. Sin un techo común, los dos últimos consiguen una cola de AP
# más larga por diseño y no por mérito. Ninguna ventana tiene más de 9 cajas anotadas,
# así que recortar acá no pierde nada real.
MAX_DETECTIONS = 64


@torch.no_grad()
def collect_detections(
    detect: BoundDetect,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    nms_iou: float | None = 0.3,
    desc: str = "detectando",
) -> tuple[Boxes, Boxes]:
    # -> (predicciones ordenadas por score descendente, verdad de terreno).
    predicted: list[Boxes] = []
    truth: list[Boxes] = []
    image_id = 0

    for images, targets in tqdm(loader, desc=desc, unit="batch", leave=False):
        detections = detect(images.to(device), MIN_SCORE)
        for detection, target in zip(detections, targets, strict=True):
            boxes, scores, labels = (t.cpu() for t in detection)
            if nms_iou is not None and len(boxes):
                # el DETR lo necesita (sus queries se pisan entre sí) y en los demás
                # saca cajas anidadas, que acá son duplicados de una misma llamada
                keep = suppress_nested(
                    box_convert(boxes, "cxcywh", "xyxy"), scores, labels, nms_iou
                )
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if len(boxes) > MAX_DETECTIONS:
                keep = scores.topk(MAX_DETECTIONS).indices
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            predicted.append(Boxes(boxes, torch.full((len(boxes),), image_id), labels, scores))
            n_target = len(target["labels"])
            truth.append(
                Boxes(
                    target["boxes"].cpu(),
                    torch.full((n_target,), image_id),
                    target["labels"].cpu(),
                    torch.ones(n_target),
                )
            )
            image_id += 1

    return sort_by_score(concat(predicted)), concat(truth)


def evaluate(
    detect: BoundDetect,
    loader: DataLoader,
    n_classes: int,
    device: str | torch.device = "cpu",
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    nms_iou: float | None = 0.3,
    beta: float = BETA,
    desc: str = "val",
) -> DetectionMetrics:
    predictions, truth = collect_detections(detect, loader, device, nms_iou, desc)
    return detection_metrics(
        predictions,
        truth,
        n_classes=n_classes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        beta=beta,
    )
