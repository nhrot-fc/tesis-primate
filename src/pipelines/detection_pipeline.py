"""Barrido de un detector sobre un split: cajas predichas y anotadas, listas de medir.

Recibe una función de detección, no un modelo, así que los tres detectores pasan por el
mismo pos-proceso antes de que `pipelines.metrics` los compare.
"""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import box_convert
from tqdm.auto import tqdm

from architectures.deformable_detr import Detections
from architectures.iou import suppress_nested
from pipelines.metrics import Boxes, concat, sort_by_score

Detect = Callable[[Tensor, float], list[Detections]]

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
    detect: Detect,
    loader: DataLoader,
    device: str | torch.device = "cpu",
    nms_iou: float | None = 0.3,
    min_score: float = MIN_SCORE,
    max_detections: int | None = MAX_DETECTIONS,
    desc: str = "detectando",
) -> tuple[Boxes, Boxes]:
    """-> (predicciones ordenadas por score descendente, verdad de terreno)."""
    predicted: list[Boxes] = []
    truth: list[Boxes] = []
    image_id = 0

    for images, targets in tqdm(loader, desc=desc, unit="batch", leave=False):
        detections = detect(images.to(device), min_score)
        for detection, target in zip(detections, targets, strict=True):
            boxes = detection.boxes.cpu()
            scores = detection.scores.cpu()
            labels = detection.labels.cpu()
            if nms_iou is not None and len(boxes):
                # el DETR lo necesita (sus queries se pisan entre sí) y en los demás
                # saca cajas anidadas, que acá son duplicados de una misma llamada
                keep = suppress_nested(
                    box_convert(boxes, "cxcywh", "xyxy"), scores, labels, nms_iou
                )
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if max_detections is not None and len(boxes) > max_detections:
                keep = scores.topk(max_detections).indices
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            predicted.append(Boxes(boxes, torch.full((len(boxes),), image_id), labels, scores))
            n_target = len(target["labels"])
            truth.append(
                Boxes(
                    target["boxes"],
                    torch.full((n_target,), image_id),
                    target["labels"],
                    torch.ones(n_target),
                )
            )
            image_id += 1

    return sort_by_score(concat(predicted)), concat(truth)
