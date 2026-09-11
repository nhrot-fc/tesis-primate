import torch
from torch import Tensor, nn

from utils.boxes import Detections


# `forward(mel, targets)` devuelve las pérdidas; `forward(mel)`, la salida cruda
class Detector(nn.Module):
    # NMS a la salida de `detect`; `None` para predicción de conjunto
    nms_iou: float | None
    clip_grad: float

    @torch.no_grad()
    def detect(self, mel: Tensor, score_threshold: float = 0.5) -> list[Detections]:
        raise NotImplementedError
