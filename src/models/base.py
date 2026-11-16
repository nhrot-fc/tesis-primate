import torch
from torch import Tensor, nn

from core.config import SCORE_THRESHOLD
from utils.boxes import Detections


# `forward(mel, targets)` devuelve las pérdidas; `forward(mel)`, la salida cruda
class Detector(nn.Module):
    nms_iou: float | None  # None: predicción de conjunto, sin NMS
    clip_grad: float
    needs_db_range: bool = False  # mel como imagen: hparams con db_low y db_high

    @torch.no_grad()
    def detect(self, mel: Tensor, score_threshold: float = SCORE_THRESHOLD) -> list[Detections]:
        raise NotImplementedError
