import torch
from torch import Tensor, nn

from core.config import SCORE_THRESHOLD
from utils.boxes import Detections


# `forward(mel, targets)` devuelve las pérdidas; `forward(mel)`, la salida cruda
class Detector(nn.Module):
    # NMS a la salida de `detect`; `None` para predicción de conjunto
    nms_iou: float | None
    clip_grad: float
    # Pinta el mel como imagen: sus `hparams` llevan `db_low` y `db_high` del caché
    needs_db_range: bool = False

    @torch.no_grad()
    def detect(self, mel: Tensor, score_threshold: float = SCORE_THRESHOLD) -> list[Detections]:
        raise NotImplementedError
