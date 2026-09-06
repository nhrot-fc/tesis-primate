import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

BEGIN, END, LOW, HIGH = "Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"
BOX_COLUMNS = [BEGIN, END, LOW, HIGH]

Corners = npt.NDArray[np.float64]  # (N, 4) con x0, x1, y0, y1


def corners(table: pd.DataFrame) -> Corners:
    return table[BOX_COLUMNS].to_numpy(dtype=float)


def iou(boxes: Corners) -> npt.NDArray[np.float64]:
    # Caja tiempo-frecuencia: el eje x son segundos y el y son hercios, pero el IoU
    # es adimensional y no le afecta que los ejes tengan unidades distintas.
    left = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    right = np.minimum(boxes[:, None, 1], boxes[None, :, 1])
    bottom = np.maximum(boxes[:, None, 2], boxes[None, :, 2])
    top = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    overlap = np.clip(right - left, 0.0, None) * np.clip(top - bottom, 0.0, None)
    area = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    union = area[:, None] + area[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def clusters(boxes: Corners, threshold: float) -> npt.NDArray[np.int64]:
    # El aglomerativo de enlace simple cortado a distancia 1-IoU es, exactamente, las
    # componentes conexas del grafo de cajas que se solapan mas que el umbral.
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    _, labels = connected_components(csr_matrix(iou(boxes) > threshold), directed=False)
    return labels
