import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

BEGIN, END, LOW, HIGH = "begin_time_s", "end_time_s", "low_freq_hz", "high_freq_hz"
BOX_COLUMNS = [BEGIN, END, LOW, HIGH]

Corners = npt.NDArray[np.float64]


def corners(table: pd.DataFrame) -> Corners:
    return table[BOX_COLUMNS].to_numpy(dtype=float)


def intersection_over_union(boxes: Corners) -> npt.NDArray[np.float64]:
    begin = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    end = np.minimum(boxes[:, None, 1], boxes[None, :, 1])
    low = np.maximum(boxes[:, None, 2], boxes[None, :, 2])
    high = np.minimum(boxes[:, None, 3], boxes[None, :, 3])

    shared = np.clip(end - begin, 0.0, None) * np.clip(high - low, 0.0, None)
    area = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
    union = area[:, None] + area[None, :] - shared
    return np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)


def single_linkage(boxes: Corners, threshold: float) -> npt.NDArray[np.int64]:
    overlapping = csr_matrix(intersection_over_union(boxes) > threshold)
    _, cluster_of_box = connected_components(overlapping, directed=False)
    return cluster_of_box
