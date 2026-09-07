import numpy as np
import numpy.typing as npt

ALPHA = 0.8

Matrix = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]


def multilabel(
    cluster_of_box: npt.NDArray[np.int64],
    class_of_box: npt.NDArray[np.int64],
    score_of_box: Matrix,
    n_classes: int,
    annotated: Mask,
) -> tuple[Mask, Matrix]:
    n_clusters = int(cluster_of_box.max(initial=-1)) + 1
    background = n_classes
    labelled = np.zeros((n_clusters, n_classes + 1), dtype=bool)
    predicted = np.zeros((n_clusters, n_classes + 1))

    np.logical_or.at(labelled, (cluster_of_box[annotated], class_of_box[annotated]), True)
    np.maximum.at(
        predicted, (cluster_of_box[~annotated], class_of_box[~annotated]), score_of_box[~annotated]
    )
    labelled[:, background] = ~labelled[:, :background].any(axis=1)
    predicted[:, background] = predicted[:, :background].sum(axis=1) == 0
    return labelled, predicted


def quality(labelled: Mask, predicted: Matrix, alpha: float = ALPHA) -> Matrix:
    self_confidence = np.where(labelled, predicted, 1.0 - predicted)
    best_to_worst = np.sort(self_confidence, axis=1)[:, ::-1]

    n_classes = best_to_worst.shape[1]
    decay = (1.0 - alpha) ** np.arange(n_classes - 1, -1, -1)
    weight = alpha * decay
    weight[0] = decay[0]
    return best_to_worst @ weight
