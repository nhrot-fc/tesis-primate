import numpy as np
import pandas as pd

from clod.cluster import BOX_COLUMNS, corners, single_linkage
from clod.quality import multilabel, quality

IOU_THRESHOLD = 0.5
TOP = 0.05
CATEGORY, SCORE = "species", "score"

SOURCE, ANNOTATION, PREDICTION = "Origen", "anotación", "modelo"
ISSUE, QUALITY, CLUSTER = "Hallazgo", "Calidad", "Grupo"
SPURIOUS, MISSING, LOCATION, LABEL = "spurious", "missing", "location", "label"


def scan(
    annotations: pd.DataFrame,
    predictions: pd.DataFrame,
    group: str,
    iou_threshold: float = IOU_THRESHOLD,
    category: str = CATEGORY,
) -> pd.DataFrame:
    boxes = pd.concat(
        [
            annotations.assign(**{SOURCE: ANNOTATION, SCORE: 0.0}),
            predictions.assign(**{SOURCE: PREDICTION}),
        ],
        ignore_index=True,
    )
    absent = [name for name in [*BOX_COLUMNS, category, group] if name not in boxes]
    if absent:
        raise KeyError(f"faltan columnas: {absent}")
    incomplete = [name for name in (category, SCORE) if boxes[name].isna().any()]
    if incomplete:
        raise ValueError(f"hay cajas sin {incomplete}")

    annotated = (boxes[SOURCE] == ANNOTATION).to_numpy()
    class_of_box, classes = pd.factorize(boxes[category])

    within_group = np.zeros(len(boxes), dtype=np.int64)
    for _, rows in boxes.groupby(group, sort=False):
        within_group[rows.index] = single_linkage(corners(rows), iou_threshold)
    cluster_of_box = pd.factorize(pd.MultiIndex.from_arrays([boxes[group], within_group]))[0]

    labelled, predicted = multilabel(
        cluster_of_box, class_of_box, boxes[SCORE].to_numpy(), len(classes), annotated
    )
    n_clusters = len(labelled)

    has_annotation, has_prediction = np.zeros(n_clusters, bool), np.zeros(n_clusters, bool)
    np.logical_or.at(has_annotation, cluster_of_box, annotated)
    np.logical_or.at(has_prediction, cluster_of_box, ~annotated)
    class_present = np.zeros((n_clusters, len(classes)), bool)
    np.logical_or.at(class_present, (cluster_of_box, class_of_box), True)

    issue_of_cluster = np.select(
        [~has_prediction, ~has_annotation, class_present.sum(axis=1) > 1],
        [SPURIOUS, MISSING, LABEL],
        default=LOCATION,
    )

    boxes[CLUSTER] = cluster_of_box
    boxes[QUALITY] = quality(labelled, predicted)[cluster_of_box]
    boxes[ISSUE] = issue_of_cluster[cluster_of_box]

    proposal = boxes.loc[~annotated].groupby(CLUSTER)[SCORE].idxmax()
    keep = annotated | (boxes[ISSUE].eq(MISSING) & boxes.index.isin(proposal))
    return boxes[keep].sort_values(QUALITY, kind="mergesort").reset_index(drop=True)


def rank(found: pd.DataFrame, top: float = TOP, category: str = CATEGORY) -> pd.DataFrame:
    one_per_cluster = found.drop_duplicates(subset=[CLUSTER, category])
    issues = one_per_cluster[ISSUE]
    position = one_per_cluster.groupby(ISSUE).cumcount()
    quota = np.maximum(1, np.round(top * issues.map(issues.value_counts())))
    return one_per_cluster[position < quota]
