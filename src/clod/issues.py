import numpy as np
import pandas as pd

from clod.cluster import clusters, corners
from clod.quality import quality, reduce

IOU = 0.5  # umbral de agrupamiento, el que el paper reporta como mejor
CATEGORY = "Species"
SINGLE_CLASS = "objeto"  # deteccion agnostica: todo cae en una sola clase

SOURCE, ANNOTATION, PREDICTION = "Origen", "anotación", "modelo"
ISSUE, QUALITY = "Hallazgo", "Calidad"

# Los cuatro errores del paper, en el orden en que se deciden.
ISSUES = {
    "spurious": "caja anotada que el modelo no ve",
    "missing": "caja que el modelo ve y no está anotada",
    "location": "caja mal ubicada",
    "label": "etiqueta cambiada",
}


def scan(
    annotations: pd.DataFrame,
    predictions: pd.DataFrame,
    iou: float = IOU,
    threshold: float = 1.0,
    top_n: int | None = None,
    group: str | None = None,
    category: str = CATEGORY,
) -> pd.DataFrame:
    boxes = pd.concat(
        [annotations.assign(**{SOURCE: ANNOTATION}), predictions.assign(**{SOURCE: PREDICTION})],
        ignore_index=True,
    )
    annotated = (boxes[SOURCE] == ANNOTATION).to_numpy()
    scores = (
        boxes["Score"].to_numpy(dtype=float)
        if "Score" in boxes.columns
        else np.full(len(boxes), np.nan)
    )
    scores = np.where(annotated, np.nan, scores)
    names = (
        boxes[category] if category in boxes.columns else pd.Series(SINGLE_CLASS, index=boxes.index)
    )

    # El paper exige que las clases predichas esten entre las anotadas; aqui basta con
    # darle columna propia a cada clase, que da lo mismo cuando esa condicion se cumple.
    classes, categories = np.unique(names.to_numpy(dtype=str), return_inverse=True)

    # 1. Agrupamiento por IoU, cada grabacion por separado.
    keys = boxes[group] if group is not None else pd.Series(0, index=boxes.index)
    groups = np.zeros(len(boxes), dtype=np.int64)
    offset = 0
    for _, rows in boxes.groupby(keys, sort=False):
        local = clusters(corners(rows), iou)
        groups[rows.index] = local + offset
        offset += int(local.max()) + 1 if len(local) else 0

    # 2. Reduccion a multietiqueta y calidad por grupo (confident learning).
    truth, probs = reduce(groups, categories, scores, len(classes))
    scored = quality(truth, probs)

    # 3. Que clase de error es cada grupo. Un grupo con anotacion y prediccion esta mal
    # ubicado si todas sus cajas comparten clase, y mal etiquetado si no.
    n = len(scored)
    with_annotations, with_predictions = np.zeros(n, bool), np.zeros(n, bool)
    np.logical_or.at(with_annotations, groups, annotated)
    np.logical_or.at(with_predictions, groups, ~annotated)
    present = np.zeros((n, len(classes)), bool)
    np.logical_or.at(present, (groups, categories), True)
    issue = np.where(
        ~with_predictions,
        "spurious",
        np.where(
            ~with_annotations, "missing", np.where(present.sum(axis=1) <= 1, "location", "label")
        ),
    )

    # 4. De cada grupo sospechoso se marca la caja que hay que corregir: la anotacion,
    # salvo cuando falta, que ahi lo unico que existe es la prediccion.
    boxes[QUALITY] = scored[groups]
    boxes[ISSUE] = issue[groups]
    keep = (boxes[QUALITY] <= threshold) & np.where(
        issue[groups] == "missing", ~annotated, annotated
    )
    found = boxes.loc[keep].sort_values(QUALITY, kind="mergesort").reset_index(drop=True)
    return found if top_n is None else found.head(top_n)
