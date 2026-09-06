import numpy as np
import numpy.typing as npt
import pandas as pd

from clod.cluster import BEGIN, BOX_COLUMNS, END, HIGH, LOW

NOISE = ("label", "location", "scale", "spurious", "missing")
FRACTION = 0.2  # porcion de cajas perturbadas, la del paper
SHIFT = 0.5  # desplazamiento en fracciones del lado de la caja
GROWTH = 1.5  # factor fijo de crecimiento o encogimiento


# Las cinco perturbaciones del paper, para medir si CLOD las encuentra. Devuelve la
# tabla estropeada y un registro (fila original, ruido) de lo que se toco.
def disturb(
    table: pd.DataFrame,
    kinds: tuple[str, ...] = NOISE,
    fraction: float = FRACTION,
    category: str = "Species",
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    # Las cajas se mueven en continuo: si llegan como enteros, pandas rechaza el cambio.
    table = table.reset_index(drop=True).astype(dict.fromkeys(BOX_COLUMNS, float))
    chosen = rng.choice(len(table), size=int(round(fraction * len(table))), replace=False)
    picked = {
        kind: chosen[rng.integers(0, len(kinds), len(chosen)) == i] for i, kind in enumerate(kinds)
    }

    boxes = table[BOX_COLUMNS].to_numpy()
    width, height = boxes[:, 1] - boxes[:, 0], boxes[:, 3] - boxes[:, 2]
    labels = table[category].to_numpy() if category in table.columns else np.empty(0)
    classes = np.unique(labels)
    dirty = table.copy()

    rows = picked.get("label", [])
    if len(rows) and len(classes) > 1:
        # Rotar la clase un paso no nulo la cambia siempre, y con igual probabilidad.
        step = rng.integers(1, len(classes), len(rows))
        dirty.loc[rows, category] = classes[
            (np.searchsorted(classes, labels[rows]) + step) % len(classes)
        ]

    rows = picked.get("location", [])
    if len(rows):
        # Angulo al azar y desplazamiento proporcional a los lados de la caja.
        angle = rng.uniform(0, 2 * np.pi, len(rows))
        boxes[rows, :2] += (SHIFT * width[rows] * np.cos(angle))[:, None]
        boxes[rows, 2:] += (SHIFT * height[rows] * np.sin(angle))[:, None]

    rows = picked.get("scale", [])
    if len(rows):
        factor = np.where(rng.random(len(rows)) < 0.5, GROWTH, 1 / GROWTH)[:, None]
        for axis in (slice(0, 2), slice(2, 4)):
            middle = boxes[rows, axis].mean(axis=1, keepdims=True)
            boxes[rows, axis] = middle + factor * (boxes[rows, axis] - middle)

    dirty[BOX_COLUMNS] = boxes

    rows = picked.get("spurious", [])
    if len(rows):
        # Cajas de dimensiones y clase al azar, dentro del rango que ocupa la tabla.
        extra = table.loc[rows].copy()  # copiar filas conserva los tipos de cada columna
        begin = rng.uniform(boxes[:, 0].min(), boxes[:, 1].max(), len(rows))
        low = rng.uniform(boxes[:, 2].min(), boxes[:, 3].max(), len(rows))
        extra[BEGIN], extra[END] = begin, begin + rng.uniform(0.5, 1.5, len(rows)) * width.mean()
        extra[LOW], extra[HIGH] = low, low + rng.uniform(0.5, 1.5, len(rows)) * height.mean()
        if len(classes):
            extra[category] = rng.choice(classes, len(rows))
        dirty = pd.concat([dirty, extra], ignore_index=True)

    dirty = dirty.drop(index=list(picked.get("missing", [])))
    log = pd.DataFrame(
        [{"row": int(row), "noise": kind} for kind, rows in picked.items() for row in rows],
        columns=["row", "noise"],
    )
    return dirty.reset_index(drop=True), log


# Area bajo la ROC por rangos: probabilidad de que una caja perturbada reciba peor
# calidad que una limpia. Es la metrica con la que el paper reporta CLOD.
def auroc(scores: npt.NDArray[np.float64], disturbed: npt.NDArray[np.bool_]) -> float | None:
    positives, negatives = int(disturbed.sum()), int((~disturbed).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Empates: se reparte el rango medio, si no el AUROC premia el orden arbitrario.
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    return float(
        (ranks[~disturbed].sum() - negatives * (negatives + 1) / 2) / (positives * negatives)
    )
