import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import rankdata

from clod.cluster import BEGIN, BOX_COLUMNS, END, HIGH, LOW
from clod.issues import CATEGORY, LABEL, LOCATION, MISSING, SPURIOUS

SCALE = "scale"
NOISE = (LABEL, LOCATION, SPURIOUS, MISSING)
FRACTION = 0.2
SHIFT = 0.5
GROWTH = 1.5

NOTHING = np.zeros(0, dtype=np.int64)


def disturb(
    table: pd.DataFrame,
    kinds: tuple[str, ...] = NOISE,
    fraction: float = FRACTION,
    category: str = CATEGORY,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classes = np.unique(table[category].to_numpy())
    if LABEL in kinds and len(classes) < 2:
        raise ValueError(f"'{LABEL}' necesita al menos dos clases en '{category}'")

    rng = np.random.default_rng(seed)
    table = table.reset_index(drop=True).astype(dict.fromkeys(BOX_COLUMNS, float))
    chosen = rng.choice(len(table), size=int(round(fraction * len(table))), replace=False)
    assignment = rng.integers(0, len(kinds), len(chosen))
    picked = {kind: chosen[assignment == index] for index, kind in enumerate(kinds)}

    boxes = table[BOX_COLUMNS].to_numpy()
    width, height = boxes[:, 1] - boxes[:, 0], boxes[:, 3] - boxes[:, 2]
    labels = table[category].to_numpy()
    dirty = table.copy()

    rows = picked.get(LABEL, NOTHING)
    step = rng.integers(1, max(len(classes), 2), len(rows))
    dirty.loc[rows, category] = classes[
        (np.searchsorted(classes, labels[rows]) + step) % len(classes)
    ]

    rows = picked.get(LOCATION, NOTHING)
    angle = rng.uniform(0, 2 * np.pi, len(rows))
    boxes[rows, :2] += (SHIFT * width[rows] * np.cos(angle))[:, None]
    boxes[rows, 2:] += (SHIFT * height[rows] * np.sin(angle))[:, None]

    rows = picked.get(SCALE, NOTHING)
    factor = np.where(rng.random(len(rows)) < 0.5, GROWTH, 1 / GROWTH)[:, None]
    for axis in (slice(0, 2), slice(2, 4)):
        middle = boxes[rows, axis].mean(axis=1, keepdims=True)
        boxes[rows, axis] = middle + factor * (boxes[rows, axis] - middle)
    dirty[BOX_COLUMNS] = boxes

    rows = picked.get(SPURIOUS, NOTHING)
    invented = table.loc[rows].copy()
    invented.index = np.arange(len(table), len(table) + len(rows))
    begin = rng.uniform(boxes[:, 0].min(), boxes[:, 1].max(), len(rows))
    low = rng.uniform(boxes[:, 2].min(), boxes[:, 3].max(), len(rows))
    invented[BEGIN] = begin
    invented[END] = begin + rng.uniform(0.5, 1.5, len(rows)) * width.mean()
    invented[LOW] = low
    invented[HIGH] = low + rng.uniform(0.5, 1.5, len(rows)) * height.mean()
    invented[category] = rng.choice(classes, len(rows))
    picked[SPURIOUS] = invented.index.to_numpy()  #  # ty: ignore[unresolved-attribute]

    dirty = pd.concat([dirty, invented]).drop(index=list(picked.get(MISSING, NOTHING)))
    log = pd.DataFrame(
        [{"row": int(row), "noise": kind} for kind, rows in picked.items() for row in rows],
        columns=["row", "noise"],
    )
    return dirty, log


def auroc(scores: npt.NDArray[np.float64], disturbed: npt.NDArray[np.bool_]) -> float:
    positives, negatives = int(disturbed.sum()), int((~disturbed).sum())
    clean_rank_sum = rankdata(scores)[~disturbed].sum() - negatives * (negatives + 1) / 2
    return float(clean_rank_sum / (positives * negatives))
