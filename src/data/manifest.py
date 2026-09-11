import logging
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import soundfile as sf

from core.config import SEED, P, Parameters
from data.species import LabelSet
from utils.audio import FloatArray, hz_to_y, window_starts

logger = logging.getLogger(__name__)

IntArray = npt.NDArray[np.int64]
MIN_BOX_SIZE = 1e-3


class ClipWindow(NamedTuple):
    audio_path: str
    clip_start_s: float
    boxes: FloatArray  # (N, 4) cxcywh normalizado al clip
    labels: IntArray  # (N,) id de clase en el `LabelSet`


def duration_of(audio_path: str) -> float | None:
    try:
        return sf.info(audio_path).duration
    except (RuntimeError, sf.LibsndfileError):
        return None


def boxes_in_window(
    group: pd.DataFrame, class_ids: IntArray, clip_start_s: float, params: Parameters
) -> tuple[FloatArray, IntArray, bool]:
    # -> (cajas, clases, si quedó afuera alguna llamada que sí pisaba la ventana)
    begin = group["begin_time_s"].to_numpy()
    end = group["end_time_s"].to_numpy()
    overlap = np.minimum(end, clip_start_s + params.clip_len_s) - np.maximum(begin, clip_start_s)
    visible = np.minimum(end - begin, params.clip_len_s)
    present = overlap > 0
    keep = present & (overlap >= params.min_overlap * visible)
    if not keep.any():
        return np.empty((0, 4)), np.empty(0, dtype=np.int64), bool(present.any())

    x0 = np.clip((begin[keep] - clip_start_s) / params.clip_len_s, 0.0, 1.0)
    x1 = np.clip((end[keep] - clip_start_s) / params.clip_len_s, 0.0, 1.0)
    y0 = np.clip(hz_to_y(group["low_freq_hz"].to_numpy()[keep], params), 0.0, 1.0)
    y1 = np.clip(hz_to_y(group["high_freq_hz"].to_numpy()[keep], params), 0.0, 1.0)
    boxes = np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis=-1)
    usable = (boxes[:, 2] >= MIN_BOX_SIZE) & (boxes[:, 3] >= MIN_BOX_SIZE)
    return boxes[usable], class_ids[keep][usable], bool(usable.sum() < present.sum())


def build_manifest(
    df: pd.DataFrame,
    labels: LabelSet,
    params: Parameters = P,
    empty_ratio: float = 0.0,
    seed: int = SEED,
) -> list[ClipWindow]:
    if not 0.0 <= empty_ratio < 1.0:
        raise ValueError(f"empty_ratio debe estar en [0, 1): {empty_ratio}")
    unknown = {name for name in df["label"].unique() if name not in labels}
    if unknown:
        raise ValueError(f"etiquetas fuera del LabelSet: {sorted(unknown)}")

    positive: list[ClipWindow] = []
    empty: list[ClipWindow] = []
    unreadable: list[str] = []
    for audio_path, group in df.groupby("audio_path"):
        duration_s = duration_of(str(audio_path))
        if duration_s is None:
            unreadable.append(str(audio_path))
            continue
        class_ids = group["label"].map(labels.id).to_numpy(dtype=np.int64)
        for clip_start_s in window_starts(duration_s, params):
            boxes, ids, incomplete = boxes_in_window(group, class_ids, float(clip_start_s), params)
            window = ClipWindow(str(audio_path), float(clip_start_s), boxes, ids)
            if len(boxes):
                positive.append(window)
            elif not incomplete:  # con media llamada afuera no es vacía
                empty.append(window)

    if unreadable:
        logger.warning(
            "%d audios ilegibles, quedan fuera:\n  %s", len(unreadable), "\n  ".join(unreadable)
        )
    if empty_ratio <= 0.0 or not empty:
        return positive
    n_empty = min(len(empty), round(len(positive) * empty_ratio / (1 - empty_ratio)))
    keep = np.random.default_rng(seed).choice(len(empty), size=n_empty, replace=False)
    return positive + [empty[i] for i in keep]


def event_windows(df: pd.DataFrame, params: Parameters = P) -> list[ClipWindow]:
    # Ventanas sin cajas de las grabaciones de fondo, sólo las que pisan un evento anotado.
    windows: list[ClipWindow] = []
    no_boxes, no_labels = np.empty((0, 4)), np.empty(0, dtype=np.int64)
    for audio_path, group in df.groupby("audio_path"):
        duration_s = duration_of(str(audio_path))
        if duration_s is None:
            logger.warning("audio de fondo ilegible, queda fuera: %s", audio_path)
            continue
        begin, end = group["begin_time_s"].to_numpy(), group["end_time_s"].to_numpy()
        for clip_start_s in window_starts(duration_s, params):
            overlap = np.minimum(end, clip_start_s + params.clip_len_s) - np.maximum(
                begin, clip_start_s
            )
            if (overlap > 0).any():
                windows.append(
                    ClipWindow(str(audio_path), float(clip_start_s), no_boxes, no_labels)
                )
    return windows


def sample_windows(windows: list[ClipWindow], n: int, seed: int = SEED) -> list[ClipWindow]:
    keep = np.random.default_rng(seed).choice(
        len(windows), size=min(len(windows), n), replace=False
    )
    return [windows[i] for i in sorted(keep)]


def split_manifest(
    manifest: list[ClipWindow],
    n_classes: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = SEED,
) -> tuple[list[ClipWindow], list[ClipWindow], list[ClipWindow]]:
    # Por grabación, estratificado por clase: cada archivo va entero a un split.
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"los ratios deben sumar 1.0: {ratios} suma {sum(ratios)}")

    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_classes))
    windows_by_file: dict[str, list[ClipWindow]] = defaultdict(list)
    for window in manifest:
        windows_by_file[window.audio_path].append(window)
        for class_id in window.labels:
            counts[window.audio_path][class_id] += 1

    files = sorted(windows_by_file)
    np.random.default_rng(seed).shuffle(files)
    total = np.sum([counts[f] for f in files], axis=0)  # (n_classes,)
    target = np.outer(ratios, total)  # (3, n_classes)
    current = np.zeros((3, n_classes))
    n_windows = np.zeros(3)
    target_windows = np.array(ratios) * len(manifest)
    assigned: dict[str, int] = {}

    # De la clase más rara a la más común, y los archivos con más cajas de esa clase primero.
    for class_id in np.argsort(total):
        pending = [f for f in files if f not in assigned and counts[f][class_id] > 0]
        pending.sort(key=lambda f: -counts[f][class_id])
        for file in pending:
            deficit = target[:, class_id] - current[:, class_id]
            best = np.flatnonzero(deficit == deficit.max())
            best = best[np.argmax((target_windows - n_windows)[best])] if len(best) > 1 else best[0]
            assigned[file] = int(best)
            current[best] += counts[file]
            n_windows[best] += len(windows_by_file[file])

    # Sin cajas de ninguna clase: van donde falten ventanas
    for file in files:
        if file not in assigned:
            best = int(np.argmax(target_windows - n_windows))
            assigned[file] = best
            n_windows[best] += len(windows_by_file[file])

    splits: list[list[ClipWindow]] = [[], [], []]
    for file, split_index in assigned.items():
        splits[split_index].extend(windows_by_file[file])
    return splits[0], splits[1], splits[2]
