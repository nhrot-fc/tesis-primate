import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from core.config import settings
from data.datasets import WindowCache
from data.manifest import ClipWindow
from data.species import LabelSet

SPLITS = ("train", "val", "test")


def split_path(split: str) -> Path:
    return settings.processed_dir / f"{split}.pt"


def read_json(name: str) -> dict:
    path = settings.processed_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"falta {path}. Corré `python src/prepare_data.py` para generar el caché."
        )
    return json.loads(path.read_text())


def meta() -> dict:
    return read_json("meta.json")


def labels() -> LabelSet:
    return LabelSet(read_json("labels.json").values())


def db_range() -> tuple[float, float]:
    low, high = meta()["db_range"]
    return float(low), float(high)


class Sources(NamedTuple):
    recordings: list[str]
    recording_of_window: list[int]


def sources_path(split: str) -> Path:
    return settings.processed_dir / f"{split}_sources.json"


def write_sources(split: str, manifest: Sequence[ClipWindow]) -> Sources:
    names: dict[str, int] = {}
    for window in manifest:
        names.setdefault(window.audio_path, len(names))
    found = Sources(list(names), [names[window.audio_path] for window in manifest])
    sources_path(split).write_text(json.dumps(found._asdict(), indent=2, ensure_ascii=False))
    return found


def sources(split: str, windows: WindowCache) -> Sources:
    path = sources_path(split)
    if not path.exists():
        raise FileNotFoundError(
            f"falta {path}. Corré `python src/prepare_data.py --sources-only`, que lo reescribe "
            "sin recalcular un solo mel."
        )

    found = Sources(**json.loads(path.read_text()))
    if len(found.recording_of_window) != len(windows.boxes):
        raise RuntimeError(
            f"{path} tiene {len(found.recording_of_window)} ventanas y {split_path(split)} tiene "
            f"{len(windows.boxes)}: son de versiones distintas de prepare_data.py. Reescribilo con "
            "`python src/prepare_data.py --sources-only`."
        )
    return found
