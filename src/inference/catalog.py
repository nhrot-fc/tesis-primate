"""Qué hay para detectar y con qué: audios, checkpoints y su umbral. Sin torch a propósito:
el visor lo consulta antes de que cargue el motor de detección."""

import json
from pathlib import Path

from core.config import PROJECT_DIR, RUNS_DIR

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}
CHECKPOINT_SUFFIXES = {".pt", ".pth"}
# Donde el paquete portable deja los checkpoints: `models/<nombre>/best.pt` (+ operating_point.json)
MODELS_DIR = PROJECT_DIR / "models"
# La tabla queda junto al audio con el mismo nombre que exporta el visor; `<audio>.txt` a
# secas es la anotación de Raven.
DETECTIONS_SUFFIX = ".detections.txt"
# Umbral elegido en val por `compare_models.py`, junto al checkpoint
OPERATING_POINT = "operating_point.json"


def is_audio(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES


def is_checkpoint(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in CHECKPOINT_SUFFIXES
        and not path.name.endswith("_predictions.pt")
    )


# Un checkpoint por carpeta: `best.pt` si está, si no el único que haya.
def checkpoint_in(directory: Path) -> Path | None:
    best = directory / "best.pt"
    if best.is_file():
        return best
    found = [p for p in directory.iterdir() if is_checkpoint(p)]
    return found[0] if len(found) == 1 else None


def available_models() -> list[Path]:
    # `models/` del paquete y, en el repo, también las corridas de `runs/`.
    found = []
    for root in (MODELS_DIR, RUNS_DIR):
        if root.is_dir():
            found.extend(checkpoint_in(d) for d in sorted(root.iterdir()) if d.is_dir())
    return [p for p in found if p is not None]


def read_operating_point(directory: Path) -> float | None:
    # -> umbral de score, o None si la corrida no pasó por `compare_models.py`
    path = directory / OPERATING_POINT
    if not path.is_file():
        return None
    threshold = json.loads(path.read_text()).get("threshold")
    return None if threshold is None else float(threshold)


def output_for(audio: Path) -> Path:
    return audio.with_name(f"{audio.stem}{DETECTIONS_SUFFIX}")


def collect_audio(paths: list[Path], recursive: bool = True) -> list[Path]:
    audio: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = path.rglob("*") if recursive else path.iterdir()
            audio.extend(p for p in sorted(found) if is_audio(p))
        elif is_audio(path):
            audio.append(path)
    return list(dict.fromkeys(audio))
