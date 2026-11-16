import json
import logging
from pathlib import Path

from core.config import PROJECT_DIR, RUNS_DIR

logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}
CHECKPOINT_SUFFIXES = {".pt", ".pth"}
MODELS_DIR = PROJECT_DIR / "models"
# <audio>.txt a secas es la anotación de Raven
DETECTIONS_SUFFIX = ".detections.txt"
OPERATING_POINT = "operating_point.json"


def is_audio(path: Path) -> bool:
    # ._x.wav: metadatos de macOS
    return (
        path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES and not path.name.startswith("._")
    )


def is_checkpoint(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in CHECKPOINT_SUFFIXES
        and not path.name.endswith("_predictions.pt")
    )


def checkpoint_in(directory: Path) -> Path | None:
    best = directory / "best.pt"
    if best.is_file():
        return best
    found = [p for p in directory.iterdir() if is_checkpoint(p)]
    return found[0] if len(found) == 1 else None


def available_models() -> list[Path]:
    found = []
    for root in (MODELS_DIR, RUNS_DIR):
        if root.is_dir():
            found.extend(checkpoint_in(d) for d in sorted(root.iterdir()) if d.is_dir())
    return [p for p in found if p is not None]


def read_operating_point(directory: Path) -> float | None:
    # None: corrida sin comparar
    path = directory / OPERATING_POINT
    if not path.is_file():
        return None
    try:
        threshold = json.loads(path.read_text()).get("threshold")
        return None if threshold is None else float(threshold)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Ignoring %s: %s", path, exc)
        return None


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
