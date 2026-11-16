import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
WORKERS = min(8, os.cpu_count() or 1)

PROJECT_DIR = Path(__file__).resolve().parents[2]
HF_DIR = PROJECT_DIR / "hf"
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEANED_DIR = DATA_DIR / "cleaned"
UNIFIED_DIR = DATA_DIR / "unified"
PROCESSED_DIR = DATA_DIR / "processed"
YOLO_DIR = DATA_DIR / "yolo"
RUNS_DIR = PROJECT_DIR / "runs"

SCORE_THRESHOLD = 0.5
SCORE_FLOOR = 0.001
SCORE_STEP = 0.01
NMS_IOU = 0.3
MAX_DETECTIONS = 100  # por ventana


def score_grid(start: float, stop: float) -> list[float]:
    # `stop` inclusive; i / n en vez de sumar pasos evita el error de coma flotante
    n = round(1 / SCORE_STEP)
    return [i / n for i in range(round(start * n), round(stop * n) + 1)]


@dataclass
class Parameters:
    clip_len_s: float = 3.0
    clip_hop_s: float = 1.5
    min_overlap: float = 0.5  # fracción de la llamada dentro de la ventana
    pad_seed: int = 0

    target_sr: int = 44100
    n_fft: int = 4096
    win_length: int = 1024
    hop_length: int = 400

    n_mels: int = 128
    f_min: float = 25.0
    f_max: float = 22050.0
    mel_scale: str = "htk"
    mel_break_hz: float = 700.0
    mel_scale_q: float = 2595.0
    eps: float = 1e-6

    @property
    def clip_len_samples(self) -> int:
        return int(round(self.clip_len_s * self.target_sr))

    @property
    def n_frames(self) -> int:
        return self.clip_len_samples // self.hop_length + 1

    @property
    def nyquist_hz(self) -> float:
        return self.target_sr / 2


P = Parameters()
