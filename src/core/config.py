import os
from dataclasses import dataclass
from pathlib import Path

SEED = 42
# Workers del DataLoader
WORKERS = min(8, os.cpu_count() or 1)

# Raíz del repo
PROJECT_DIR = Path(__file__).resolve().parents[2]
# Copia local del AST
HF_DIR = PROJECT_DIR / "hf"
DATA_DIR = PROJECT_DIR / "data"
# Audio y tablas de Raven
RAW_DIR = DATA_DIR / "raw"
# Anotaciones normalizadas, una por grabación
CLEANED_DIR = DATA_DIR / "cleaned"
# Caché de ventanas
PROCESSED_DIR = DATA_DIR / "processed"
# Export PNG para Ultralytics
YOLO_DIR = DATA_DIR / "yolo"
RUNS_DIR = PROJECT_DIR / "runs"


@dataclass
class Parameters:
    # Ventanas
    clip_len_s: float = 3.0
    clip_hop_s: float = 1.5
    # Fracción de la llamada que debe caer en la ventana
    min_overlap: float = 0.5
    pad_seed: int = 0

    # STFT
    target_sr: int = 44100
    n_fft: int = 4096
    win_length: int = 1024
    hop_length: int = 400

    # Mel
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


P = Parameters()
