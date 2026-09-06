from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SEED = 42


class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"

    HF_TOKEN: SecretStr | None = None
    PROJECT_DIR: Path = Path.cwd()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def hf_dir(self) -> Path:
        return self.PROJECT_DIR / "hf"

    @property
    def data_dir(self) -> Path:
        return self.PROJECT_DIR / "data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def yolo_dir(self) -> Path:
        return self.data_dir / "yolo"

    @property
    def runs_dir(self) -> Path:
        return self.PROJECT_DIR / "runs"


@dataclass
class Parameters:
    # Clips
    clip_len_s: float = 3.0
    clip_hop_s: float = 1.5
    min_overlap: float = 0.5
    pad_seed: int = 0

    # STFT
    target_sr: int = 44100
    n_fft: int = 4096
    win_length: int = 1024
    hop_length: int = 400

    # Mel spectrogram
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
settings = Settings()
