from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Hugging Face
    HF_TOKEN: SecretStr | None = None

    # Data
    GDRIVE_ROOT_PATH: str = ""
    PROJECT_DIR: Path | None = None
    SECRETS_DIR: Path | None = None
    DATA_DIR: Path | None = None
    DATA_ZIP_DIR: Path | None = None
    DATA_RAW_DIR: Path | None = None
    DATA_PREPROCESSED_DIR: Path | None = None

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "ERROR"

    # Env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def update_path_settings(project_dir: Path) -> None:
    settings.PROJECT_DIR = project_dir
    settings.SECRETS_DIR = project_dir / "secrets"
    settings.DATA_DIR = project_dir / "data"
    settings.DATA_ZIP_DIR = settings.DATA_DIR / "zip"
    settings.DATA_RAW_DIR = settings.DATA_DIR / "raw"
    settings.DATA_PREPROCESSED_DIR = settings.DATA_DIR / "preprocessed"


settings = Settings()  # type: ignore
