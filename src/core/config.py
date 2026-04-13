from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Hugging Face
    HF_TOKEN: SecretStr | None = None

    # Google Drive / Data
    GOOGLE_APPLICATION_CREDENTIALS: str
    GDRIVE_ROOT_PATH: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()  # type: ignore
