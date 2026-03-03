from src.core.config import settings
from src.services.google_drive import build_drive_service, download_all_from_path


def setup_data() -> None:
    if (
        settings.DATA_DIR is None
        or settings.SECRETS_DIR is None
        or settings.DATA_ZIP_DIR is None
    ):
        raise ValueError(
            "DATA_DIR, DATA_ZIP_DIR or SECRETS_DIR is not set. Please call update_path_settings() first."
        )

    service = build_drive_service(settings.SECRETS_DIR / "google_credentials.json")
    download_all_from_path(service, settings.GDRIVE_ROOT_PATH, settings.DATA_ZIP_DIR)
