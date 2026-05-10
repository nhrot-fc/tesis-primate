import zipfile

from core.config import settings
from services.google_drive import download_folder_contents, get_drive_service


def setup_data() -> None:
    assert settings.DATA_DIR and settings.DATA_DIR.is_dir(), (
        f"DATA_DIR is not a valid directory: {settings.DATA_DIR}"
    )
    assert settings.DATA_RAW_DIR and settings.DATA_RAW_DIR.is_dir(), (
        f"DATA_RAW_DIR is not a valid directory: {settings.DATA_RAW_DIR}"
    )
    assert settings.DATA_ZIP_DIR and settings.DATA_ZIP_DIR.is_dir(), (
        f"DATA_ZIP_DIR is not a valid directory: {settings.DATA_ZIP_DIR}"
    )
    assert settings.SECRETS_DIR and settings.SECRETS_DIR.is_dir(), (
        f"SECRETS_DIR is not a valid directory: {settings.SECRETS_DIR}"
    )
    assert settings.GOOGLE_DRIVE_FOLDER_ID, "GOOGLE_DRIVE_FOLDER_ID is not set."

    if settings.data_loaded:
        print("Data already loaded. Skipping download and extraction.")
        return
    settings.data_loaded = True

    service = get_drive_service(settings.SECRETS_DIR / "google_credentials.json")
    download_folder_contents(service, str(settings.GOOGLE_DRIVE_FOLDER_ID), settings.DATA_ZIP_DIR)

    for zip_file in settings.DATA_ZIP_DIR.glob(pattern="*.zip"):
        settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(settings.DATA_RAW_DIR / zip_file.stem)
        except zipfile.BadZipFile:
            print(f"Error: {zip_file} is not a valid zip file.")
