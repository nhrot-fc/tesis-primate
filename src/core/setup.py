import json
import logging
import shutil
import zipfile
from pathlib import Path

from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from core.config import settings

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def setup_project_path(project_dir: Path) -> None:
    settings.PROJECT_DIR = project_dir
    logger.info("Ruta del proyecto configurada en: %s", settings.PROJECT_DIR)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _drive_service(credentials_path: Path):
    """Cliente de Drive, con credenciales de usuario (OAuth) o de cuenta de servicio."""
    if "installed" in json.loads(credentials_path.read_text()):
        creds = InstalledAppFlow.from_client_secrets_file(
            credentials_path, SCOPES
        ).run_local_server(port=0)
    else:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
    return build("drive", "v3", credentials=creds)


def setup_data(force_override: bool = False) -> None:
    """Descarga y descomprime los zips del dataset desde la carpeta de Drive."""
    if not (settings.GOOGLE_DRIVE_FOLDER_ID and settings.GOOGLE_DRIVE_CREDENTIALS_PATH):
        return logger.warning("Faltan credenciales o el ID de la carpeta de Drive.")

    if settings.raw_dir.exists():
        if not force_override:
            return logger.info("Datos existentes en %s. Omitiendo.", settings.raw_dir)
        shutil.rmtree(settings.raw_dir)
        shutil.rmtree(settings.zip_dir, ignore_errors=True)

    settings.zip_dir.mkdir(parents=True, exist_ok=True)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    service = _drive_service(settings.GOOGLE_DRIVE_CREDENTIALS_PATH)
    query = (
        f"'{settings.GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed = false "
        "and mimeType != 'application/vnd.google-apps.folder'"
    )
    files = service.files().list(q=query, fields="files(id, name)").execute().get("files", [])

    for f in files:
        logger.info("Descargando: %s", f["name"])
        request = service.files().get_media(fileId=f["id"])
        with (settings.zip_dir / f["name"]).open("wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    for zip_file in settings.zip_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_file) as z:
                z.extractall(settings.raw_dir / zip_file.stem)
        except zipfile.BadZipFile:
            logger.error("%s no es un zip válido.", zip_file.name)
