import io
import json
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def build_drive_service(credentials_path: Path):
    if not credentials_path:
        raise ValueError(
            "Se debe proporcionar la ruta a las credenciales de Google Drive."
        )

    with open(credentials_path, "r") as file:
        cred_data = json.load(file)

    if "installed" in cred_data:
        creds = _authenticate_installed_app(credentials_path)
    else:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )

    return build("drive", "v3", credentials=creds)


def download_all_from_path(
    service,
    drive_folder_path: str,
    local_dir: Path,
) -> list[Path]:
    folder_id = _resolve_path_to_id(service, drive_folder_path)
    if not folder_id:
        raise FileNotFoundError(f"Carpeta no encontrada: {drive_folder_path}")

    query = f"'{folder_id}' in parents and trashed = false"
    try:
        results = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name, mimeType)")
            .execute()
        )
        files: list[dict[str, Any]] = results.get("files", [])
    except HttpError as error:
        raise RuntimeError(f"Error al listar archivos: {error}") from error

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: list[Path] = []
    for file in files:
        if file.get("mimeType") == "application/vnd.google-apps.folder":
            continue

        local_path = local_dir / file["name"]
        _download_file_by_id(service, file["id"], local_path)
        downloaded_files.append(local_path)

    return downloaded_files


def _authenticate_installed_app(client_secrets_file: Path):
    creds = None
    token_path = client_secrets_file.parent / "token.json"

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return creds


def _find_item_id(
    service,
    name: str,
    parent_id: str = "root",
    mime_type_folder: bool = False,
) -> str | None:
    query = f"'{parent_id}' in parents and name = '{name}' and trashed = false"
    if mime_type_folder:
        query += " and mimeType = 'application/vnd.google-apps.folder'"

    try:
        results = (
            service.files()
            .list(q=query, spaces="drive", fields="files(id, name, mimeType)")
            .execute()
        )
        items = results.get("files", [])
        if not items:
            return None
        return items[0]["id"]
    except HttpError as error:
        raise RuntimeError(f"Error resolviendo elemento de Drive: {error}") from error


def _resolve_path_to_id(service, path: str) -> str | None:
    parts = Path(path).parts
    if not parts:
        return None

    current_id = "root"
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        current_id = _find_item_id(
            service, part, parent_id=current_id, mime_type_folder=not is_last
        )
        if not current_id:
            return None

    return current_id


def _download_file_by_id(service, file_id: str, local_path: Path) -> None:
    request = service.files().get_media(fileId=file_id)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando en {local_path}")

    with io.FileIO(str(local_path), "wb") as file_handle:
        downloader = MediaIoBaseDownload(file_handle, request)
        done = False
        while done is False:
            _, done = downloader.next_chunk()
