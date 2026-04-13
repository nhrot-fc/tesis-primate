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


class GoogleDriveService:
    def __init__(self, credentials_path: str):
        self.creds = None

        if not credentials_path:
            raise ValueError("Se debe proporcionar la ruta a las credenciales de Google Drive.")

        # Detectar el tipo de credencial (Service Account vs OAuth Client ID)
        with open(credentials_path) as f:
            cred_data = json.load(f)

        if "installed" in cred_data:
            self.creds = self._authenticate_installed_app(credentials_path)
        else:
            self.creds = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=SCOPES
            )

        self.service = build("drive", "v3", credentials=self.creds)

    def _authenticate_installed_app(self, client_secrets_file: str):
        """Maneja el flujo de autenticación para aplicaciones instaladas (OAuth 2.0)."""
        creds = None
        # El token se guarda en el mismo directorio que el secrets
        token_path = Path(client_secrets_file).parent / "token.json"

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, SCOPES)
                creds = flow.run_local_server(port=0)

            # Guardar las credenciales para la próxima ejecución
            with open(token_path, "w") as token:
                token.write(creds.to_json())

        return creds

    def _find_item_id(
        self, name: str, parent_id: str = "root", mime_type_folder: bool = False
    ) -> str | None:
        query = f"'{parent_id}' in parents and name = '{name}' and trashed = false"
        if mime_type_folder:
            query += " and mimeType = 'application/vnd.google-apps.folder'"

        try:
            results = (
                self.service.files()
                .list(q=query, spaces="drive", fields="files(id, name, mimeType)")
                .execute()
            )
            items = results.get("files", [])

            if not items:
                return None
            return items[0]["id"]
        except HttpError as error:
            print(f"An error occurred: {error}")
            return None

    def _resolve_path_to_id(self, path: str) -> str | None:
        parts = Path(path).parts
        if not parts:
            return None

        current_id = "root"

        for i, part in enumerate(parts):
            is_last = i == len(parts) - 1
            current_id = self._find_item_id(
                part, parent_id=current_id, mime_type_folder=not is_last
            )

            if not current_id:
                return None

        return current_id

    def list_files(
        self, folder_path: str | None = None, folder_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Lista los archivos contenidos en una ruta de Google Drive.
        Se puede especificar la ruta por nombre (folder_path) o el ID directo (folder_id).
        """
        if folder_id is None:
            if folder_path:
                folder_id = self._resolve_path_to_id(folder_path)
                if not folder_id:
                    raise FileNotFoundError(f"Carpeta no encontrada: {folder_path}")
            else:
                folder_id = "root"

        query = f"'{folder_id}' in parents and trashed = false"
        try:
            results = (
                self.service.files()
                .list(q=query, spaces="drive", fields="files(id, name, mimeType)")
                .execute()
            )
            return results.get("files", [])
        except HttpError as error:
            print(f"Error al listar archivos: {error}")
            return []

    def download_file(
        self,
        local_path: str | Path,
        drive_path: str | None = None,
        file_id: str | None = None,
    ) -> None:
        """
        Descarga un archivo desde Google Drive a la ruta local especificada.
        """
        if file_id is None:
            if drive_path:
                file_id = self._resolve_path_to_id(drive_path)
                if not file_id:
                    raise FileNotFoundError(
                        f"No se pudo encontrar el archivo en Google Drive: {drive_path}"
                    )
            else:
                raise ValueError("Se debe proporcionar drive_path o file_id")

        request = self.service.files().get_media(fileId=file_id)

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando en {local_path}")

        with io.FileIO(str(local_path), "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
