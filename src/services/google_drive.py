import io
import json
from pathlib import Path

from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service(credentials_path: Path):
    with open(credentials_path) as file:
        cred_data = json.load(file)

    if "installed" in cred_data:
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
        creds = flow.run_local_server(port=0)
    else:
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )

    return build("drive", "v3", credentials=creds)


def download_folder_contents(service, folder_id: str, out_dir: Path):
    query = f"'{folder_id}' in parents and trashed = false"
    results = (
        service.files().list(q=query, spaces="drive", fields="files(id, name, mimeType)").execute()
    )
    files = results.get("files", [])

    if not files:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if file.get("mimeType") == "application/vnd.google-apps.folder":
            continue

        local_path = out_dir / file["name"]
        print(f"  Downloading: {file['name']} ...")

        request = service.files().get_media(fileId=file["id"])
        with io.FileIO(str(local_path), "wb") as file_handle:
            downloader = MediaIoBaseDownload(file_handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
