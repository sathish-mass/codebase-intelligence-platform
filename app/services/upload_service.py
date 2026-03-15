import shutil
import uuid
import zipfile
from pathlib import Path
from typing import List

from fastapi import UploadFile


UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)


def save_uploaded_files(files: List[UploadFile]) -> Path:
    """
    Save uploaded files into a unique session folder.
    If a zip file is uploaded, extract it inside the same session folder.
    Returns the folder path to scan.
    """
    session_id = str(uuid.uuid4())
    session_dir = UPLOAD_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if not file.filename:
            continue

        destination = session_dir / file.filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # If uploaded file is a zip, extract it
        if destination.suffix.lower() == ".zip":
            extract_dir = session_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(destination, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

    return session_dir