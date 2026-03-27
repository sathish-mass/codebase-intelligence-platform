import shutil
import uuid
import zipfile
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import UploadFile

from app.services.parser import parse_codebase
from app.services.project_catalog import add_project_to_catalog
from app.services.project_profile_service import (
    aggregate_project_metadata,
    build_catalog_tags_from_project_metadata,
)
from app.services.vector_store import index_documents, normalize_workspace_id

UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)


def derive_workspace_name(files: List[UploadFile], session_id: str) -> str:
    """
    Create a user-friendly workspace/project name.
    Rules:
    - one normal file -> use file name
    - one zip file -> use zip file stem
    - multiple files -> use first file stem + file count
    - fallback -> use session id
    """
    valid_names = [f.filename for f in files if f.filename]

    if not valid_names:
        return f"project-{session_id[:8]}"

    if len(valid_names) == 1:
        name = Path(valid_names[0]).name
        if name.lower().endswith(".zip"):
            return Path(name).stem
        return name

    first_name = Path(valid_names[0]).stem
    return f"{first_name} + {len(valid_names) - 1} more files"


def save_uploaded_files(files: List[UploadFile], tags: Optional[List[str]] = None) -> Dict:
    """
    Save files, parse them, index them once, enrich project metadata,
    and register the project in the catalog.
    """
    session_id = str(uuid.uuid4())
    session_dir = (UPLOAD_ROOT / session_id).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    saved_files = 0

    for file in files:
        if not file.filename:
            continue

        destination = session_dir / file.filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files += 1

        if destination.suffix.lower() == ".zip":
            extract_dir = session_dir / "extracted"
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(destination, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

    workspace_path = str(session_dir)
    workspace_id = normalize_workspace_id(workspace_path)
    workspace_name = derive_workspace_name(files, session_id)

    documents = parse_codebase(workspace_path)

    for doc in documents:
        doc["project_name"] = workspace_name

    index_stats = index_documents(
        documents=documents,
        workspace_id=workspace_id,
        project_name=workspace_name,
        replace_existing=True,
    )

    project_profile = aggregate_project_metadata(
        documents=documents,
        workspace_name=workspace_name,
    )

    catalog_tags = sorted(
        set((tags or []) + build_catalog_tags_from_project_metadata(project_profile))
    )

    add_project_to_catalog(
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        tags=catalog_tags,
        workspace_path=workspace_path,
        metadata={
            "source": "upload",
            "files_uploaded": saved_files,
            "files_found": len(documents),
            **project_profile,
        },
        files_indexed=index_stats["files_indexed"],
        chunks_indexed=index_stats["chunks_indexed"],
    )

    return {
        "session_dir": session_dir,
        "workspace_path": workspace_path,
        "workspace_name": workspace_name,
        "workspace_id": workspace_id,
        "files_uploaded": saved_files,
        "files_found": len(documents),
        "sample_files": [doc["path"] for doc in documents[:10]],
        "index_stats": index_stats,
        "project_profile": project_profile,
    }