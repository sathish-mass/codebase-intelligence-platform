import zipfile
from typing import Annotated, List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.services.parser import parse_codebase
from app.services.upload_service import save_uploaded_files
from app.services.vector_store import index_documents, get_collection_stats

from app.services.vector_store import (
    index_documents,
    get_collection_stats,
    search_similar_chunks,
    normalize_workspace_id,
)

from app.services.answer_service import build_grounded_answer
from app.services.generation_service import build_generation_output
from app.services.file_writer import write_generated_code




router = APIRouter()


class ScanRequest(BaseModel):
    path: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    workspace_path: str | None = None

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    workspace_path: str | None = None

class GenerateRequest(BaseModel):
    task: str
    top_k: int = 5
    workspace_path: str | None = None

class GenerateAndSaveRequest(BaseModel):
    task: str
    top_k: int = 5
    overwrite: bool = False
    workspace_path: str | None = None

@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/scan-codebase")
def scan_codebase(request: ScanRequest):
    try:
        documents = parse_codebase(request.path)

        return {
            "message": "Codebase scanned successfully",
            "path": request.path,
            "files_found": len(documents),
            "sample_files": [doc["path"] for doc in documents[:10]]
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/upload-codebase")
async def upload_codebase(
    files: Annotated[
        List[UploadFile],
        File(description="Upload one or more code files or a zip file")
    ]
):
    try:
        session_dir = save_uploaded_files(files)
        documents = parse_codebase(str(session_dir))

        return {
            "message": "Files uploaded and scanned successfully",
            "upload_path": str(session_dir),
            "files_uploaded": len(files),
            "files_found": len(documents),
            "sample_files": [doc["path"] for doc in documents[:10]]
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file uploaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/upload-ui", response_class=HTMLResponse)
def upload_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Codebase Upload UI</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 700px;
                margin: 40px auto;
                padding: 20px;
            }
            h2 {
                margin-bottom: 10px;
            }
            form {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                background: #fafafa;
            }
            input, button {
                margin-top: 12px;
                font-size: 16px;
            }
            button {
                padding: 10px 16px;
                cursor: pointer;
            }
            .note {
                color: #555;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <h2>Upload Codebase</h2>
        <form action="/upload-codebase" enctype="multipart/form-data" method="post">
            <input name="files" type="file" multiple />
            <br />
            <button type="submit">Upload and Scan</button>
        </form>
        <p class="note">
            You can upload one file, multiple code files, or a zip file.
        </p>
    </body>
    </html>
    """

@router.post("/index-codebase")
def index_codebase(request: ScanRequest):
    try:
        documents = parse_codebase(request.path)
        workspace_id = normalize_workspace_id(request.path)
        stats = index_documents(documents, workspace_id=workspace_id)

        return {
            "message": "Codebase indexed successfully",
            "path": request.path,
            **stats
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/index-stats")
def index_stats():
    try:
        return get_collection_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    


@router.post("/search-codebase")
def search_codebase(request: SearchRequest):
    try:
        workspace_id = (
            normalize_workspace_id(request.workspace_path)
            if request.workspace_path else None
        )

        matches = search_similar_chunks(
            query=request.query,
            top_k=request.top_k,
            workspace_id=workspace_id
        )

        return {
            "message": "Search completed successfully",
            "query": request.query,
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            "matches_found": len(matches),
            "results": matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

@router.post("/ask-codebase")
def ask_codebase(request: AskRequest):
    try:
        workspace_id = (
            normalize_workspace_id(request.workspace_path)
            if request.workspace_path else None
        )

        matches = search_similar_chunks(
            query=request.question,
            top_k=request.top_k,
            workspace_id=workspace_id
        )

        response = build_grounded_answer(
            question=request.question,
            matches=matches
        )

        return {
            "message": "Answer generated successfully",
            "question": request.question,
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            **response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

@router.post("/upload-and-index")
async def upload_and_index(
    files: Annotated[
        List[UploadFile],
        File(description="Upload code files, documentation files, or a zip file")
    ]
):
    try:
        session_dir = save_uploaded_files(files)
        documents = parse_codebase(str(session_dir))
        workspace_id = normalize_workspace_id(str(session_dir))
        stats = index_documents(documents, workspace_id=workspace_id)

        return {
            "message": "Files uploaded and indexed successfully",
            "upload_path": str(session_dir),
            "workspace_id": workspace_id,
            "files_uploaded": len(files),
            "documents_parsed": len(documents),
            "sample_files": [doc["path"] for doc in documents[:10]],
            **stats
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file uploaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/generate-code")
def generate_code(request: GenerateRequest):
    try:
        workspace_id = (
            normalize_workspace_id(request.workspace_path)
            if request.workspace_path else None
        )

        matches = search_similar_chunks(
            query=request.task,
            top_k=request.top_k,
            workspace_id=workspace_id
        )

        response = build_generation_output(
            task=request.task,
            matches=matches
        )

        return {
            "message": "Code skeleton generated successfully",
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            **response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/generate-and-save")
def generate_and_save(request: GenerateAndSaveRequest):
    try:
        workspace_id = (
            normalize_workspace_id(request.workspace_path)
            if request.workspace_path else None
        )

        matches = search_similar_chunks(
            query=request.task,
            top_k=request.top_k,
            workspace_id=workspace_id
        )

        generation = build_generation_output(
            task=request.task,
            matches=matches
        )

        write_result = write_generated_code(
            target_file=generation["target_file"],
            generated_code=generation["generated_code"],
            overwrite=request.overwrite,
            workspace_path=request.workspace_path
        )

        return {
            "message": "Code generated and file operation completed",
            "task": request.task,
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            "generation_type": generation["generation_type"],
            "suggested_relative_target": generation["target_file"],
            "final_target_file": write_result["target_file"],
            "write_status": write_result["status"],
            "write_message": write_result["message"],
            "references": generation["references"],
            "notes": generation["notes"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

