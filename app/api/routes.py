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
from app.services.summary_service import summarize_codebase



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


class SummaryRequest(BaseModel):
    top_k: int = 8
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

@router.get("/app-ui", response_class=HTMLResponse)
def app_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Codebase Assistant UI</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 30px auto;
                padding: 20px;
                background: #f8fafc;
                color: #1f2937;
            }
            h1 {
                margin-bottom: 8px;
            }
            h2 {
                margin-top: 0;
            }
            .sub {
                color: #6b7280;
                margin-bottom: 24px;
            }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .card {
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }
            .full {
                grid-column: 1 / -1;
            }
            label {
                font-weight: 600;
                display: block;
                margin-top: 10px;
                margin-bottom: 6px;
            }
            input[type="text"],
            input[type="number"],
            textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                font-size: 14px;
                box-sizing: border-box;
            }
            textarea {
                min-height: 110px;
                resize: vertical;
            }
            input[type="file"] {
                margin-top: 8px;
            }
            button {
                margin-top: 14px;
                padding: 10px 16px;
                border: none;
                border-radius: 8px;
                background: #111827;
                color: white;
                cursor: pointer;
                font-size: 14px;
            }
            button:hover {
                background: #000000;
            }
            pre {
                background: #0f172a;
                color: #e2e8f0;
                padding: 14px;
                border-radius: 10px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-break: break-word;
                font-size: 13px;
            }
            .muted {
                color: #6b7280;
                font-size: 14px;
            }
            .workspace-box {
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                padding: 12px;
                border-radius: 8px;
                margin-top: 10px;
                word-break: break-all;
            }
            .output-box {
                background: #f9fafb;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 14px;
                margin-top: 14px;
                white-space: pre-wrap;
                word-break: break-word;
                line-height: 1.6;
            }
            .code-box {
                background: #111827;
                color: #f9fafb;
                border-radius: 10px;
                padding: 14px;
                margin-top: 14px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-break: break-word;
                font-family: Consolas, Monaco, monospace;
                font-size: 13px;
            }
            .section-title {
                margin-top: 16px;
                font-size: 14px;
                font-weight: 700;
                color: #374151;
            }
            .file-list {
                margin-top: 8px;
                padding-left: 18px;
            }
            .status-good {
                color: #065f46;
                font-weight: 600;
            }
            .status-warn {
                color: #92400e;
                font-weight: 600;
            }
            .small {
                font-size: 12px;
            }
            details {
                margin-top: 12px;
            }
            summary {
                cursor: pointer;
                font-weight: 600;
                color: #374151;
            }
        </style>
    </head>
    <body>
        <h1>AI Codebase Assistant</h1>
        <div class="sub">
            Upload code, index it, search it, ask questions, summarize architecture, and generate grounded code.
        </div>

        <div class="grid">
            <div class="card full">
                <h2>1. Upload and Index Codebase</h2>
                <p class="muted">Upload one file, multiple files, or a zip archive. The backend will save and index the uploaded content.</p>
                <input id="uploadFiles" type="file" multiple />
                <br />
                <button onclick="uploadAndIndex()">Upload and Index</button>

                <div class="workspace-box">
                    <strong>Current Workspace Path:</strong>
                    <div id="workspacePath">(not set yet)</div>
                </div>

                <div class="section-title">Upload Summary</div>
                <div id="uploadSummary" class="output-box">No upload yet.</div>

                <details>
                    <summary>Show Raw Upload Response</summary>
                    <pre id="uploadResult">No upload yet.</pre>
                </details>
            </div>

            <div class="card">
                <h2>2. Search Codebase</h2>
                <label>Query</label>
                <textarea id="searchQuery">Where is order placement implemented?</textarea>
                <label>Top K</label>
                <input id="searchTopK" type="number" value="5" />
                <button onclick="searchCodebase()">Search</button>

                <div class="section-title">Search Highlights</div>
                <div id="searchSummary" class="output-box">No search yet.</div>

                <details>
                    <summary>Show Raw Search Response</summary>
                    <pre id="searchResult">No search yet.</pre>
                </details>
            </div>

            <div class="card">
                <h2>3. Ask Codebase</h2>
                <label>Question</label>
                <textarea id="askQuestion">What does this codebase do?</textarea>
                <label>Top K</label>
                <input id="askTopK" type="number" value="5" />
                <button onclick="askCodebase()">Ask</button>

                <div class="section-title">Answer</div>
                <div id="askAnswer" class="output-box">No answer yet.</div>

                <div class="section-title">Key Files</div>
                <div id="askFiles" class="output-box">No key files yet.</div>

                <details>
                    <summary>Show Raw Ask Response</summary>
                    <pre id="askResult">No answer yet.</pre>
                </details>
            </div>

            <div class="card">
                <h2>4. Summarize Codebase</h2>
                <label>Top K</label>
                <input id="summaryTopK" type="number" value="8" />
                <button onclick="summarizeCodebase()">Summarize</button>

                <div class="section-title">Summary</div>
                <div id="summaryText" class="output-box">No summary yet.</div>

                <div class="section-title">Important Files</div>
                <div id="summaryFiles" class="output-box">No important files yet.</div>

                <details>
                    <summary>Show Raw Summary Response</summary>
                    <pre id="summaryResult">No summary yet.</pre>
                </details>
            </div>

            <div class="card">
                <h2>5. Generate Code</h2>
                <label>Task</label>
                <textarea id="generateTask">Create a helper function for modifying stoploss orders in Dhan similar to the existing order placement style</textarea>
                <label>Top K</label>
                <input id="generateTopK" type="number" value="5" />
                <button onclick="generateCode()">Generate</button>

                <div class="section-title">Suggested Target File</div>
                <div id="generateTarget" class="output-box">No target file yet.</div>

                <div class="section-title">Generated Code</div>
                <pre id="generateCodeBlock" class="code-box">No generation yet.</pre>

                <details>
                    <summary>Show Raw Generation Response</summary>
                    <pre id="generateResult">No generation yet.</pre>
                </details>
            </div>

            <div class="card full">
                <h2>6. Generate and Save Code</h2>
                <label>Task</label>
                <textarea id="saveTask">Create a helper function for modifying stoploss orders in Dhan similar to the existing order placement style</textarea>
                <label>Top K</label>
                <input id="saveTopK" type="number" value="5" />
                <label>
                    <input id="overwriteFlag" type="checkbox" />
                    Overwrite existing file
                </label>
                <br />
                <button onclick="generateAndSaveCode()">Generate and Save</button>

                <div class="section-title">Save Result</div>
                <div id="saveSummary" class="output-box">Nothing saved yet.</div>

                <details>
                    <summary>Show Raw Save Response</summary>
                    <pre id="saveResult">Nothing saved yet.</pre>
                </details>
            </div>
        </div>

        <script>
            let currentWorkspacePath = "";

            function setWorkspacePath(path) {
                currentWorkspacePath = path || "";
                document.getElementById("workspacePath").innerText = currentWorkspacePath || "(not set yet)";
            }

            function pretty(obj) {
                return JSON.stringify(obj, null, 2);
            }

            function formatList(items) {
                if (!items || !items.length) return "None";
                return items.map(x => "- " + x).join("\\n");
            }

            function formatSearchResults(results) {
                if (!results || !results.length) return "No matching chunks found.";

                return results.map((item, index) => {
                    const snippet = (item.content || "").slice(0, 250);
                    return `${index + 1}. ${item.file_path}\\n   chunk: ${item.chunk_index} | distance: ${item.distance}\\n   snippet: ${snippet}...`;
                }).join("\\n\\n");
            }

            async function uploadAndIndex() {
                const filesInput = document.getElementById("uploadFiles");
                const resultBox = document.getElementById("uploadResult");
                const summaryBox = document.getElementById("uploadSummary");

                if (!filesInput.files.length) {
                    summaryBox.textContent = "Please select at least one file.";
                    resultBox.textContent = "Please select at least one file.";
                    return;
                }

                const formData = new FormData();
                for (const file of filesInput.files) {
                    formData.append("files", file);
                }

                summaryBox.textContent = "Uploading and indexing...";
                resultBox.textContent = "Uploading and indexing...";

                try {
                    const response = await fetch("/upload-and-index", {
                        method: "POST",
                        body: formData
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);

                    if (data.upload_path) {
                        setWorkspacePath(data.upload_path);
                    }

                    summaryBox.textContent =
                        `Upload completed successfully.\\n\\n` +
                        `Files uploaded: ${data.files_uploaded ?? "-"}\\n` +
                        `Documents parsed: ${data.documents_parsed ?? "-"}\\n` +
                        `Chunks indexed: ${data.chunks_indexed ?? "-"}\\n` +
                        `Workspace: ${data.upload_path ?? "-"}`;
                } catch (err) {
                    summaryBox.textContent = "Error: " + err;
                    resultBox.textContent = "Error: " + err;
                }
            }

            async function searchCodebase() {
                const resultBox = document.getElementById("searchResult");
                const summaryBox = document.getElementById("searchSummary");
                resultBox.textContent = "Searching...";
                summaryBox.textContent = "Searching...";

                try {
                    const response = await fetch("/search-codebase", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            query: document.getElementById("searchQuery").value,
                            top_k: parseInt(document.getElementById("searchTopK").value),
                            workspace_path: currentWorkspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);
                    summaryBox.textContent = formatSearchResults(data.results || []);
                } catch (err) {
                    summaryBox.textContent = "Error: " + err;
                    resultBox.textContent = "Error: " + err;
                }
            }

            async function askCodebase() {
                const resultBox = document.getElementById("askResult");
                const answerBox = document.getElementById("askAnswer");
                const filesBox = document.getElementById("askFiles");

                resultBox.textContent = "Generating answer...";
                answerBox.textContent = "Generating answer...";
                filesBox.textContent = "Loading key files...";

                try {
                    const response = await fetch("/ask-codebase", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            question: document.getElementById("askQuestion").value,
                            top_k: parseInt(document.getElementById("askTopK").value),
                            workspace_path: currentWorkspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);
                    answerBox.textContent = data.answer || "No answer returned.";
                    filesBox.textContent = formatList(data.key_files || []);
                } catch (err) {
                    answerBox.textContent = "Error: " + err;
                    filesBox.textContent = "Error loading key files.";
                    resultBox.textContent = "Error: " + err;
                }
            }

            async function summarizeCodebase() {
                const resultBox = document.getElementById("summaryResult");
                const summaryTextBox = document.getElementById("summaryText");
                const filesBox = document.getElementById("summaryFiles");

                resultBox.textContent = "Summarizing...";
                summaryTextBox.textContent = "Summarizing...";
                filesBox.textContent = "Loading important files...";

                try {
                    const response = await fetch("/summarize-codebase", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            top_k: parseInt(document.getElementById("summaryTopK").value),
                            workspace_path: currentWorkspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);
                    summaryTextBox.textContent = data.summary || "No summary returned.";
                    filesBox.textContent = formatList(data.important_files || []);
                } catch (err) {
                    summaryTextBox.textContent = "Error: " + err;
                    filesBox.textContent = "Error loading important files.";
                    resultBox.textContent = "Error: " + err;
                }
            }

            async function generateCode() {
                const resultBox = document.getElementById("generateResult");
                const targetBox = document.getElementById("generateTarget");
                const codeBox = document.getElementById("generateCodeBlock");

                resultBox.textContent = "Generating code...";
                targetBox.textContent = "Resolving target file...";
                codeBox.textContent = "Generating code...";

                try {
                    const response = await fetch("/generate-code", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            task: document.getElementById("generateTask").value,
                            top_k: parseInt(document.getElementById("generateTopK").value),
                            workspace_path: currentWorkspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);
                    targetBox.textContent = data.target_file || "No target file returned.";
                    codeBox.textContent = data.generated_code || "No code returned.";
                } catch (err) {
                    targetBox.textContent = "Error: " + err;
                    codeBox.textContent = "Error generating code.";
                    resultBox.textContent = "Error: " + err;
                }
            }

            async function generateAndSaveCode() {
                const resultBox = document.getElementById("saveResult");
                const summaryBox = document.getElementById("saveSummary");

                resultBox.textContent = "Generating and saving...";
                summaryBox.textContent = "Generating and saving...";

                try {
                    const response = await fetch("/generate-and-save", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            task: document.getElementById("saveTask").value,
                            top_k: parseInt(document.getElementById("saveTopK").value),
                            overwrite: document.getElementById("overwriteFlag").checked,
                            workspace_path: currentWorkspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);

                    summaryBox.textContent =
                        `Status: ${data.write_status ?? "-"}\\n` +
                        `Target: ${data.final_target_file ?? "-"}\\n\\n` +
                        `${data.write_message ?? "No save message returned."}`;
                } catch (err) {
                    summaryBox.textContent = "Error: " + err;
                    resultBox.textContent = "Error: " + err;
                }
            }
        </script>
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
            "message": "Code generated successfully",
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

@router.post("/summarize-codebase")
def summarize_uploaded_codebase(request: SummaryRequest):
    try:
        workspace_id = (
            normalize_workspace_id(request.workspace_path)
            if request.workspace_path else None
        )

        matches = search_similar_chunks(
            query="overall architecture modules services routes helpers main purpose of this codebase",
            top_k=request.top_k,
            workspace_id=workspace_id
        )

        response = summarize_codebase(matches)

        return {
            "message": "Codebase summary generated successfully",
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            **response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
