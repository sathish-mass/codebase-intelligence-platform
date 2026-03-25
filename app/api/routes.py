import zipfile
from pathlib import Path
from typing import Annotated, List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.services.parser import parse_codebase
from app.services.upload_service import save_uploaded_files
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
from app.services.retrieval_router import (
    routed_search_results,
    build_project_routing_plan,
)
from app.services.task_router import build_task_routing
from app.services.project_catalog import get_all_projects, add_project_to_catalog


router = APIRouter()



class ScanRequest(BaseModel):
    path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    workspace_path: str | None = None
    workspace_id: str | None = None
    search_scope: str | None = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    workspace_path: str | None = None
    workspace_id: str | None = None
    search_scope: str | None = None


class GenerateRequest(BaseModel):
    task: str
    top_k: int = 5
    workspace_path: str | None = None
    workspace_id: str | None = None
    search_scope: str | None = None


class GenerateAndSaveRequest(BaseModel):
    task: str
    top_k: int = 5
    overwrite: bool = False
    workspace_path: str | None = None
    workspace_id: str | None = None
    search_scope: str | None = None


class SummaryRequest(BaseModel):
    top_k: int = 8
    workspace_path: str | None = None
    workspace_id: str | None = None
    search_scope: str | None = None


class AIAssistRequest(BaseModel):
    prompt: str
    top_k: int = 5
    workspace_path: str | None = None
    workspace_id: str | None = None
    search_scope: str | None = None


def resolve_request_workspace_id(
    workspace_id: str | None = None,
    workspace_path: str | None = None,
) -> str | None:
    """
    Step 26 canonical workspace identity resolver.

    Priority:
    1. explicit workspace_id
    2. normalized workspace_path
    3. None
    """
    if workspace_id:
        return workspace_id.strip()

    if workspace_path:
        return normalize_workspace_id(workspace_path)

    return None

def dedupe_matches(matches: List[dict]) -> List[dict]:
    """
    Remove duplicate retrieved matches after multi-project search merge.
    """
    seen = set()
    unique_matches = []

    for match in matches:
        key = (
            match.get("workspace_id"),
            match.get("file_path"),
            match.get("chunk_index"),
            match.get("symbol_name"),
            match.get("symbol_type"),
            match.get("parent_symbol"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_matches.append(match)

    return unique_matches


def run_catalog_routed_search(
    *,
    query: str,
    top_k: int,
    workspace_id: str | None = None,
    search_scope: str | None = None,
) -> dict:
    """
    Step 27:
    1. load project catalog
    2. choose candidate workspaces
    3. search each candidate workspace
    4. merge
    5. rerank globally
    """
    projects = get_all_projects()

    routing_plan = build_project_routing_plan(
        projects=projects,
        query=query,
        active_workspace_id=workspace_id,
        search_mode_override=search_scope,
        project_limit=4,
    )

    all_matches = []

    searched_workspace_ids = routing_plan.get("searched_workspace_ids", [])

    if searched_workspace_ids:
        per_project_top_k = max(top_k, 4)

        for candidate_workspace_id in searched_workspace_ids:
            project_matches = search_similar_chunks(
                query=query,
                top_k=per_project_top_k,
                workspace_id=candidate_workspace_id,
            )
            all_matches.extend(project_matches)

    else:
        # final fallback if catalog has no candidates
        all_matches = search_similar_chunks(
            query=query,
            top_k=max(top_k, 5),
            workspace_id=workspace_id,
        )

    all_matches = dedupe_matches(all_matches)

    routed = routed_search_results(
        matches=all_matches,
        query=query,
        preferences=routing_plan["preferences"],
    )

    routed["matches"] = routed["matches"][:top_k]

    return {
        "project_routing": routing_plan,
        "routing_preferences": routed["preferences"],
        "matches": routed["matches"],
    }

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
        upload_info = save_uploaded_files(files)

        return {
            "message": "Files uploaded and indexed successfully",
            "upload_path": upload_info["workspace_path"],
            "workspace_path": upload_info["workspace_path"],
            "workspace_id": upload_info["workspace_id"],
            "workspace_name": upload_info["workspace_name"],
            "files_uploaded": upload_info["files_uploaded"],
            "files_found": upload_info["files_found"],
            "sample_files": upload_info["sample_files"],
            **upload_info["index_stats"],
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
            h1 { margin-bottom: 8px; }
            h2 { margin-top: 0; }
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
            .full { grid-column: 1 / -1; }
            label {
                font-weight: 600;
                display: block;
                margin-top: 10px;
                margin-bottom: 6px;
            }
            input[type="number"], textarea {
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
            input[type="file"] { margin-top: 8px; }
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
            button:hover { background: #000000; }
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
                word-break: break-word;
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
            .small { font-size: 12px; }
            details { margin-top: 12px; }
            summary {
                cursor: pointer;
                font-weight: 600;
                color: #374151;
            }
            .status-good {
                color: #065f46;
                font-weight: 600;
            }
            .status-warn {
                color: #92400e;
                font-weight: 600;
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
                    <strong>Current Project:</strong>
                    <div id="workspaceNameText">(not set yet)</div>

                    <details>
                        <summary>Technical Details</summary>
                        <div class="small muted" id="workspacePathText">(not set yet)</div>
                        <div class="small muted" id="workspaceIdText">(not set yet)</div>
                    </details>
                </div>

                <div class="section-title">Upload Summary</div>
                <div id="uploadSummary" class="output-box">No upload yet.</div>

                <details>
                    <summary>Show Raw Upload Response</summary>
                    <pre id="uploadResult">No upload yet.</pre>
                </details>
            </div>

            <div class="card full">
                <h2>2. AI Assist</h2>
                <p class="muted">
                    Use one prompt and let the platform decide whether this is a question,
                    generation request, summary, KT, or comparison task.
                </p>

                <label>Prompt</label>
                <textarea id="aiPrompt">How is order placement handled?</textarea>

                <label>Top K</label>
                <input id="aiTopK" type="number" value="5" />

                <button onclick="runAIAssist()">Run AI Assist</button>

                <div class="section-title">Detected Task Type</div>
                <div id="aiTaskType" class="output-box">No task detected yet.</div>

                <div class="section-title">Routing Preferences</div>
                <div id="aiRouting" class="output-box">No routing info yet.</div>

                <div class="section-title">Main Output</div>
                <div id="aiMainOutput" class="output-box">No output yet.</div>

                <div class="section-title">Generated Code</div>
                <pre id="aiGeneratedCode" class="code-box">No generated code yet.</pre>

                <div class="section-title">Key Files</div>
                <div id="aiKeyFiles" class="output-box">No key files yet.</div>

                <div class="section-title">Evidence / References</div>
                <div id="aiEvidence" class="output-box">No evidence yet.</div>

                <details>
                    <summary>Show Raw AI Assist Response</summary>
                    <pre id="aiRawResult">No AI assist response yet.</pre>
                </details>
            </div>

            <div class="card">
                <h2>3. Search Codebase</h2>
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
                <h2>4. Ask Codebase</h2>
                <label>Question</label>
                <textarea id="askQuestion">What does this codebase do?</textarea>
                <label>Top K</label>
                <input id="askTopK" type="number" value="5" />
                <button onclick="askCodebase()">Ask</button>

                <div class="section-title">Answer</div>
                <div id="askAnswer" class="output-box">No answer yet.</div>

                <div class="section-title">Key Files</div>
                <div id="askFiles" class="output-box">No key files yet.</div>

                <div class="section-title">Evidence</div>
                <div id="askEvidence" class="output-box">No evidence yet.</div>

                <details>
                    <summary>Show Raw Ask Response</summary>
                    <pre id="askResult">No answer yet.</pre>
                </details>
            </div>

            <div class="card">
                <h2>5. Summarize Codebase</h2>
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
                <h2>6. Generate Code</h2>
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
                <h2>7. Generate and Save Code</h2>
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
            const STORAGE_KEYS = {
                workspacePath: "ai_codebase_active_workspace_path",
                workspaceName: "ai_codebase_active_workspace_name",
                workspaceId: "ai_codebase_active_workspace_id"
            };

            let currentWorkspacePath = "";
            let currentWorkspaceName = "";
            let currentWorkspaceId = "";

            function pretty(obj) {
                return JSON.stringify(obj, null, 2);
            }

            window.renderList = function(items) {
                if (!items || !items.length) return "None";
                return items.map(x => "- " + x).join("\\n");
            };

            window.renderEvidence = function(items) {
                if (!items || !items.length) return "No evidence returned.";
                return items.map((item, index) => {
                    const symbolLabel = item.symbol_label || "symbol: unknown";
                    const filePath = item.file_path || "unknown file";
                    const snippet = item.snippet || "";
                    return `${index + 1}. ${symbolLabel}\\n   file: ${filePath}\\n   snippet: ${snippet}`;
                }).join("\\n\\n");
            };

            function formatSearchResults(results) {
                if (!results || !results.length) return "No matching chunks found.";
                return results.map((item, index) => {
                    const symbolInfo = item.symbol_name
                        ? `${item.symbol_type || "symbol"}: ${item.parent_symbol ? item.parent_symbol + "." : ""}${item.symbol_name}`
                        : `chunk: ${item.chunk_index}`;

                    const snippet = (item.content || "").slice(0, 250);
                    return `${index + 1}. ${symbolInfo}\\n   file: ${item.file_path}\\n   distance: ${item.distance}\\n   snippet: ${snippet}...`;
                }).join("\\n\\n");
            }

            function formatRoutingPreferences(prefs) {
                if (!prefs) return "No routing preferences returned.";

                return [
                    `Mode: ${prefs.mode || "-"}`,
                    `System Tags: ${(prefs.system_tags || []).join(", ") || "None"}`,
                    `Preferred Scope Kinds: ${(prefs.preferred_scope_kinds || []).join(", ") || "None"}`
                ].join("\\n");
            }

            function persistWorkspaceInfo(path, name, workspaceId) {
                localStorage.setItem(STORAGE_KEYS.workspacePath, path || "");
                localStorage.setItem(STORAGE_KEYS.workspaceName, name || "");
                localStorage.setItem(STORAGE_KEYS.workspaceId, workspaceId || "");
            }

            function loadWorkspaceInfo() {
                currentWorkspacePath = localStorage.getItem(STORAGE_KEYS.workspacePath) || "";
                currentWorkspaceName = localStorage.getItem(STORAGE_KEYS.workspaceName) || "";
                currentWorkspaceId = localStorage.getItem(STORAGE_KEYS.workspaceId) || "";
                renderWorkspaceInfo();
            }

            function clearWorkspaceInfo() {
                currentWorkspacePath = "";
                currentWorkspaceName = "";
                currentWorkspaceId = "";
                localStorage.removeItem(STORAGE_KEYS.workspacePath);
                localStorage.removeItem(STORAGE_KEYS.workspaceName);
                localStorage.removeItem(STORAGE_KEYS.workspaceId);
                renderWorkspaceInfo();
            }

            function setWorkspaceInfo(path, name, workspaceId) {
                currentWorkspacePath = path || "";
                currentWorkspaceName = name || "";
                currentWorkspaceId = workspaceId || "";
                persistWorkspaceInfo(currentWorkspacePath, currentWorkspaceName, currentWorkspaceId);
                renderWorkspaceInfo();
            }

            function renderWorkspaceInfo() {
                document.getElementById("workspaceNameText").innerText = currentWorkspaceName || "(not set yet)";
                document.getElementById("workspacePathText").innerText =
                    currentWorkspacePath ? `Workspace Path: ${currentWorkspacePath}` : "(not set yet)";
                document.getElementById("workspaceIdText").innerText =
                    currentWorkspaceId ? `Workspace ID: ${currentWorkspaceId}` : "";
            }

            function getWorkspacePath() {
                return currentWorkspacePath || "";
            }

            function ensureActiveProject() {
                const workspacePath = getWorkspacePath();
                if (!workspacePath) {
                    throw new Error("No active project selected. Upload and index a project first.");
                }
                return workspacePath;
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

                    if (!response.ok) {
                        summaryBox.textContent = data.detail || "Upload failed.";
                        return;
                    }

                    setWorkspaceInfo(
                        data.upload_path || "",
                        data.workspace_name || "Uploaded Project",
                        data.workspace_id || ""
                    );

                    summaryBox.textContent =
                        `Upload completed successfully.\\n\\n` +
                        `Project: ${data.workspace_name ?? "-"}\\n` +
                        `Files uploaded: ${data.files_uploaded ?? "-"}\\n` +
                        `Documents parsed: ${data.documents_parsed ?? "-"}\\n` +
                        `Chunks indexed: ${data.chunks_indexed ?? "-"}\\n` +
                        `Workspace: ${data.upload_path ?? "-"}`;
                } catch (err) {
                    summaryBox.textContent = "Error: " + err.message;
                    resultBox.textContent = "Error: " + err.message;
                }
            }

            async function runAIAssist() {
                const rawBox = document.getElementById("aiRawResult");
                const taskTypeBox = document.getElementById("aiTaskType");
                const routingBox = document.getElementById("aiRouting");
                const mainOutputBox = document.getElementById("aiMainOutput");
                const generatedCodeBox = document.getElementById("aiGeneratedCode");
                const keyFilesBox = document.getElementById("aiKeyFiles");
                const evidenceBox = document.getElementById("aiEvidence");

                rawBox.textContent = "Running AI assist...";
                taskTypeBox.textContent = "Detecting task type...";
                routingBox.textContent = "Inferring routing preferences...";
                mainOutputBox.textContent = "Waiting for output...";
                generatedCodeBox.textContent = "No generated code yet.";
                keyFilesBox.textContent = "Loading key files...";
                evidenceBox.textContent = "Loading evidence...";

                try {
                    const workspacePath = ensureActiveProject();

                    const response = await fetch("/ai-assist", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            prompt: document.getElementById("aiPrompt").value,
                            top_k: parseInt(document.getElementById("aiTopK").value),
                            workspace_path: workspacePath
                        })
                    });

                    const data = await response.json();
                    rawBox.textContent = pretty(data);

                    if (!response.ok) {
                        taskTypeBox.textContent = "Error";
                        routingBox.textContent = "No routing preferences.";
                        mainOutputBox.textContent = data.detail || "AI assist failed.";
                        keyFilesBox.textContent = "None";
                        evidenceBox.textContent = "None";
                        generatedCodeBox.textContent = "No generated code.";
                        return;
                    }

                    taskTypeBox.textContent = data.task_type || "unknown";
                    routingBox.textContent = formatRoutingPreferences(data.routing_preferences || {});

                    if (data.answer) {
                        mainOutputBox.textContent = data.answer;
                    } else if (data.summary) {
                        mainOutputBox.textContent = data.summary;
                    } else if (data.generated_code) {
                        mainOutputBox.textContent = `Target File: ${data.target_file || "-"}`;
                        generatedCodeBox.textContent = data.generated_code;
                    } else {
                        mainOutputBox.textContent = "No main output returned.";
                    }

                    if (data.generated_code && !generatedCodeBox.textContent.trim()) {
                        generatedCodeBox.textContent = data.generated_code;
                    } else if (!data.generated_code) {
                        generatedCodeBox.textContent = "No generated code for this task type.";
                    }

                    keyFilesBox.textContent = window.renderList(data.key_files || []);
                    evidenceBox.textContent = window.renderEvidence(data.evidence || data.references || []);
                } catch (err) {
                    taskTypeBox.textContent = "Error";
                    routingBox.textContent = "Error";
                    mainOutputBox.textContent = "Error: " + err.message;
                    generatedCodeBox.textContent = "Error";
                    keyFilesBox.textContent = "Error";
                    evidenceBox.textContent = "Error";
                    rawBox.textContent = "Error: " + err.message;
                }
            }

            async function searchCodebase() {
                const resultBox = document.getElementById("searchResult");
                const summaryBox = document.getElementById("searchSummary");
                resultBox.textContent = "Searching...";
                summaryBox.textContent = "Searching...";

                try {
                    const workspacePath = ensureActiveProject();

                    const response = await fetch("/search-codebase", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            query: document.getElementById("searchQuery").value,
                            top_k: parseInt(document.getElementById("searchTopK").value),
                            workspace_path: workspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);

                    if (!response.ok) {
                        summaryBox.textContent = data.detail || "Search failed.";
                        return;
                    }

                    summaryBox.textContent = formatSearchResults(data.results || []);
                } catch (err) {
                    summaryBox.textContent = "Error: " + err.message;
                    resultBox.textContent = "Error: " + err.message;
                }
            }

            async function askCodebase() {
                const resultBox = document.getElementById("askResult");
                const answerBox = document.getElementById("askAnswer");
                const filesBox = document.getElementById("askFiles");
                const evidenceBox = document.getElementById("askEvidence");

                resultBox.textContent = "Generating answer...";
                answerBox.textContent = "Generating answer...";
                filesBox.textContent = "Loading key files...";
                evidenceBox.textContent = "Loading evidence...";

                try {
                    const workspacePath = ensureActiveProject();

                    const response = await fetch("/ask-codebase", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            question: document.getElementById("askQuestion").value,
                            top_k: parseInt(document.getElementById("askTopK").value),
                            workspace_path: workspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);

                    if (!response.ok) {
                        answerBox.textContent = data.detail || "Ask failed.";
                        filesBox.textContent = "None";
                        evidenceBox.textContent = "None";
                        return;
                    }

                    answerBox.textContent = data.answer || "No answer returned.";
                    filesBox.textContent = window.renderList(data.key_files || []);
                    evidenceBox.textContent = window.renderEvidence(data.evidence || []);
                } catch (err) {
                    answerBox.textContent = "Error: " + err.message;
                    filesBox.textContent = "Error loading key files.";
                    evidenceBox.textContent = "Error loading evidence.";
                    resultBox.textContent = "Error: " + err.message;
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
                    const workspacePath = ensureActiveProject();

                    const response = await fetch("/summarize-codebase", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            top_k: parseInt(document.getElementById("summaryTopK").value),
                            workspace_path: workspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);

                    if (!response.ok) {
                        summaryTextBox.textContent = data.detail || "Summary failed.";
                        filesBox.textContent = "None";
                        return;
                    }

                    summaryTextBox.textContent = data.summary || "No summary returned.";
                    filesBox.textContent = window.renderList(data.important_files || []);
                } catch (err) {
                    summaryTextBox.textContent = "Error: " + err.message;
                    filesBox.textContent = "Error loading important files.";
                    resultBox.textContent = "Error: " + err.message;
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
                    const workspacePath = ensureActiveProject();

                    const response = await fetch("/generate-code", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            task: document.getElementById("generateTask").value,
                            top_k: parseInt(document.getElementById("generateTopK").value),
                            workspace_path: workspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);

                    if (!response.ok) {
                        targetBox.textContent = data.detail || "Generation failed.";
                        codeBox.textContent = "No code returned.";
                        return;
                    }

                    targetBox.textContent = data.target_file || "No target file returned.";
                    codeBox.textContent = data.generated_code || "No code returned.";
                } catch (err) {
                    targetBox.textContent = "Error: " + err.message;
                    codeBox.textContent = "Error generating code.";
                    resultBox.textContent = "Error: " + err.message;
                }
            }

            async function generateAndSaveCode() {
                const resultBox = document.getElementById("saveResult");
                const summaryBox = document.getElementById("saveSummary");

                resultBox.textContent = "Generating and saving...";
                summaryBox.textContent = "Generating and saving...";

                try {
                    const workspacePath = ensureActiveProject();

                    const response = await fetch("/generate-and-save", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            task: document.getElementById("saveTask").value,
                            top_k: parseInt(document.getElementById("saveTopK").value),
                            overwrite: document.getElementById("overwriteFlag").checked,
                            workspace_path: workspacePath
                        })
                    });

                    const data = await response.json();
                    resultBox.textContent = pretty(data);

                    if (!response.ok) {
                        summaryBox.textContent = data.detail || "Save failed.";
                        return;
                    }

                    summaryBox.textContent =
                        `Status: ${data.write_status ?? "-"}\\n` +
                        `Target: ${data.final_target_file ?? "-"}\\n\\n` +
                        `${data.write_message ?? "No save message returned."}`;
                } catch (err) {
                    summaryBox.textContent = "Error: " + err.message;
                    resultBox.textContent = "Error: " + err.message;
                }
            }

            window.addEventListener("DOMContentLoaded", () => {
                loadWorkspaceInfo();
            });
        </script>
    </body>
    </html>
    """

@router.post("/index-codebase")
def index_codebase(request: ScanRequest):
    try:
        documents = parse_codebase(request.path)
        workspace_id = normalize_workspace_id(request.path)
        workspace_name = Path(request.path).resolve().name

        for doc in documents:
            doc["project_name"] = workspace_name

        stats = index_documents(
            documents=documents,
            workspace_id=workspace_id,
            project_name=workspace_name,
            replace_existing=True,
        )

        add_project_to_catalog(
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            tags=[],
            workspace_path=request.path,
            metadata={
                "source": "manual_index",
                "files_found": len(documents),
            },
            files_indexed=stats["files_indexed"],
            chunks_indexed=stats["chunks_indexed"],
        )

        return {
            "message": "Codebase indexed successfully",
            "path": request.path,
            "workspace_path": normalize_workspace_id(request.path),
            "workspace_id": workspace_id,
            "workspace_name": workspace_name,
            **stats,
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
        workspace_id = resolve_request_workspace_id(
            workspace_id=request.workspace_id,
            workspace_path=request.workspace_path,
        )

        search_result = run_catalog_routed_search(
            query=request.query,
            top_k=request.top_k,
            workspace_id=workspace_id,
            search_scope=request.search_scope,
        )

        return {
            "message": "Search completed successfully",
            "query": request.query,
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            "search_scope": request.search_scope,
            "project_routing": search_result["project_routing"],
            "routing_preferences": search_result["routing_preferences"],
            "matches_found": len(search_result["matches"]),
            "results": search_result["matches"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    

@router.post("/ask-codebase")
def ask_codebase(request: AskRequest):
    try:
        workspace_id = resolve_request_workspace_id(
            workspace_id=request.workspace_id,
            workspace_path=request.workspace_path,
        )

        search_result = run_catalog_routed_search(
            query=request.question,
            top_k=request.top_k,
            workspace_id=workspace_id,
            search_scope=request.search_scope,
        )

        response = build_grounded_answer(
            question=request.question,
            matches=search_result["matches"]
        )

        return {
            "message": "Answer generated successfully",
            "question": request.question,
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            "search_scope": request.search_scope,
            "project_routing": search_result["project_routing"],
            "routing_preferences": search_result["routing_preferences"],
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
        upload_info = save_uploaded_files(files)

        return {
            "message": "Files uploaded and indexed successfully",
            "upload_path": upload_info["workspace_path"],
            "workspace_path": upload_info["workspace_path"],
            "workspace_id": upload_info["workspace_id"],
            "workspace_name": upload_info["workspace_name"],
            "files_uploaded": upload_info["files_uploaded"],
            "files_found": upload_info["files_found"],
            "sample_files": upload_info["sample_files"],
            **upload_info["index_stats"],
        }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file uploaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/generate-code")
def generate_code(request: GenerateRequest):
    try:
        workspace_id = resolve_request_workspace_id(
            workspace_id=request.workspace_id,
            workspace_path=request.workspace_path,
        )

        search_result = run_catalog_routed_search(
            query=request.task,
            top_k=request.top_k,
            workspace_id=workspace_id,
            search_scope=request.search_scope,
        )

        response = build_generation_output(
            task=request.task,
            matches=search_result["matches"]
        )

        return {
            "message": "Code generated successfully",
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            "search_scope": request.search_scope,
            "project_routing": search_result["project_routing"],
            "routing_preferences": search_result["routing_preferences"],
            **response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/generate-and-save")
def generate_and_save(request: GenerateAndSaveRequest):
    try:
        workspace_id = resolve_request_workspace_id(
            workspace_id=request.workspace_id,
            workspace_path=request.workspace_path,
        )

        search_result = run_catalog_routed_search(
            query=request.task,
            top_k=request.top_k,
            workspace_id=workspace_id,
            search_scope=request.search_scope,
        )

        generation = build_generation_output(
            task=request.task,
            matches=search_result["matches"]
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
            "search_scope": request.search_scope,
            "project_routing": search_result["project_routing"],
            "routing_preferences": search_result["routing_preferences"],
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
        workspace_id = resolve_request_workspace_id(
            workspace_id=request.workspace_id,
            workspace_path=request.workspace_path,
        )

        summary_query = "overall architecture modules services routes helpers main purpose of this codebase"

        search_result = run_catalog_routed_search(
            query=summary_query,
            top_k=request.top_k,
            workspace_id=workspace_id,
            search_scope=request.search_scope,
        )

        response = summarize_codebase(search_result["matches"])

        return {
            "message": "Codebase summary generated successfully",
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            "search_scope": request.search_scope,
            "project_routing": search_result["project_routing"],
            "routing_preferences": search_result["routing_preferences"],
            **response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/ai-assist")
def ai_assist(request: AIAssistRequest):
    try:
        workspace_id = resolve_request_workspace_id(
            workspace_id=request.workspace_id,
            workspace_path=request.workspace_path,
        )

        task_plan = build_task_routing(request.prompt)

        search_result = run_catalog_routed_search(
            query=request.prompt,
            top_k=request.top_k,
            workspace_id=workspace_id,
            search_scope=request.search_scope,
        )

        matches = search_result["matches"]
        task_type = task_plan["task_type"]

        if task_type == "generate":
            generation = build_generation_output(
                task=request.prompt,
                matches=matches
            )
            return {
                "message": "AI assist completed successfully",
                "task_type": task_type,
                "prompt": request.prompt,
                "workspace_path": request.workspace_path,
                "workspace_id": workspace_id,
                "search_scope": request.search_scope,
                "project_routing": search_result["project_routing"],
                "routing_preferences": search_result["routing_preferences"],
                **generation
            }

        if task_type in {"summary", "kt"}:
            summary = summarize_codebase(matches)
            return {
                "message": "AI assist completed successfully",
                "task_type": task_type,
                "prompt": request.prompt,
                "workspace_path": request.workspace_path,
                "workspace_id": workspace_id,
                "search_scope": request.search_scope,
                "project_routing": search_result["project_routing"],
                "routing_preferences": search_result["routing_preferences"],
                **summary
            }

        if task_type == "compare":
            response = build_grounded_answer(
                question=request.prompt,
                matches=matches
            )
            return {
                "message": "AI assist completed successfully",
                "task_type": task_type,
                "prompt": request.prompt,
                "workspace_path": request.workspace_path,
                "workspace_id": workspace_id,
                "search_scope": request.search_scope,
                "project_routing": search_result["project_routing"],
                "routing_preferences": search_result["routing_preferences"],
                **response
            }

        response = build_grounded_answer(
            question=request.prompt,
            matches=matches
        )
        return {
            "message": "AI assist completed successfully",
            "task_type": task_type,
            "prompt": request.prompt,
            "workspace_path": request.workspace_path,
            "workspace_id": workspace_id,
            "search_scope": request.search_scope,
            "project_routing": search_result["project_routing"],
            "routing_preferences": search_result["routing_preferences"],
            **response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.get("/list-projects")
def list_projects():
    projects = get_all_projects()
    return {
        "projects": projects,
        "count": len(projects)
    }