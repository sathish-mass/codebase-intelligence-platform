from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.api.routes import router

app = FastAPI(
    title="AI Codebase Assistant",
    version="1.0.0",
    description=(
        "An AI-powered backend platform for uploading codebases, "
        "indexing code, searching semantically, answering questions, "
        "summarizing architecture, and generating grounded code."
    ),
)

app.include_router(router)


@app.get("/", response_class=HTMLResponse, tags=["Root"])
def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Codebase Assistant</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                line-height: 1.6;
                color: #222;
            }
            .card {
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 24px;
                background: #fafafa;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            h1 {
                margin-top: 0;
            }
            code {
                background: #f1f1f1;
                padding: 2px 6px;
                border-radius: 6px;
            }
            ul {
                padding-left: 20px;
            }
            .actions {
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 20px;
            }
            a.button {
                display: inline-block;
                padding: 10px 16px;
                border-radius: 8px;
                background: #111827;
                color: white;
                text-decoration: none;
            }
            a.button.secondary {
                background: #374151;
            }
            .muted {
                color: #555;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>AI Codebase Assistant</h1>
            <p class="muted">
                Upload codebases, index them with embeddings, search relevant code,
                ask architecture questions, summarize projects, and generate grounded code.
            </p>

            <h2>Main Capabilities</h2>
            <ul>
                <li>Upload single files, multiple files, or zip archives</li>
                <li>Semantic code search</li>
                <li>Codebase question answering</li>
                <li>Architecture summarization</li>
                <li>Context-aware code generation</li>
                <li>Save generated code into uploaded workspace</li>
            </ul>

            <h2>Main Endpoints</h2>
            <ul>
                <li><code>POST /upload-and-index</code></li>
                <li><code>POST /search-codebase</code></li>
                <li><code>POST /ask-codebase</code></li>
                <li><code>POST /summarize-codebase</code></li>
                <li><code>POST /generate-code</code></li>
                <li><code>POST /generate-and-save</code></li>
            </ul>

            <div class="actions">
                <a class="button" href="/app-ui">Open App UI</a>
                <a class="button secondary" href="/docs">Open Swagger Docs</a>
            </div>
        </div>
    </body>
    </html>
    """