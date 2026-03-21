# AI Codebase Assistant

An AI-powered backend platform that can ingest a codebase, index it using embeddings, search relevant code, answer questions grounded in project context, and generate new code following existing project patterns.

## Features

- Upload single code files, multiple files, or zip archives
- Parse Python code and supported project files
- Chunk and index code into a vector database
- Search relevant code snippets using semantic retrieval
- Ask natural-language questions about the codebase
- Generate new code using retrieved project context
- Save generated code directly into the uploaded workspace

## Tech Stack

- **FastAPI** — backend API framework
- **Hugging Face Inference** — LLM-based answering and code generation
- **Hugging Face Embeddings** — semantic code retrieval
- **ChromaDB** — vector storage
- **Pydantic** — request validation
- **GitHub Codespaces** — cloud development environment

## Architecture

```text
Code Files / Zip Upload
        ↓
Upload Service
        ↓
Parser
        ↓
Chunking
        ↓
Embedding Model
        ↓
Chroma Vector Store
        ↓
Search / Retrieval
        ↓
LLM (Hugging Face)
        ↓
Answer / Code Generation
```

## Project Structure

```text
app
├── api
│   └── routes.py
├── services
│   ├── parser.py
│   ├── upload_service.py
│   ├── vector_store.py
│   ├── answer_service.py
│   ├── generation_service.py
│   ├── llm_service.py
│   └── file_writer.py
└── main.py
```

## Setup

### 1. Clone repository

```bash
git clone <your-repo-url>
cd codebase-intelligence-platform
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bat
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Then update these values:

- `HF_TOKEN`
- `HF_MODEL`
- `HF_PROVIDER`

Example:

```env
HF_TOKEN=your_huggingface_token_here
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_PROVIDER=cerebras
```

### 5. Run server

```bash
python -m uvicorn app.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Main API Endpoints

### Health

- `GET /health`

### Upload / Scan / Index

- `POST /scan-codebase`
- `POST /upload-codebase`
- `POST /upload-and-index`
- `POST /index-codebase`
- `GET /index-stats`

### Search / Ask

- `POST /search-codebase`
- `POST /ask-codebase`

### Code Generation

- `POST /generate-code`
- `POST /generate-and-save`

## Example Workflow

### 1. Upload and index code

Use `/upload-and-index` to upload files or a zip archive and automatically index them.

### 2. Ask codebase questions

Example questions:

- `Where is order placement implemented?`
- `What does this file do?`
- `How is stoploss logic handled?`

### 3. Generate code

Example task:

- `Create a helper function for modifying stoploss orders in Dhan similar to the existing order placement style`

### 4. Save generated code

Use `/generate-and-save` to write the generated output into the uploaded workspace.

## Example Request Bodies

### Search codebase

```json
{
  "query": "Where is order placement implemented?",
  "top_k": 5,
  "workspace_path": "uploads/your-workspace-id"
}
```

### Ask codebase

```json
{
  "question": "What does this file do?",
  "top_k": 5,
  "workspace_path": "uploads/your-workspace-id"
}
```

### Generate code

```json
{
  "task": "Create a helper function for modifying stoploss orders in Dhan similar to the existing order placement style",
  "top_k": 5,
  "workspace_path": "uploads/your-workspace-id"
}
```

### Generate and save code

```json
{
  "task": "Create a helper function for modifying stoploss orders in Dhan similar to the existing order placement style",
  "top_k": 5,
  "overwrite": false,
  "workspace_path": "uploads/your-workspace-id"
}
```

## What This Project Demonstrates

This project demonstrates practical AI backend engineering skills:

- Retrieval-Augmented Generation (RAG)
- semantic search over code
- LLM-based code understanding
- context-aware code generation
- backend API design with FastAPI
- workspace-safe file generation

## Current Limitations

- parsing is currently basic
- chunking is not yet function/class aware
- multi-turn conversation memory is not implemented
- code generation should still be reviewed manually before production use

## Future Improvements

- function-level and class-level parsing
- repository dependency graph
- architecture summary endpoint
- chat history / session memory
- frontend chat UI
- support for more languages beyond Python

## Author

Built as an AI backend engineering portfolio project focused on code intelligence, semantic retrieval, and grounded code generation.
