from pathlib import Path
from typing import Dict, List, Optional
import uuid
import ast
import hashlib

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from app.services.knowledge_schema import build_knowledge_metadata
from app.services.knowledge_classifier import classify_knowledge_item


CHROMA_DIR = Path("data/chroma")
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "codebase_chunks_v2"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def normalize_workspace_id(workspace_path: str) -> str:
    return str(Path(workspace_path).resolve())


def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def extract_python_symbols(text: str) -> List[Dict]:
    """
    Parse Python source with AST and extract meaningful structural chunks:
    - top-level functions
    - classes
    - methods inside classes
    """
    if not text or not text.strip():
        return []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    chunks: List[Dict] = []

    def get_node_source(node) -> str:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return ""

        start = node.lineno - 1
        end = node.end_lineno
        return "\n".join(lines[start:end]).strip()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            source = get_node_source(node)
            if source:
                chunks.append(
                    {
                        "content": source,
                        "symbol_name": node.name,
                        "symbol_type": "function",
                        "parent_symbol": None,
                    }
                )

        elif isinstance(node, ast.ClassDef):
            class_source = get_node_source(node)
            if class_source:
                chunks.append(
                    {
                        "content": class_source,
                        "symbol_name": node.name,
                        "symbol_type": "class",
                        "parent_symbol": None,
                    }
                )

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_source = get_node_source(child)
                    if method_source:
                        chunks.append(
                            {
                                "content": method_source,
                                "symbol_name": child.name,
                                "symbol_type": "method",
                                "parent_symbol": node.name,
                            }
                        )

    return chunks


def build_structured_chunks(doc: Dict[str, str]) -> List[Dict]:
    """
    Build best possible chunks for a document.

    Strategy:
    - For Python code files: use AST-based symbol extraction
    - For everything else: fallback to fixed-size chunking
    """
    source_type = doc.get("source_type", "code")
    file_path = doc.get("path", "")
    content = doc.get("content", "")

    is_python_code = source_type == "code" and file_path.endswith(".py")

    if is_python_code:
        symbol_chunks = extract_python_symbols(content)
        if symbol_chunks:
            return symbol_chunks

    return [
        {
            "content": chunk,
            "symbol_name": None,
            "symbol_type": "text_chunk",
            "parent_symbol": None,
        }
        for chunk in chunk_text(content)
    ]


def build_chunk_id(
    workspace_id: str,
    file_path: str,
    chunk_index: int,
    symbol_name: Optional[str],
    symbol_type: Optional[str],
    parent_symbol: Optional[str],
) -> str:
    raw = "||".join(
        [
            workspace_id or "",
            file_path or "",
            str(chunk_index),
            symbol_name or "",
            symbol_type or "",
            parent_symbol or "",
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def delete_workspace_chunks(workspace_id: str) -> None:
    """
    Remove old chunks for one workspace before re-indexing.
    This prevents stale chunks and duplicate accumulation.
    """
    try:
        collection.delete(where={"workspace_id": workspace_id})
    except Exception:
        # Safe to ignore if nothing exists yet
        pass


def index_documents(
    documents: List[Dict[str, str]],
    workspace_id: str,
    project_name: Optional[str] = None,
    replace_existing: bool = True,
) -> Dict[str, int]:
    embedder = get_embedder()

    if replace_existing:
        delete_workspace_chunks(workspace_id)

    all_chunks: List[str] = []
    all_metadatas: List[Dict] = []
    all_ids: List[str] = []

    files_indexed = 0

    for doc in documents:
        normalized_file_path = str(Path(doc["path"]).resolve()) if doc.get("path") else ""
        structured_chunks = build_structured_chunks(doc)

        if not structured_chunks:
            continue

        files_indexed += 1

        resolved_project_name = doc.get("project_name") or project_name or doc.get("file_name", "")

        for chunk_index, chunk_info in enumerate(structured_chunks):
            chunk_content = chunk_info["content"]
            symbol_name = chunk_info.get("symbol_name")
            symbol_type = chunk_info.get("symbol_type")
            parent_symbol = chunk_info.get("parent_symbol")

            inferred = classify_knowledge_item(
                file_path=normalized_file_path,
                file_name=doc.get("file_name", ""),
                source_type=doc.get("source_type", "code"),
                content=chunk_content,
                symbol_name=symbol_name,
            )

            metadata = build_knowledge_metadata(
                workspace_id=workspace_id,
                project_name=resolved_project_name,
                file_path=normalized_file_path,
                file_name=doc.get("file_name", ""),
                source_type=doc.get("source_type", "code"),
                chunk_index=chunk_index,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                parent_symbol=parent_symbol,
                content_kind=inferred["content_kind"],
                scope_kind=inferred["scope_kind"],
                system_tags=inferred["system_tags"].split(",") if inferred["system_tags"] else [],
                role_tags=inferred["role_tags"].split(",") if inferred["role_tags"] else [],
                style_tags=inferred["style_tags"].split(",") if inferred["style_tags"] else [],
                project_tags=inferred["project_tags"].split(",") if inferred["project_tags"] else [],
            )

            chunk_id = build_chunk_id(
                workspace_id=workspace_id,
                file_path=normalized_file_path,
                chunk_index=chunk_index,
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                parent_symbol=parent_symbol,
            )

            all_chunks.append(chunk_content)
            all_metadatas.append(metadata)
            all_ids.append(chunk_id)

    if not all_chunks:
        return {
            "workspace_id": workspace_id,
            "files_indexed": 0,
            "chunks_indexed": 0,
            "total_chunks_in_collection": collection.count(),
        }

    embeddings = embedder.embed_documents(all_chunks)

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
    )

    return {
        "workspace_id": workspace_id,
        "files_indexed": files_indexed,
        "chunks_indexed": len(all_chunks),
        "total_chunks_in_collection": collection.count(),
    }


def search_similar_chunks(
    query: str,
    top_k: int = 5,
    workspace_id: Optional[str] = None
) -> List[Dict]:
    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": top_k
    }

    if workspace_id:
        query_kwargs["where"] = {"workspace_id": workspace_id}

    results = collection.query(**query_kwargs)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches = []

    for doc, metadata, distance in zip(documents, metadatas, distances):
        matches.append(
            {
                "workspace_id": metadata.get("workspace_id"),
                "project_name": metadata.get("project_name"),
                "file_path": metadata.get("file_path"),
                "file_name": metadata.get("file_name"),
                "source_type": metadata.get("source_type"),
                "chunk_index": metadata.get("chunk_index"),
                "symbol_name": metadata.get("symbol_name"),
                "symbol_type": metadata.get("symbol_type"),
                "parent_symbol": metadata.get("parent_symbol"),
                "content_kind": metadata.get("content_kind"),
                "scope_kind": metadata.get("scope_kind"),
                "system_tags": metadata.get("system_tags", ""),
                "role_tags": metadata.get("role_tags", ""),
                "style_tags": metadata.get("style_tags", ""),
                "project_tags": metadata.get("project_tags", ""),
                "distance": distance,
                "content": doc,
            }
        )

    return matches


def get_collection_stats() -> Dict[str, int]:
    return {
        "total_chunks_in_collection": collection.count()
    }