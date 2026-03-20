from pathlib import Path
from typing import Dict, List, Optional
import uuid

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings


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


def index_documents(documents: List[Dict[str, str]], workspace_id: str) -> Dict[str, int]:
    embedder = get_embedder()

    all_chunks: List[str] = []
    all_metadatas: List[Dict] = []
    all_ids: List[str] = []

    files_indexed = 0

    for doc in documents:
        chunks = chunk_text(doc["content"])
        if not chunks:
            continue

        files_indexed += 1

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "workspace_id": workspace_id,
                    "file_path": doc["path"],
                    "file_name": doc.get("file_name", ""),
                    "source_type": doc.get("source_type", "code"),
                    "chunk_index": chunk_index
                }
            )
            all_ids.append(uuid.uuid4().hex)

    if not all_chunks:
        return {
            "workspace_id": workspace_id,
            "files_indexed": 0,
            "chunks_indexed": 0,
            "total_chunks_in_collection": collection.count()
        }

    embeddings = embedder.embed_documents(all_chunks)

    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas
    )

    return {
        "workspace_id": workspace_id,
        "files_indexed": files_indexed,
        "chunks_indexed": len(all_chunks),
        "total_chunks_in_collection": collection.count()
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
                "file_path": metadata.get("file_path"),
                "file_name": metadata.get("file_name"),
                "source_type": metadata.get("source_type"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
                "content": doc
            }
        )

    return matches


def get_collection_stats() -> Dict[str, int]:
    return {
        "total_chunks_in_collection": collection.count()
    }