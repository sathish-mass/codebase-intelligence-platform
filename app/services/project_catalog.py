import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

PROJECT_CATALOG_PATH = Path("data/project_catalog.json")
PROJECT_CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_workspace_path(workspace_path: Optional[str]) -> str:
    if not workspace_path:
        return ""
    return str(Path(workspace_path).resolve())


def load_catalog() -> Dict:
    if PROJECT_CATALOG_PATH.exists():
        try:
            with open(PROJECT_CATALOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_catalog(catalog: Dict):
    with open(PROJECT_CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=4, ensure_ascii=False)


def add_project_to_catalog(
    workspace_id: str,
    workspace_name: str,
    tags: List[str],
    metadata: Optional[Dict] = None,
    workspace_path: Optional[str] = None,
    files_indexed: int = 0,
    chunks_indexed: int = 0,
) -> Dict:
    """
    Upsert one reliable catalog record for a project/workspace.
    """
    catalog = load_catalog()
    existing = catalog.get(workspace_id, {})

    existing_metadata = existing.get("metadata", {})
    if not isinstance(existing_metadata, dict):
        existing_metadata = {}

    merged_metadata = dict(existing_metadata)
    if metadata:
        merged_metadata.update(metadata)

    existing_tags = existing.get("tags", [])
    if not isinstance(existing_tags, list):
        existing_tags = []

    merged_tags = sorted(set(str(tag).strip() for tag in (existing_tags + (tags or [])) if str(tag).strip()))

    record = {
        "workspace_id": workspace_id,
        "workspace_name": workspace_name,
        "workspace_path": _normalize_workspace_path(workspace_path or existing.get("workspace_path")),
        "tags": merged_tags,
        "metadata": merged_metadata,
        "upload_timestamp": existing.get("upload_timestamp") or _utc_now_iso(),
        "last_indexed_at": _utc_now_iso(),
        "indexed": True,
        "files_indexed": int(files_indexed or 0),
        "chunks_indexed": int(chunks_indexed or 0),
    }

    catalog[workspace_id] = record
    save_catalog(catalog)
    return record


def get_all_projects() -> List[Dict]:
    catalog = load_catalog()
    projects: List[Dict] = []

    for workspace_id, record in catalog.items():
        if not isinstance(record, dict):
            continue

        normalized = dict(record)
        normalized["workspace_id"] = normalized.get("workspace_id") or workspace_id
        normalized["workspace_path"] = _normalize_workspace_path(normalized.get("workspace_path"))
        normalized["tags"] = normalized.get("tags") if isinstance(normalized.get("tags"), list) else []
        normalized["metadata"] = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
        projects.append(normalized)

    projects.sort(key=lambda item: item.get("upload_timestamp", ""), reverse=True)
    return projects


def get_project_by_workspace_id(workspace_id: str) -> Optional[Dict]:
    catalog = load_catalog()
    record = catalog.get(workspace_id)

    if not isinstance(record, dict):
        return None

    normalized = dict(record)
    normalized["workspace_id"] = normalized.get("workspace_id") or workspace_id
    normalized["workspace_path"] = _normalize_workspace_path(normalized.get("workspace_path"))
    normalized["tags"] = normalized.get("tags") if isinstance(normalized.get("tags"), list) else []
    normalized["metadata"] = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}

    return normalized