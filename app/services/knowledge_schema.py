from typing import Dict, List, Optional


def default_knowledge_metadata() -> Dict:
    """
    Default metadata schema for any indexed knowledge item.

    This schema is intentionally split into:
    1. factual metadata -> directly known from file/symbol structure
    2. inferred metadata -> filled later by classification logic

    Step 23A only defines the schema.
    Step 23B+ will populate inferred fields.
    """
    return {
        # -----------------------------
        # Factual / identity metadata
        # -----------------------------
        "workspace_id": "",
        "project_name": "",
        "file_path": "",
        "file_name": "",
        "source_type": "",      # code / documentation / config / api_spec
        "chunk_index": 0,

        # -----------------------------
        # Symbol metadata
        # -----------------------------
        "symbol_name": None,
        "symbol_type": None,    # class / function / method / text_chunk
        "parent_symbol": None,

        # -----------------------------
        # Inferred knowledge classification
        # -----------------------------
        "content_kind": "unknown",   # code / doc / api_spec / config / example / test / unknown
        "scope_kind": "unknown",     # broker_specific / shared / wrapper / abstraction / hybrid / unknown

        # Multi-tag fields
        "system_tags": "",           # stored as comma-separated string for vector metadata compatibility
        "role_tags": "",             # stored as comma-separated string
        "style_tags": "",            # stored as comma-separated string
        "project_tags": "",          # stored as comma-separated string
    }


def normalize_tag_list(tags: Optional[List[str]]) -> str:
    """
    Convert a list of tags into a stable comma-separated string.
    This is useful because vector DB metadata works more reliably
    with plain strings than nested Python lists.

    Example:
        ["xts", "dhan", "tradehull"] -> "dhan,tradehull,xts"
    """
    if not tags:
        return ""

    clean_tags = []
    for tag in tags:
        if tag is None:
            continue
        tag = str(tag).strip().lower()
        if not tag:
            continue
        clean_tags.append(tag)

    return ",".join(sorted(set(clean_tags)))


def parse_tag_string(tag_string: Optional[str]) -> List[str]:
    """
    Convert stored comma-separated tag string back into a Python list.

    Example:
        "dhan,tradehull,xts" -> ["dhan", "tradehull", "xts"]
    """
    if not tag_string:
        return []

    return [part.strip() for part in str(tag_string).split(",") if part.strip()]


def build_knowledge_metadata(
    *,
    workspace_id: str,
    project_name: str,
    file_path: str,
    file_name: str,
    source_type: str,
    chunk_index: int,
    symbol_name: Optional[str] = None,
    symbol_type: Optional[str] = None,
    parent_symbol: Optional[str] = None,
    content_kind: str = "unknown",
    scope_kind: str = "unknown",
    system_tags: Optional[List[str]] = None,
    role_tags: Optional[List[str]] = None,
    style_tags: Optional[List[str]] = None,
    project_tags: Optional[List[str]] = None,
) -> Dict:
    """
    Build one metadata dictionary using the platform schema.

    This function ensures every indexed knowledge item follows the same
    metadata structure, even before we add advanced classification logic.
    """
    metadata = default_knowledge_metadata()

    metadata.update(
        {
            "workspace_id": workspace_id,
            "project_name": project_name,
            "file_path": file_path,
            "file_name": file_name,
            "source_type": source_type,
            "chunk_index": chunk_index,
            "symbol_name": symbol_name,
            "symbol_type": symbol_type,
            "parent_symbol": parent_symbol,
            "content_kind": content_kind,
            "scope_kind": scope_kind,
            "system_tags": normalize_tag_list(system_tags),
            "role_tags": normalize_tag_list(role_tags),
            "style_tags": normalize_tag_list(style_tags),
            "project_tags": normalize_tag_list(project_tags),
        }
    )

    return metadata