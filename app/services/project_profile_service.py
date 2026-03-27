from collections import Counter
from typing import Dict, List

from app.services.knowledge_classifier import classify_knowledge_item
from app.services.knowledge_schema import normalize_tag_list, parse_tag_string


def _counter_to_dict(counter: Counter, limit: int = 10) -> Dict[str, int]:
    return {key: value for key, value in counter.most_common(limit)}


def _top_items(counter: Counter, limit: int = 5, skip_unknown: bool = True) -> List[str]:
    items = []
    for key, _ in counter.most_common():
        if not key:
            continue
        if skip_unknown and str(key).strip().lower() == "unknown":
            continue
        items.append(str(key).strip().lower())
        if len(items) >= limit:
            break
    return items


def _top_value(counter: Counter, default: str = "unknown", skip_unknown: bool = True) -> str:
    items = _top_items(counter, limit=1, skip_unknown=skip_unknown)
    return items[0] if items else default


def build_project_summary_hint(
    *,
    workspace_name: str,
    primary_system: str,
    primary_scope: str,
    project_role_tags: List[str],
    project_style_tags: List[str],
    primary_source_type: str,
) -> str:
    parts: List[str] = []

    if workspace_name:
        parts.append(workspace_name)

    if primary_system != "unknown":
        parts.append(f"{primary_system}-leaning")

    if primary_scope != "unknown":
        parts.append(f"{primary_scope} project")

    if primary_source_type != "unknown":
        parts.append(f"{primary_source_type}-heavy")

    if project_role_tags:
        parts.append("roles: " + ", ".join(project_role_tags[:3]))

    if project_style_tags:
        parts.append("style: " + ", ".join(project_style_tags[:2]))

    return "; ".join(parts) if parts else "general code project"


def aggregate_project_metadata(documents: List[Dict], workspace_name: str = "") -> Dict:
    """
    Build project-level metadata from parsed documents.

    This is Step 28's aggregation layer:
    document signals -> project profile stored in catalog metadata
    """
    source_type_counter: Counter = Counter()
    content_kind_counter: Counter = Counter()
    scope_kind_counter: Counter = Counter()
    system_tag_counter: Counter = Counter()
    role_tag_counter: Counter = Counter()
    style_tag_counter: Counter = Counter()
    project_tag_counter: Counter = Counter()

    files_analyzed = 0

    for doc in documents:
        file_path = doc.get("path", "")
        file_name = doc.get("file_name", "")
        source_type = doc.get("source_type", "unknown")
        content = doc.get("content", "")

        files_analyzed += 1
        source_type_counter[source_type] += 1

        inferred = classify_knowledge_item(
            file_path=file_path,
            file_name=file_name,
            source_type=source_type,
            content=content,
            symbol_name=None,
        )

        content_kind = inferred.get("content_kind", "unknown")
        scope_kind = inferred.get("scope_kind", "unknown")

        content_kind_counter[content_kind] += 1
        scope_kind_counter[scope_kind] += 1

        for tag in parse_tag_string(inferred.get("system_tags", "")):
            system_tag_counter[tag] += 1

        for tag in parse_tag_string(inferred.get("role_tags", "")):
            role_tag_counter[tag] += 1

        for tag in parse_tag_string(inferred.get("style_tags", "")):
            style_tag_counter[tag] += 1

        for tag in parse_tag_string(inferred.get("project_tags", "")):
            project_tag_counter[tag] += 1

    project_system_tags = _top_items(system_tag_counter, limit=5)
    project_role_tags = _top_items(role_tag_counter, limit=6)
    project_style_tags = _top_items(style_tag_counter, limit=5)
    project_scope_tags = _top_items(scope_kind_counter, limit=5)

    primary_system = _top_value(system_tag_counter, default="unknown")
    primary_scope = _top_value(scope_kind_counter, default="unknown")
    primary_role = _top_value(role_tag_counter, default="unknown")
    primary_source_type = _top_value(source_type_counter, default="unknown", skip_unknown=False)

    summary_hint = build_project_summary_hint(
        workspace_name=workspace_name,
        primary_system=primary_system,
        primary_scope=primary_scope,
        project_role_tags=project_role_tags,
        project_style_tags=project_style_tags,
        primary_source_type=primary_source_type,
    )

    return {
        "files_analyzed_for_profile": files_analyzed,
        "project_system_tags": normalize_tag_list(project_system_tags),
        "project_role_tags": normalize_tag_list(project_role_tags),
        "project_style_tags": normalize_tag_list(project_style_tags),
        "project_scope_tags": normalize_tag_list(project_scope_tags),
        "primary_system": primary_system,
        "primary_scope": primary_scope,
        "primary_role": primary_role,
        "primary_source_type": primary_source_type,
        "source_type_counts": _counter_to_dict(source_type_counter),
        "content_kind_counts": _counter_to_dict(content_kind_counter),
        "scope_kind_counts": _counter_to_dict(scope_kind_counter),
        "system_tag_counts": _counter_to_dict(system_tag_counter),
        "role_tag_counts": _counter_to_dict(role_tag_counter),
        "style_tag_counts": _counter_to_dict(style_tag_counter),
        "project_tag_counts": _counter_to_dict(project_tag_counter),
        "project_summary_hint": summary_hint,
    }


def build_catalog_tags_from_project_metadata(project_metadata: Dict) -> List[str]:
    """
    Convert project profile metadata into top-level catalog tags
    so routing can use them even without deep metadata inspection.
    """
    tags: List[str] = []

    for key in [
        "project_system_tags",
        "project_role_tags",
        "project_style_tags",
        "project_scope_tags",
    ]:
        tags.extend(parse_tag_string(project_metadata.get(key, "")))

    primary_system = str(project_metadata.get("primary_system", "")).strip().lower()
    primary_scope = str(project_metadata.get("primary_scope", "")).strip().lower()

    if primary_system and primary_system != "unknown":
        tags.append(primary_system)

    if primary_scope and primary_scope != "unknown":
        tags.append(primary_scope)

    return sorted(set(tag for tag in tags if tag))