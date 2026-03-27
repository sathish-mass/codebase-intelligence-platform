from typing import Dict, List, Optional, Tuple
import re

from app.services.knowledge_schema import parse_tag_string
from app.services.knowledge_classifier import SYSTEM_SEED_KEYWORDS


def normalize_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize_text(text: Optional[str]) -> List[str]:
    clean = normalize_text(text)
    parts = re.split(r"[^a-zA-Z0-9_]+", clean)
    return [part for part in parts if len(part) >= 2]


def infer_query_system_tags(query: str) -> List[str]:
    query_l = normalize_text(query)
    tags = []

    for system_name, keywords in SYSTEM_SEED_KEYWORDS.items():
        if any(keyword in query_l for keyword in keywords):
            tags.append(system_name)

    return sorted(set(tags))


def infer_query_scope_preferences(query: str) -> List[str]:
    query_l = normalize_text(query)
    scopes: List[str] = []

    if any(word in query_l for word in ["common", "shared", "generic", "reusable"]):
        scopes.extend(["shared", "abstraction"])

    if any(word in query_l for word in ["wrapper", "adapter", "dispatcher"]):
        scopes.append("wrapper")

    if any(word in query_l for word in ["base", "interface", "abstract"]):
        scopes.append("abstraction")

    if any(word in query_l for word in ["compare", "difference", "vs", "versus"]):
        scopes.append("hybrid")

    if infer_query_system_tags(query_l):
        scopes.append("broker_specific")
        scopes.append("shared")

    if not scopes:
        scopes.append("shared")

    return list(dict.fromkeys(scopes))


def infer_query_mode(query: str) -> str:
    query_l = normalize_text(query)

    if any(word in query_l for word in ["compare", "difference", "vs", "versus"]):
        return "compare"

    if infer_query_system_tags(query_l):
        return "system_specific"

    if any(word in query_l for word in ["common", "shared", "generic", "tradehull style", "platform style"]):
        return "shared_style"

    return "general"


def infer_project_search_mode(query: str, active_workspace_id: Optional[str] = None) -> str:
    query_l = normalize_text(query)

    if any(
        phrase in query_l
        for phrase in [
            "all projects",
            "all codebases",
            "across all projects",
            "across projects",
            "cross project",
            "cross-project",
            "whole catalog",
            "entire catalog",
            "all workspaces",
        ]
    ):
        return "all_projects"

    if any(word in query_l for word in ["compare", "difference", "vs", "versus"]):
        return "candidate_multi"

    if active_workspace_id:
        return "active_only"

    return "candidate_multi"


def infer_query_routing_preferences(
    query: str,
    active_workspace_id: Optional[str] = None,
    search_mode_override: Optional[str] = None,
) -> Dict:
    system_tags = infer_query_system_tags(query)
    preferred_scope_kinds = infer_query_scope_preferences(query)
    mode = infer_query_mode(query)

    project_search_mode = search_mode_override or infer_project_search_mode(
        query=query,
        active_workspace_id=active_workspace_id,
    )

    return {
        "query": query,
        "mode": mode,
        "system_tags": system_tags,
        "preferred_scope_kinds": preferred_scope_kinds,
        "project_search_mode": project_search_mode,
    }


def _safe_metadata(project: Dict) -> Dict:
    metadata = project.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _metadata_tag_list(metadata: Dict, key: str) -> List[str]:
    value = metadata.get(key, "")
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return parse_tag_string(value)


def score_match(match: Dict, preferences: Dict) -> float:
    distance = float(match.get("distance", 999))
    base_score = -distance

    match_system_tags = parse_tag_string(match.get("system_tags", ""))
    match_scope_kind = (match.get("scope_kind") or "").strip().lower()
    match_role_tags = parse_tag_string(match.get("role_tags", ""))
    match_style_tags = parse_tag_string(match.get("style_tags", ""))

    query_system_tags = preferences.get("system_tags", [])
    preferred_scope_kinds = preferences.get("preferred_scope_kinds", [])
    mode = preferences.get("mode", "general")

    score = base_score

    if query_system_tags:
        overlap = len(set(query_system_tags).intersection(set(match_system_tags)))
        score += overlap * 2.5

    if match_scope_kind and match_scope_kind in preferred_scope_kinds:
        score += 1.5

    if mode == "general":
        if match_scope_kind in {"shared", "abstraction"}:
            score += 0.8
        if "platform_style" in match_style_tags or "helper_style" in match_style_tags:
            score += 0.5

    if mode == "shared_style":
        if match_scope_kind in {"shared", "abstraction", "wrapper"}:
            score += 1.8
        if "platform_style" in match_style_tags:
            score += 1.0

    if mode == "compare":
        if match_scope_kind in {"hybrid", "wrapper"}:
            score += 1.8
        if len(match_system_tags) >= 2:
            score += 1.2

    query_l = normalize_text(preferences.get("query"))

    if "order" in query_l and "order" in match_role_tags:
        score += 0.6
    if "stoploss" in query_l and "stoploss" in match_role_tags:
        score += 0.6
    if "target" in query_l and "target" in match_role_tags:
        score += 0.6
    if "auth" in query_l and "auth" in match_role_tags:
        score += 0.6
    if "wrapper" in query_l and "wrapper" in match_role_tags:
        score += 0.6

    return score


def rerank_matches(matches: List[Dict], preferences: Dict) -> List[Dict]:
    scored = []

    for match in matches:
        routing_score = score_match(match, preferences)
        enriched = dict(match)
        enriched["routing_score"] = routing_score
        scored.append(enriched)

    scored.sort(key=lambda item: item["routing_score"], reverse=True)
    return scored


def _project_search_blob(project: Dict) -> str:
    workspace_name = project.get("workspace_name", "")
    workspace_path = project.get("workspace_path", "")
    tags = " ".join(project.get("tags", []) or [])

    metadata = _safe_metadata(project)

    important_meta = " ".join(
        [
            str(metadata.get("primary_system", "")),
            str(metadata.get("primary_scope", "")),
            str(metadata.get("primary_role", "")),
            str(metadata.get("primary_source_type", "")),
            str(metadata.get("project_system_tags", "")),
            str(metadata.get("project_role_tags", "")),
            str(metadata.get("project_style_tags", "")),
            str(metadata.get("project_scope_tags", "")),
            str(metadata.get("project_summary_hint", "")),
        ]
    )

    return normalize_text(f"{workspace_name} {workspace_path} {tags} {important_meta}")


def score_project_candidate(
    project: Dict,
    preferences: Dict,
    active_workspace_id: Optional[str] = None,
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    project_workspace_id = project.get("workspace_id", "")
    metadata = _safe_metadata(project)

    blob = _project_search_blob(project)
    blob_tokens = set(tokenize_text(blob))

    query = preferences.get("query", "")
    query_tokens = set(tokenize_text(query))
    query_system_tags = preferences.get("system_tags", [])
    preferred_scope_kinds = preferences.get("preferred_scope_kinds", [])
    mode = preferences.get("mode", "general")
    project_search_mode = preferences.get("project_search_mode", "active_only")

    project_system_tags = set(_metadata_tag_list(metadata, "project_system_tags"))
    project_role_tags = set(_metadata_tag_list(metadata, "project_role_tags"))
    project_style_tags = set(_metadata_tag_list(metadata, "project_style_tags"))
    project_scope_tags = set(_metadata_tag_list(metadata, "project_scope_tags"))

    primary_system = normalize_text(metadata.get("primary_system", ""))
    primary_scope = normalize_text(metadata.get("primary_scope", ""))
    summary_hint = normalize_text(metadata.get("project_summary_hint", ""))

    if active_workspace_id and project_workspace_id == active_workspace_id:
        if project_search_mode == "active_only":
            score += 8.0
            reasons.append("active_workspace_strong_match")
        else:
            score += 2.0
            reasons.append("active_workspace_soft_match")

    overlap_tokens = sorted(query_tokens.intersection(blob_tokens))
    if overlap_tokens:
        token_score = min(3.0, len(overlap_tokens) * 0.6)
        score += token_score
        reasons.append(f"name_or_path_overlap:{','.join(overlap_tokens[:6])}")

    for system_tag in query_system_tags:
        system_keywords = SYSTEM_SEED_KEYWORDS.get(system_tag, [])

        if system_tag in project_system_tags or system_tag == primary_system:
            score += 4.0
            reasons.append(f"project_metadata_system:{system_tag}")
            continue

        if system_tag in blob or any(keyword in blob for keyword in system_keywords):
            score += 2.5
            reasons.append(f"system_keyword:{system_tag}")

    if primary_scope and primary_scope in preferred_scope_kinds:
        score += 1.8
        reasons.append(f"primary_scope:{primary_scope}")

    for scope in preferred_scope_kinds:
        if scope in project_scope_tags:
            score += 0.9
            reasons.append(f"project_scope_tag:{scope}")

    query_l = normalize_text(query)

    if "order" in query_l and "order" in project_role_tags:
        score += 1.2
        reasons.append("project_role:order")

    if "auth" in query_l and "auth" in project_role_tags:
        score += 1.2
        reasons.append("project_role:auth")

    if "wrapper" in query_l and "wrapper" in project_role_tags:
        score += 1.2
        reasons.append("project_role:wrapper")

    if mode == "shared_style":
        if primary_scope in {"shared", "abstraction", "wrapper"}:
            score += 1.8
            reasons.append("shared_style_scope_match")
        if "platform_style" in project_style_tags:
            score += 1.0
            reasons.append("platform_style_match")

    if mode == "compare":
        matched_count = 0
        for system_tag in query_system_tags:
            system_keywords = SYSTEM_SEED_KEYWORDS.get(system_tag, [])
            if (
                system_tag in project_system_tags
                or system_tag == primary_system
                or system_tag in blob
                or any(keyword in blob for keyword in system_keywords)
            ):
                matched_count += 1

        if matched_count:
            score += matched_count * 1.5
            reasons.append("compare_project_match")

        if len(project_system_tags) >= 1:
            score += 0.4
            reasons.append("project_has_system_profile")

    if summary_hint:
        summary_overlap = set(tokenize_text(summary_hint)).intersection(query_tokens)
        if summary_overlap:
            score += min(1.5, len(summary_overlap) * 0.4)
            reasons.append("summary_hint_overlap")

    return score, reasons


def _fallback_project_record(workspace_id: str) -> Dict:
    return {
        "workspace_id": workspace_id,
        "workspace_name": workspace_id.split("/")[-1] or workspace_id,
        "workspace_path": workspace_id,
        "tags": [],
        "metadata": {},
    }


def select_candidate_projects(
    projects: List[Dict],
    preferences: Dict,
    active_workspace_id: Optional[str] = None,
    limit: int = 4,
) -> List[Dict]:
    project_search_mode = preferences.get("project_search_mode", "active_only")

    scored_projects = []
    for project in projects:
        score, reasons = score_project_candidate(
            project=project,
            preferences=preferences,
            active_workspace_id=active_workspace_id,
        )

        enriched = dict(project)
        enriched["project_routing_score"] = score
        enriched["project_routing_reasons"] = reasons
        scored_projects.append(enriched)

    scored_projects.sort(
        key=lambda item: (
            item.get("project_routing_score", 0),
            item.get("last_indexed_at", ""),
            item.get("upload_timestamp", ""),
        ),
        reverse=True,
    )

    if project_search_mode == "active_only":
        if active_workspace_id:
            for project in scored_projects:
                if project.get("workspace_id") == active_workspace_id:
                    return [project]
            return [_fallback_project_record(active_workspace_id)]

        return scored_projects[:1]

    if project_search_mode == "all_projects":
        return scored_projects[: max(limit, len(scored_projects))]

    positive_projects = [project for project in scored_projects if project.get("project_routing_score", 0) > 0]

    if positive_projects:
        if preferences.get("mode") == "compare":
            return positive_projects[: max(2, min(limit, len(positive_projects)))]
        return positive_projects[:limit]

    if active_workspace_id:
        for project in scored_projects:
            if project.get("workspace_id") == active_workspace_id:
                return [project]
        return [_fallback_project_record(active_workspace_id)]

    return scored_projects[:limit]


def build_project_routing_plan(
    projects: List[Dict],
    query: str,
    active_workspace_id: Optional[str] = None,
    search_mode_override: Optional[str] = None,
    project_limit: int = 4,
) -> Dict:
    preferences = infer_query_routing_preferences(
        query=query,
        active_workspace_id=active_workspace_id,
        search_mode_override=search_mode_override,
    )

    candidate_projects = select_candidate_projects(
        projects=projects,
        preferences=preferences,
        active_workspace_id=active_workspace_id,
        limit=project_limit,
    )

    searched_workspace_ids = []
    seen = set()

    for project in candidate_projects:
        workspace_id = project.get("workspace_id")
        if not workspace_id or workspace_id in seen:
            continue
        seen.add(workspace_id)
        searched_workspace_ids.append(workspace_id)

    return {
        "preferences": preferences,
        "project_search_mode": preferences.get("project_search_mode"),
        "active_workspace_id": active_workspace_id,
        "total_projects_available": len(projects),
        "candidate_projects": candidate_projects,
        "searched_workspace_ids": searched_workspace_ids,
    }


def routed_search_results(matches: List[Dict], query: str, preferences: Optional[Dict] = None) -> Dict:
    preferences = preferences or infer_query_routing_preferences(query)
    reranked = rerank_matches(matches, preferences)

    return {
        "preferences": preferences,
        "matches": reranked,
    }