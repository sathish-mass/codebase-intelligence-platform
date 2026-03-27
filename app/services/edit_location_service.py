from collections import defaultdict
from typing import Dict, List, Tuple

from app.services.answer_service import clean_snippet, format_symbol_label
from app.services.knowledge_schema import parse_tag_string
from app.services.retrieval_router import (
    infer_query_system_tags,
    normalize_text,
    tokenize_text,
)


ROLE_KEYWORD_MAP = {
    "order": ["order", "orders"],
    "placement": ["place", "placement", "entry"],
    "modify": ["modify", "update", "edit", "patch", "change"],
    "cancel": ["cancel", "remove"],
    "stoploss": ["stoploss", "stop-loss", "sl"],
    "target": ["target", "takeprofit", "take-profit", "tp"],
    "auth": ["auth", "login", "token", "session"],
    "margin": ["margin", "fund"],
    "config": ["config", "setting"],
    "helper": ["helper", "utility"],
    "wrapper": ["wrapper", "adapter", "dispatcher"],
    "api": ["api", "endpoint", "request", "response"],
    "quantity": ["qty", "quantity", "size"],
}


def infer_edit_role_tags(task: str) -> List[str]:
    task_l = normalize_text(task)
    tags = []

    for role_tag, keywords in ROLE_KEYWORD_MAP.items():
        if any(keyword in task_l for keyword in keywords):
            tags.append(role_tag)

    return sorted(set(tags))


def infer_edit_preferences(task: str) -> Dict:
    return {
        "task": task,
        "task_tokens": set(tokenize_text(task)),
        "system_tags": infer_query_system_tags(task),
        "role_tags": infer_edit_role_tags(task),
    }


def build_symbol_blob(match: Dict) -> str:
    return " ".join(
        [
            str(match.get("file_name", "") or ""),
            str(match.get("file_path", "") or ""),
            str(match.get("symbol_name", "") or ""),
            str(match.get("symbol_type", "") or ""),
            str(match.get("parent_symbol", "") or ""),
        ]
    )


def recommend_edit_action(match: Dict) -> str:
    symbol_type = (match.get("symbol_type") or "").strip().lower()
    scope_kind = (match.get("scope_kind") or "").strip().lower()

    if symbol_type in {"function", "method"}:
        return "edit_existing_symbol"

    if symbol_type == "class":
        return "edit_class_or_add_method"

    if scope_kind == "wrapper":
        return "extend_wrapper_layer"

    if scope_kind in {"abstraction", "shared"}:
        return "extend_shared_or_base_layer"

    return "inspect_file_and_patch"


def score_edit_candidate(match: Dict, preferences: Dict) -> Tuple[float, List[str]]:
    reasons: List[str] = []

    score = float(match.get("routing_score", 0.0))
    distance = float(match.get("distance", 999))

    if "routing_score" not in match:
        score += -distance

    symbol_type = normalize_text(match.get("symbol_type", ""))
    source_type = normalize_text(match.get("source_type", ""))
    scope_kind = normalize_text(match.get("scope_kind", ""))
    file_path = normalize_text(match.get("file_path", ""))
    symbol_blob = normalize_text(build_symbol_blob(match))

    match_system_tags = set(parse_tag_string(match.get("system_tags", "")))
    match_role_tags = set(parse_tag_string(match.get("role_tags", "")))
    match_style_tags = set(parse_tag_string(match.get("style_tags", "")))

    task_tokens = preferences.get("task_tokens", set())
    task_system_tags = set(preferences.get("system_tags", []))
    task_role_tags = set(preferences.get("role_tags", []))
    task_l = normalize_text(preferences.get("task", ""))

    if symbol_type in {"function", "method"}:
        score += 2.2
        reasons.append("direct_function_or_method_edit")
    elif symbol_type == "class":
        score += 1.7
        reasons.append("class_level_edit")
    elif symbol_type == "text_chunk":
        score += 0.2
        reasons.append("file_context_only")

    if source_type == "code":
        score += 0.6
        reasons.append("code_file")

    if task_system_tags:
        overlap = sorted(task_system_tags.intersection(match_system_tags))
        if overlap:
            score += len(overlap) * 1.8
            reasons.append(f"system_match:{','.join(overlap)}")

    if task_role_tags:
        role_overlap = sorted(task_role_tags.intersection(match_role_tags))
        if role_overlap:
            score += len(role_overlap) * 1.2
            reasons.append(f"role_match:{','.join(role_overlap[:4])}")

    token_overlap = sorted(task_tokens.intersection(set(tokenize_text(symbol_blob))))
    if token_overlap:
        score += min(2.0, len(token_overlap) * 0.4)
        reasons.append(f"symbol_or_file_overlap:{','.join(token_overlap[:5])}")

    if "wrapper" in task_role_tags and scope_kind == "wrapper":
        score += 1.8
        reasons.append("wrapper_scope_match")

    if any(tag in task_role_tags for tag in {"helper", "api", "config"}) and scope_kind == "abstraction":
        score += 1.2
        reasons.append("abstraction_scope_match")

    if any(word in task_l for word in ["shared", "generic", "common", "base"]) and scope_kind in {"abstraction", "shared"}:
        score += 1.6
        reasons.append("shared_base_scope_match")

    if "routes.py" in file_path and any(word in task_l for word in ["route", "endpoint", "api"]):
        score += 1.4
        reasons.append("routes_file_match")

    if "service" in file_path and "service" in task_l:
        score += 1.0
        reasons.append("service_file_match")

    if any(name in file_path for name in ["support_library", "support-library", "library"]) and any(
        tag in task_role_tags for tag in {"helper", "wrapper"}
    ):
        score += 1.2
        reasons.append("support_library_match")

    if "platform_style" in match_style_tags and any(word in task_l for word in ["similar", "same style", "platform style"]):
        score += 0.8
        reasons.append("style_match")

    return score, reasons


def group_edit_targets(matches: List[Dict], preferences: Dict) -> List[Dict]:
    grouped: Dict[Tuple, Dict] = {}

    for match in matches:
        edit_score, reasons = score_edit_candidate(match, preferences)

        key = (
            match.get("workspace_id"),
            match.get("file_path"),
            match.get("symbol_name"),
            match.get("symbol_type"),
            match.get("parent_symbol"),
        )

        existing = grouped.get(key)
        if not existing or edit_score > existing["edit_score"]:
            grouped[key] = {
                "workspace_id": match.get("workspace_id"),
                "project_name": match.get("project_name"),
                "file_path": match.get("file_path"),
                "file_name": match.get("file_name"),
                "source_type": match.get("source_type"),
                "chunk_index": match.get("chunk_index"),
                "symbol_name": match.get("symbol_name"),
                "symbol_type": match.get("symbol_type"),
                "parent_symbol": match.get("parent_symbol"),
                "symbol_label": format_symbol_label(match),
                "scope_kind": match.get("scope_kind"),
                "system_tags": match.get("system_tags", ""),
                "role_tags": match.get("role_tags", ""),
                "style_tags": match.get("style_tags", ""),
                "routing_score": match.get("routing_score"),
                "distance": match.get("distance"),
                "edit_score": edit_score,
                "edit_reasons": reasons,
                "recommended_action": recommend_edit_action(match),
                "snippet": clean_snippet(match.get("content", "")),
                "supporting_matches": 1,
            }
        else:
            existing["supporting_matches"] += 1
            merged_reasons = list(dict.fromkeys(existing["edit_reasons"] + reasons))
            existing["edit_reasons"] = merged_reasons[:8]

    targets = list(grouped.values())
    targets.sort(
        key=lambda item: (
            item.get("edit_score", 0),
            item.get("routing_score", 0) if item.get("routing_score") is not None else -999,
            -(item.get("distance", 999)),
        ),
        reverse=True,
    )
    return targets


def build_edit_location_response(task: str, matches: List[Dict], top_k: int = 6) -> Dict:
    if not matches:
        return {
            "summary": "I could not find any good edit locations in the indexed knowledge base.",
            "key_files": [],
            "edit_targets": [],
        }

    preferences = infer_edit_preferences(task)
    targets = group_edit_targets(matches, preferences)
    targets = targets[:top_k]

    key_files = []
    seen_files = set()
    for target in targets:
        file_path = target.get("file_path")
        if file_path and file_path not in seen_files:
            seen_files.add(file_path)
            key_files.append(file_path)

    best_target = targets[0] if targets else None
    if best_target:
        summary = (
            f"Best edit target is {best_target.get('symbol_label')} in "
            f"{best_target.get('file_path')} via {best_target.get('recommended_action')}."
        )
    else:
        summary = "I could not determine a strong edit target."

    return {
        "summary": summary,
        "key_files": key_files,
        "edit_targets": targets,
    }