from collections import defaultdict
from typing import Dict, List, Tuple

from app.services.answer_service import clean_snippet, format_symbol_label
from app.services.edit_location_service import infer_edit_preferences, group_edit_targets
from app.services.knowledge_schema import parse_tag_string
from app.services.retrieval_router import normalize_text, tokenize_text


def _match_identity(match: Dict) -> Tuple:
    return (
        match.get("workspace_id"),
        match.get("file_path"),
        match.get("symbol_name"),
        match.get("symbol_type"),
        match.get("parent_symbol"),
    )


def _target_identity(target: Dict) -> Tuple:
    return (
        target.get("workspace_id"),
        target.get("file_path"),
        target.get("symbol_name"),
        target.get("symbol_type"),
        target.get("parent_symbol"),
    )


def infer_impact_category(match: Dict, direct_targets: List[Dict]) -> str:
    match_id = _match_identity(match)

    for target in direct_targets:
        target_id = _target_identity(target)

        if match_id == target_id:
            return "direct_edit_target"

        if (
            match.get("workspace_id") == target.get("workspace_id")
            and match.get("file_path") == target.get("file_path")
        ):
            return "same_file_neighbor"

    scope_kind = normalize_text(match.get("scope_kind", ""))

    if scope_kind == "wrapper":
        return "wrapper_review"

    if scope_kind in {"abstraction", "shared"}:
        return "shared_or_base_review"

    return "related_symbol_review"


def score_impact_candidate(match: Dict, direct_targets: List[Dict], preferences: Dict) -> Tuple[float, List[str], str]:
    reasons: List[str] = []

    score = float(match.get("routing_score", 0.0))
    if "routing_score" not in match:
        score += -float(match.get("distance", 999))

    match_workspace_id = match.get("workspace_id")
    match_file_path = match.get("file_path")
    match_symbol_name = normalize_text(match.get("symbol_name", ""))
    match_parent_symbol = normalize_text(match.get("parent_symbol", ""))
    match_scope_kind = normalize_text(match.get("scope_kind", ""))
    match_symbol_type = normalize_text(match.get("symbol_type", ""))
    match_role_tags = set(parse_tag_string(match.get("role_tags", "")))
    match_system_tags = set(parse_tag_string(match.get("system_tags", "")))
    match_style_tags = set(parse_tag_string(match.get("style_tags", "")))

    task_role_tags = set(preferences.get("role_tags", []))
    task_system_tags = set(preferences.get("system_tags", []))
    task_tokens = set(preferences.get("task_tokens", set()))
    task_l = normalize_text(preferences.get("task", ""))

    impact_category = infer_impact_category(match, direct_targets)

    if impact_category == "direct_edit_target":
        score += 3.0
        reasons.append("direct_edit_target")
    elif impact_category == "same_file_neighbor":
        score += 2.0
        reasons.append("same_file_neighbor")
    elif impact_category == "wrapper_review":
        score += 1.1
        reasons.append("wrapper_review")
    elif impact_category == "shared_or_base_review":
        score += 1.0
        reasons.append("shared_or_base_review")
    else:
        reasons.append("related_symbol_review")

    if match_symbol_type in {"function", "method"}:
        score += 0.8
        reasons.append("callable_symbol")
    elif match_symbol_type == "class":
        score += 0.5
        reasons.append("class_symbol")

    for target in direct_targets:
        target_workspace_id = target.get("workspace_id")
        target_file_path = target.get("file_path")
        target_symbol_name = normalize_text(target.get("symbol_name", ""))
        target_parent_symbol = normalize_text(target.get("parent_symbol", ""))
        target_scope_kind = normalize_text(target.get("scope_kind", ""))
        target_role_tags = set(parse_tag_string(target.get("role_tags", "")))
        target_system_tags = set(parse_tag_string(target.get("system_tags", "")))

        if match_workspace_id == target_workspace_id:
            score += 0.5
            reasons.append("same_workspace_as_direct_target")

        if match_file_path and match_file_path == target_file_path:
            score += 1.5
            reasons.append("same_file_as_direct_target")

        if match_parent_symbol and match_parent_symbol == target_symbol_name:
            score += 1.2
            reasons.append("same_parent_class_as_direct_target")

        if target_parent_symbol and match_parent_symbol == target_parent_symbol:
            score += 0.8
            reasons.append("same_parent_scope")

        if match_symbol_name and match_symbol_name == target_symbol_name:
            score += 0.8
            reasons.append("same_symbol_name_pattern")

        role_overlap_with_target = sorted(match_role_tags.intersection(target_role_tags))
        if role_overlap_with_target:
            score += min(1.6, len(role_overlap_with_target) * 0.5)
            reasons.append(f"role_overlap_with_direct:{','.join(role_overlap_with_target[:4])}")

        system_overlap_with_target = sorted(match_system_tags.intersection(target_system_tags))
        if system_overlap_with_target:
            score += min(1.6, len(system_overlap_with_target) * 0.8)
            reasons.append(f"system_overlap_with_direct:{','.join(system_overlap_with_target[:3])}")

        if match_scope_kind and target_scope_kind and match_scope_kind == target_scope_kind:
            score += 0.6
            reasons.append("same_scope_as_direct_target")

    role_overlap = sorted(match_role_tags.intersection(task_role_tags))
    if role_overlap:
        score += min(1.8, len(role_overlap) * 0.6)
        reasons.append(f"task_role_overlap:{','.join(role_overlap[:4])}")

    system_overlap = sorted(match_system_tags.intersection(task_system_tags))
    if system_overlap:
        score += min(2.0, len(system_overlap) * 0.9)
        reasons.append(f"task_system_overlap:{','.join(system_overlap[:3])}")

    symbol_blob = normalize_text(
        " ".join(
            [
                str(match.get("file_name", "") or ""),
                str(match.get("file_path", "") or ""),
                str(match.get("symbol_name", "") or ""),
                str(match.get("parent_symbol", "") or ""),
            ]
        )
    )
    token_overlap = sorted(task_tokens.intersection(set(tokenize_text(symbol_blob))))
    if token_overlap:
        score += min(1.5, len(token_overlap) * 0.3)
        reasons.append(f"task_token_overlap:{','.join(token_overlap[:5])}")

    if "route" in task_l or "endpoint" in task_l or "api" in task_l:
        if "routes.py" in normalize_text(match_file_path or ""):
            score += 1.2
            reasons.append("route_layer_review")

    if "service" in task_l and "service" in normalize_text(match_file_path or ""):
        score += 0.8
        reasons.append("service_layer_review")

    if "platform_style" in match_style_tags and any(word in task_l for word in ["shared", "common", "platform style"]):
        score += 0.8
        reasons.append("platform_style_alignment")

    reasons = list(dict.fromkeys(reasons))
    return score, reasons[:10], impact_category


def group_impacted_files(scored_matches: List[Dict], top_k: int = 6) -> List[Dict]:
    file_groups: Dict[str, Dict] = {}

    for match in scored_matches:
        file_path = match.get("file_path")
        if not file_path:
            continue

        group = file_groups.get(file_path)
        if not group:
            group = {
                "workspace_id": match.get("workspace_id"),
                "project_name": match.get("project_name"),
                "file_path": file_path,
                "file_name": match.get("file_name"),
                "impact_score": match.get("impact_score", 0.0),
                "impact_categories": [match.get("impact_category")],
                "top_symbols": [],
                "reasons": list(match.get("impact_reasons", [])),
                "match_count": 0,
            }
            file_groups[file_path] = group

        group["match_count"] += 1
        group["impact_score"] = max(group["impact_score"], match.get("impact_score", 0.0))
        group["impact_categories"] = list(dict.fromkeys(group["impact_categories"] + [match.get("impact_category")]))
        group["reasons"] = list(dict.fromkeys(group["reasons"] + match.get("impact_reasons", [])))[:10]

        symbol_label = format_symbol_label(match)
        symbol_entry = {
            "symbol_label": symbol_label,
            "symbol_name": match.get("symbol_name"),
            "symbol_type": match.get("symbol_type"),
            "parent_symbol": match.get("parent_symbol"),
            "impact_score": match.get("impact_score"),
            "impact_category": match.get("impact_category"),
        }

        existing_labels = {entry["symbol_label"] for entry in group["top_symbols"]}
        if symbol_label not in existing_labels:
            group["top_symbols"].append(symbol_entry)

    impacted_files = list(file_groups.values())
    impacted_files.sort(key=lambda item: item.get("impact_score", 0), reverse=True)

    for item in impacted_files:
        item["top_symbols"].sort(key=lambda s: s.get("impact_score", 0), reverse=True)
        item["top_symbols"] = item["top_symbols"][:5]

    return impacted_files[:top_k]


def build_review_checklist(task: str, direct_targets: List[Dict], impacted_files: List[Dict]) -> List[str]:
    checklist: List[str] = []

    if direct_targets:
        best = direct_targets[0]
        checklist.append(
            f"Start with the direct edit target: {best.get('symbol_label')} in {best.get('file_path')}."
        )

    if any(target.get("scope_kind") == "wrapper" for target in direct_targets):
        checklist.append("Review wrapper helpers and related cancel/modify/place flows in the same workspace.")

    if any(target.get("scope_kind") in {"abstraction", "shared"} for target in direct_targets):
        checklist.append("Review shared/base helpers because changes may affect more than one system path.")

    if "auth" in normalize_text(task):
        checklist.append("Verify login/session/token handling paths and any dependent request helpers.")

    if "order" in normalize_text(task):
        checklist.append("Review order placement, modify, cancel, target, and stoploss helper paths for side effects.")

    if impacted_files:
        secondary = [item["file_path"] for item in impacted_files[1:4]]
        if secondary:
            checklist.append("Also review likely secondary files: " + ", ".join(secondary) + ".")

    if not checklist:
        checklist.append("Review the direct target first, then inspect same-file neighbors and related helpers.")

    return checklist[:6]


def build_change_impact_response(task: str, matches: List[Dict], top_k: int = 6) -> Dict:
    if not matches:
        return {
            "summary": "I could not find enough relevant code to analyze change impact.",
            "direct_edit_targets": [],
            "impacted_files": [],
            "related_symbols": [],
            "review_checklist": [],
        }

    preferences = infer_edit_preferences(task)
    direct_targets = group_edit_targets(matches, preferences)[: min(4, top_k)]

    scored_matches: List[Dict] = []
    direct_target_ids = {_target_identity(target) for target in direct_targets}

    for match in matches:
        impact_score, impact_reasons, impact_category = score_impact_candidate(
            match=match,
            direct_targets=direct_targets,
            preferences=preferences,
        )

        enriched = dict(match)
        enriched["impact_score"] = impact_score
        enriched["impact_reasons"] = impact_reasons
        enriched["impact_category"] = impact_category
        scored_matches.append(enriched)

    scored_matches.sort(
        key=lambda item: (
            item.get("impact_score", 0),
            item.get("routing_score", 0) if item.get("routing_score") is not None else -999,
            -(item.get("distance", 999)),
        ),
        reverse=True,
    )

    impacted_files = group_impacted_files(scored_matches, top_k=top_k)

    related_symbols = []
    seen_related = set()
    for match in scored_matches:
        match_id = _match_identity(match)
        if match_id in direct_target_ids:
            continue

        if match_id in seen_related:
            continue
        seen_related.add(match_id)

        related_symbols.append(
            {
                "workspace_id": match.get("workspace_id"),
                "project_name": match.get("project_name"),
                "file_path": match.get("file_path"),
                "file_name": match.get("file_name"),
                "symbol_name": match.get("symbol_name"),
                "symbol_type": match.get("symbol_type"),
                "parent_symbol": match.get("parent_symbol"),
                "symbol_label": format_symbol_label(match),
                "scope_kind": match.get("scope_kind"),
                "impact_category": match.get("impact_category"),
                "impact_score": match.get("impact_score"),
                "impact_reasons": match.get("impact_reasons"),
                "snippet": clean_snippet(match.get("content", "")),
            }
        )

        if len(related_symbols) >= top_k:
            break

    direct_edit_targets = []
    for target in direct_targets:
        direct_edit_targets.append(
            {
                "workspace_id": target.get("workspace_id"),
                "project_name": target.get("project_name"),
                "file_path": target.get("file_path"),
                "file_name": target.get("file_name"),
                "symbol_name": target.get("symbol_name"),
                "symbol_type": target.get("symbol_type"),
                "parent_symbol": target.get("parent_symbol"),
                "symbol_label": target.get("symbol_label"),
                "scope_kind": target.get("scope_kind"),
                "edit_score": target.get("edit_score"),
                "edit_reasons": target.get("edit_reasons"),
                "recommended_action": target.get("recommended_action"),
                "snippet": target.get("snippet"),
            }
        )

    if direct_edit_targets:
        best = direct_edit_targets[0]
        summary = (
            f"Primary impact starts at {best.get('symbol_label')} in {best.get('file_path')}. "
            f"Review same-file neighbors and related helper/wrapper paths listed below."
        )
    else:
        summary = "I found related files, but could not isolate a strong primary edit target."

    review_checklist = build_review_checklist(
        task=task,
        direct_targets=direct_targets,
        impacted_files=impacted_files,
    )

    return {
        "summary": summary,
        "direct_edit_targets": direct_edit_targets,
        "impacted_files": impacted_files,
        "related_symbols": related_symbols,
        "review_checklist": review_checklist,
    }