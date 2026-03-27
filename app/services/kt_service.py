from collections import defaultdict, Counter
from typing import Dict, List

from app.services.edit_location_service import recommend_edit_action
from app.services.knowledge_schema import parse_tag_string
from app.services.retrieval_router import normalize_text


def _top_items(counter: Counter, limit: int = 5) -> List[str]:
    return [key for key, _ in counter.most_common(limit) if key]


def _safe_list_tags(matches: List[Dict], key: str) -> List[str]:
    counter = Counter()

    for match in matches:
        for tag in parse_tag_string(match.get(key, "")):
            if tag:
                counter[tag] += 1

    return _top_items(counter, limit=6)


def _safe_scope_counter(matches: List[Dict]) -> Counter:
    counter = Counter()
    for match in matches:
        scope = normalize_text(match.get("scope_kind", ""))
        if scope:
            counter[scope] += 1
    return counter


def _safe_symbol_counter(matches: List[Dict]) -> Counter:
    counter = Counter()
    for match in matches:
        symbol_name = (match.get("symbol_name") or "").strip()
        if symbol_name:
            counter[symbol_name] += 1
    return counter


def _infer_module_responsibility(matches: List[Dict], file_path: str) -> str:
    """
    Better responsibility inference using:
    - file path
    - scope kinds
    - role tags
    - system tags
    """
    path_l = normalize_text(file_path)
    role_tags = set(_safe_list_tags(matches, "role_tags"))
    system_tags = set(_safe_list_tags(matches, "system_tags"))
    scope_counter = _safe_scope_counter(matches)
    primary_scope = _top_items(scope_counter, limit=1)
    primary_scope = primary_scope[0] if primary_scope else "unknown"

    if "order" in role_tags or "placement" in role_tags or "stoploss" in role_tags or "target" in role_tags:
        if primary_scope == "wrapper":
            return "Provides wrapper-level order handling helpers, including placement, modification, cancel, and status-related flows."
        if primary_scope in {"shared", "abstraction"}:
            return "Provides shared/base order management logic used across placement, modify, cancel, stoploss, and target workflows."
        return "Handles order-related logic such as placement, modification, cancellation, and status management."

    if "auth" in role_tags:
        return "Handles authentication, token/session handling, and API access setup."

    if primary_scope == "wrapper":
        if system_tags:
            return f"Acts as a wrapper/integration layer for {', '.join(sorted(system_tags)[:3])} related functionality."
        return "Acts as a wrapper/integration layer around external broker or API functionality."

    if "support_library" in path_l or "support-library" in path_l or "library" in path_l:
        return "Contains shared support helpers, broker-facing utility methods, and reusable order-management utilities."

    if "helper" in role_tags:
        return "Contains reusable helper functions, utility methods, and common support logic used across the project."

    if "service" in path_l:
        return "Provides service-layer functionality such as orchestration, data access, and request handling."

    return "Contains module-specific implementation and supporting utility logic."


def _build_module_overview(file_path: str, matches: List[Dict]) -> str:
    role_tags = _safe_list_tags(matches, "role_tags")
    system_tags = _safe_list_tags(matches, "system_tags")
    style_tags = _safe_list_tags(matches, "style_tags")
    scope_counter = _safe_scope_counter(matches)
    symbol_counter = _safe_symbol_counter(matches)

    primary_scope = _top_items(scope_counter, limit=1)
    primary_scope = primary_scope[0] if primary_scope else "unknown"

    top_symbols = _top_items(symbol_counter, limit=4)

    parts = []

    if system_tags:
        parts.append(f"System focus: {', '.join(system_tags[:3])}.")
    if primary_scope != "unknown":
        parts.append(f"Primary scope: {primary_scope}.")
    if role_tags:
        parts.append(f"Main responsibilities: {', '.join(role_tags[:4])}.")
    if style_tags:
        parts.append(f"Style signals: {', '.join(style_tags[:3])}.")
    if top_symbols:
        parts.append(f"Important symbols: {', '.join(top_symbols[:4])}.")

    if not parts:
        parts.append("This module contains project-specific logic and relevant symbols retrieved for the current KT request.")

    return " ".join(parts)


def build_kt_summary(file_path: str, matches: List[Dict]) -> Dict:
    """
    Build one KT summary per unique file.
    """
    file_summary: Dict = {
        "file_path": file_path,
        "responsibility": _infer_module_responsibility(matches, file_path),
        "module_overview": _build_module_overview(file_path, matches),
        "function_ownership": [],
        "related_symbols": [],
    }

    module_functions = defaultdict(list)

    for match in matches:
        function_name = (match.get("symbol_name") or "").strip()
        if function_name:
            module_functions[function_name].append(match)

    function_entries = []
    for function_name, function_matches in module_functions.items():
        best_match = function_matches[0]

        associated_symbols = []
        seen_symbols = set()
        for match in function_matches:
            symbol_repr = f"{match.get('symbol_type', '')}: {match.get('symbol_name', '')}"
            if symbol_repr not in seen_symbols:
                seen_symbols.add(symbol_repr)
                associated_symbols.append(symbol_repr)

        role_tags = _safe_list_tags(function_matches, "role_tags")
        function_entries.append(
            {
                "function_name": function_name,
                "function_description": recommend_edit_action(best_match),
                "associated_symbols": associated_symbols,
                "scope_kind": best_match.get("scope_kind"),
                "role_tags": role_tags,
            }
        )

    function_entries.sort(key=lambda item: item["function_name"].lower())
    file_summary["function_ownership"] = function_entries[:12]

    related_symbols = []
    seen_related = set()
    for match in matches:
        related_symbol = (match.get("symbol_name") or "").strip()
        if related_symbol and related_symbol not in seen_related:
            seen_related.add(related_symbol)
            related_symbols.append(related_symbol)

    file_summary["related_symbols"] = related_symbols[:15]

    return file_summary


def build_kt_checklist(task: str, modules_summary: List[Dict]) -> List[str]:
    """
    Better KT checklist using task + discovered module traits.
    """
    task_l = normalize_text(task)
    checklist = []

    if "order" in task_l:
        checklist.append("Review order placement, modify, cancel, stoploss, and target handling across the relevant modules.")

    if "wrapper" in task_l:
        checklist.append("Review wrapper/integration methods and any broker-facing helper functions that call external APIs.")

    if "helper" in task_l:
        checklist.append("Check shared helper methods and support-library utilities that may be reused by multiple flows.")

    if "modify" in task_l:
        checklist.append("Inspect all modify/update paths along with any related orderbook or status update helpers.")

    if "auth" in task_l:
        checklist.append("Verify authentication, token/session handling, and any dependent request helpers.")

    has_wrapper = any(
        "Primary scope: wrapper." in (module.get("module_overview") or "")
        or "wrapper" in normalize_text(module.get("responsibility", ""))
        for module in modules_summary
    )
    has_shared = any(
        "Primary scope: shared." in (module.get("module_overview") or "")
        or "Primary scope: abstraction." in (module.get("module_overview") or "")
        for module in modules_summary
    )

    if has_wrapper:
        checklist.append("Review wrapper-layer functions together with same-file neighbor methods that share the same broker/system flow.")

    if has_shared:
        checklist.append("Review shared/base helpers because changes here may affect multiple workflows or systems.")

    secondary_files = [module["file_path"] for module in modules_summary[1:4] if module.get("file_path")]
    if secondary_files:
        checklist.append("Also review related modules: " + ", ".join(secondary_files) + ".")

    if not checklist:
        checklist.append("Review the primary module first, then inspect related symbols and neighboring helper methods.")

    # dedupe while preserving order
    deduped = []
    seen = set()
    for item in checklist:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped[:6]


def generate_kt_report(files: List[str], matches: List[Dict], task: str) -> Dict:
    """
    Generate the KT report with:
    - one summary per unique file
    - richer module responsibility
    - populated module overview
    - improved review checklist
    """
    unique_files = []
    seen_files = set()

    for file_path in files:
        if not file_path or file_path in seen_files:
            continue
        seen_files.add(file_path)
        unique_files.append(file_path)

    modules_summary = []
    for file_path in unique_files:
        file_matches = [match for match in matches if match.get("file_path") == file_path]
        if file_matches:
            modules_summary.append(build_kt_summary(file_path, file_matches))

    modules_summary.sort(
        key=lambda item: (
            -len(item.get("function_ownership", [])),
            item.get("file_path", ""),
        )
    )

    kt_report: Dict = {
        "task": task,
        "modules_summary": modules_summary,
        "review_checklist": build_kt_checklist(task, modules_summary),
    }

    return kt_report