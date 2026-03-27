from typing import Dict, List

from app.services.answer_service import format_symbol_label
from app.services.change_impact_service import build_change_impact_response
from app.services.edit_location_service import build_edit_location_response
from app.services.kt_service import generate_kt_report
from app.services.retrieval_router import normalize_text


def _build_step_sequence(
    task: str,
    direct_edit_targets: List[Dict],
    impacted_files: List[Dict],
    modules_summary: List[Dict],
) -> List[str]:
    steps: List[str] = []

    if direct_edit_targets:
        primary = direct_edit_targets[0]
        steps.append(
            f"Start with the primary implementation target: {primary.get('symbol_label')} in {primary.get('file_path')}."
        )

    if len(direct_edit_targets) > 1:
        secondary_targets = [
            target.get("symbol_label")
            for target in direct_edit_targets[1:4]
            if target.get("symbol_label")
        ]
        if secondary_targets:
            steps.append(
                "Review nearby implementation targets before coding: " + ", ".join(secondary_targets) + "."
            )

    if impacted_files:
        secondary_files = [
            item.get("file_path")
            for item in impacted_files[1:4]
            if item.get("file_path")
        ]
        if secondary_files:
            steps.append(
                "Update or review supporting files after the primary change: " + ", ".join(secondary_files) + "."
            )

    task_l = normalize_text(task)

    if "wrapper" in task_l:
        steps.append("Implement the wrapper-layer change first, then verify downstream helper or shared methods that depend on it.")

    if "shared" in task_l or "common" in task_l or "base" in task_l:
        steps.append("Apply the change in the shared/base layer carefully, then validate all dependent flows using the same helper path.")

    if "order" in task_l:
        steps.append("Verify order placement, modify, cancel, stoploss, and target-related methods for consistency after the change.")

    if "auth" in task_l:
        steps.append("Re-check authentication/session setup and any dependent API request methods after implementation.")

    if modules_summary:
        important_modules = [module.get("file_path") for module in modules_summary[:3] if module.get("file_path")]
        if important_modules:
            steps.append("Use these modules as the main implementation context: " + ", ".join(important_modules) + ".")

    return steps[:8]


def _build_validation_checklist(task: str, direct_edit_targets: List[Dict], impacted_files: List[Dict]) -> List[str]:
    checklist: List[str] = []
    task_l = normalize_text(task)

    if direct_edit_targets:
        primary = direct_edit_targets[0]
        checklist.append(
            f"Confirm the updated behavior in the primary target: {primary.get('symbol_label')}."
        )

    if "order" in task_l:
        checklist.append("Test placement, modify, cancel, stoploss, and target paths affected by the change.")
        checklist.append("Verify status/result payloads and any orderbook updates remain consistent.")

    if "wrapper" in task_l:
        checklist.append("Test wrapper input/output handling and validate broker/API-facing calls.")

    if "shared" in task_l or "common" in task_l or "base" in task_l:
        checklist.append("Test multiple flows that reuse the same shared/base helper to ensure there is no cross-flow regression.")

    if "auth" in task_l:
        checklist.append("Validate login/session/token behavior and any retry or failure-handling paths.")

    if impacted_files and len(impacted_files) > 1:
        checklist.append("Review secondary impacted files and run targeted regression checks for those modules.")

    if not checklist:
        checklist.append("Validate the primary implementation target and review nearby impacted methods for regressions.")

    # dedupe
    deduped = []
    seen = set()
    for item in checklist:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped[:8]


def _build_risk_notes(task: str, direct_edit_targets: List[Dict], impacted_files: List[Dict]) -> List[str]:
    notes: List[str] = []
    task_l = normalize_text(task)

    if len(impacted_files) > 1:
        notes.append("This change may affect more than one module, so shared assumptions and helper reuse should be reviewed.")

    if any(target.get("scope_kind") == "wrapper" for target in direct_edit_targets):
        notes.append("Wrapper-layer changes can affect downstream broker/API interactions and may require end-to-end validation.")

    if any(target.get("scope_kind") in {"shared", "abstraction"} for target in direct_edit_targets):
        notes.append("Shared/base-layer changes may introduce regressions across multiple workflows that reuse the same helper path.")

    if "order" in task_l:
        notes.append("Order-flow changes may affect placement, modify, cancel, quantity normalization, and status/result handling.")

    if "auth" in task_l:
        notes.append("Authentication-related changes can break dependent request flows even if the primary module change appears small.")

    if not notes:
        notes.append("Review the impacted modules carefully and verify adjacent helper flows after implementation.")

    # dedupe
    deduped = []
    seen = set()
    for item in notes:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped[:6]


def build_implementation_plan(task: str, matches: List[Dict], top_k: int = 6) -> Dict:
    if not matches:
        return {
            "summary": "I could not find enough relevant code to build an implementation plan.",
            "primary_target": None,
            "supporting_modules": [],
            "implementation_steps": [],
            "validation_checklist": [],
            "risk_notes": [],
        }

    edit_result = build_edit_location_response(task=task, matches=matches, top_k=top_k)
    impact_result = build_change_impact_response(task=task, matches=matches, top_k=top_k)
    kt_result = generate_kt_report(
        files=[match.get("file_path") for match in matches],
        matches=matches,
        task=task,
    )

    direct_edit_targets = impact_result.get("direct_edit_targets", []) or edit_result.get("edit_targets", [])
    impacted_files = impact_result.get("impacted_files", [])
    modules_summary = kt_result.get("modules_summary", [])

    primary_target = direct_edit_targets[0] if direct_edit_targets else None

    supporting_modules = []
    seen_files = set()

    for item in impacted_files:
        file_path = item.get("file_path")
        if not file_path or file_path in seen_files:
            continue
        seen_files.add(file_path)

        supporting_modules.append(
            {
                "file_path": file_path,
                "file_name": item.get("file_name"),
                "impact_categories": item.get("impact_categories", []),
                "top_symbols": item.get("top_symbols", []),
                "reasons": item.get("reasons", []),
            }
        )

        if len(supporting_modules) >= top_k:
            break

    implementation_steps = _build_step_sequence(
        task=task,
        direct_edit_targets=direct_edit_targets,
        impacted_files=impacted_files,
        modules_summary=modules_summary,
    )

    validation_checklist = _build_validation_checklist(
        task=task,
        direct_edit_targets=direct_edit_targets,
        impacted_files=impacted_files,
    )

    risk_notes = _build_risk_notes(
        task=task,
        direct_edit_targets=direct_edit_targets,
        impacted_files=impacted_files,
    )

    if primary_target:
        summary = (
            f"Implement the change starting at {primary_target.get('symbol_label')} "
            f"in {primary_target.get('file_path')}, then review the supporting modules listed below."
        )
    else:
        summary = "I found related implementation areas, but could not isolate a strong primary target."

    return {
        "summary": summary,
        "primary_target": primary_target,
        "supporting_modules": supporting_modules,
        "implementation_steps": implementation_steps,
        "validation_checklist": validation_checklist,
        "risk_notes": risk_notes,
        "kt_modules_summary": modules_summary[: min(4, len(modules_summary))],
    }