from typing import Dict


def infer_task_type(prompt: str) -> str:
    """
    Infer the user's high-level task intent.

    Possible values:
    - generate
    - compare
    - summary
    - kt
    - ask
    """
    text = (prompt or "").strip().lower()

    if any(word in text for word in [
        "create ", "generate ", "build ", "write ", "implement ",
        "make ", "add ", "develop "
    ]):
        return "generate"

    if any(word in text for word in [
        "compare ", "difference ", "vs ", " versus "
    ]):
        return "compare"

    if any(word in text for word in [
        "summary", "summarize", "architecture", "overview"
    ]):
        return "summary"

    if any(word in text for word in [
        "kt", "handover", "knowledge transfer", "explain module", "onboarding"
    ]):
        return "kt"

    return "ask"


def build_task_routing(prompt: str) -> Dict:
    """
    Return a small routing plan for the prompt.
    """
    task_type = infer_task_type(prompt)

    return {
        "task_type": task_type,
        "original_prompt": prompt,
    }