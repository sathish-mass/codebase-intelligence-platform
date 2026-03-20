from pathlib import Path
from typing import Dict, Optional


def resolve_target_path(target_file: str, workspace_path: Optional[str] = None) -> Path:
    """
    Resolve target file path.
    - If workspace_path is provided, write inside that workspace.
    - Otherwise write relative to current project.
    """
    target = Path(target_file)

    if workspace_path:
        workspace = Path(workspace_path).resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        resolved = (workspace / target).resolve()

        # Prevent path traversal outside workspace
        if workspace not in resolved.parents and resolved != workspace:
            raise ValueError("Resolved target path escapes the workspace directory")

        return resolved

    return target.resolve()


def write_generated_code(
    target_file: str,
    generated_code: str,
    overwrite: bool = False,
    workspace_path: Optional[str] = None
) -> Dict:
    """
    Save generated code into a target file.
    - If workspace_path is provided, save inside that uploaded workspace.
    - If file is routes.py, append safely.
    - If file does not exist, create it.
    - If file exists and overwrite=False, do not replace it.
    """
    path = resolve_target_path(target_file, workspace_path=workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.name == "routes.py":
            existing = path.read_text(encoding="utf-8")
            new_content = (
                existing.rstrip()
                + "\n\n\n# ===== Generated Route Start =====\n"
                + generated_code
                + "\n# ===== Generated Route End =====\n"
            )
            path.write_text(new_content, encoding="utf-8")
            return {
                "status": "appended",
                "target_file": str(path),
                "message": "Generated route code appended to existing routes.py"
            }

        if not overwrite:
            return {
                "status": "skipped",
                "target_file": str(path),
                "message": "File already exists. Set overwrite=true to replace it."
            }

    path.write_text(generated_code, encoding="utf-8")
    return {
        "status": "written",
        "target_file": str(path),
        "message": "Generated code written successfully"
    }