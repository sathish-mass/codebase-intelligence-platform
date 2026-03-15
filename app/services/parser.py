from pathlib import Path
from typing import List, Dict


ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".c",
    ".json", ".yaml", ".yml", ".md", ".txt", ".sql"
}


def detect_source_type(file_path: Path) -> str:
    """
    Classify file type so later retrieval/generation can distinguish
    between code, docs, API specs, and config-like files.
    """
    name = file_path.name.lower()
    suffix = file_path.suffix.lower()

    if "openapi" in name or "swagger" in name:
        return "api_spec"

    if suffix in {".md", ".txt"}:
        return "documentation"

    if suffix in {".yaml", ".yml", ".json"}:
        return "config"

    return "code"


def parse_codebase(base_path: str) -> List[Dict[str, str]]:
    """
    Scan a folder recursively and return readable code/docs/spec files.
    """
    root = Path(base_path)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path}")

    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base_path}")

    documents: List[Dict[str, str]] = []

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        if any(part in {".git", "__pycache__", "venv", ".venv", "node_modules"} for part in file_path.parts):
            continue

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except Exception:
            continue

        documents.append(
            {
                "path": str(file_path),
                "file_name": file_path.name,
                "source_type": detect_source_type(file_path),
                "content": content
            }
        )

    return documents