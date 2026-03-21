import re
from typing import Dict, List

from app.services.llm_service import ask_huggingface_llm


def to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "generated_item"


def infer_generation_type(task: str) -> str:
    task_lower = task.lower()

    if any(word in task_lower for word in ["route", "endpoint", "api"]):
        return "fastapi_route"

    if any(word in task_lower for word in ["service", "logic", "function", "helper"]):
        return "service_function"

    return "generic_python"


def extract_meaningful_name(task: str) -> str:
    task_lower = task.lower().strip()

    cleanup_phrases = [
        "create a ",
        "create an ",
        "create ",
        "generate a ",
        "generate an ",
        "generate ",
        "write a ",
        "write an ",
        "write ",
        "build a ",
        "build an ",
        "build ",
        "helper function for ",
        "function for ",
        "helper for ",
        "similar to the existing order placement style",
        "similar to existing order placement style",
        "similar to existing style",
        "similar to the current style",
        "using existing project style",
        "following existing project style",
    ]

    for phrase in cleanup_phrases:
        task_lower = task_lower.replace(phrase, " ")

    task_lower = re.sub(r"\b(in|for|with|using|based on|similar to)\b.*$", "", task_lower).strip()

    slug = to_snake_case(task_lower)

    if not slug or slug in {"create", "generate", "write", "build"}:
        return "generated_item"

    words = slug.split("_")

    if len(words) > 6:
        words = words[:6]

    return "_".join(words)


def build_fastapi_route(task: str, route_name: str) -> str:
    model_name = "".join(part.capitalize() for part in route_name.split("_")) + "Request"

    return f'''from pydantic import BaseModel


class {model_name}(BaseModel):
    # TODO: define request fields
    name: str


def {route_name}_service(payload: {model_name}) -> dict:
    """
    TODO: move this into app/services/ if needed.
    """
    return {{
        "message": "{task}",
        "data": payload.dict()
    }}


@router.post("/{route_name.replace("_", "-")}")
def {route_name}(request: {model_name}):
    result = {route_name}_service(request)
    return result
'''


def build_service_function(task: str, func_name: str) -> str:
    return f'''def {func_name}(input_data: dict) -> dict:
    """
    {task}
    """
    # TODO: implement business logic
    # TODO: validate input_data
    # TODO: connect this with routes or workers if needed

    result = {{
        "message": "{task}",
        "input": input_data
    }}

    return result
'''


def build_generic_python(task: str, func_name: str) -> str:
    return f'''def {func_name}(input_data):
    """
    {task}
    """
    # TODO: implement logic
    return {{
        "message": "{task}",
        "input": input_data
    }}
'''


def fallback_generation(task: str, matches: List[Dict]) -> Dict:
    generation_type = infer_generation_type(task)
    slug = extract_meaningful_name(task)

    if generation_type == "fastapi_route":
        target_file = "app/api/routes.py"
        generated_code = build_fastapi_route(task, slug)
    elif generation_type == "service_function":
        target_file = f"app/services/{slug}.py"
        generated_code = build_service_function(task, slug)
    else:
        target_file = f"app/services/{slug}.py"
        generated_code = build_generic_python(task, slug)

    references = []
    for match in matches[:3]:
        references.append(
            {
                "file_path": match.get("file_path"),
                "source_type": match.get("source_type"),
                "chunk_index": match.get("chunk_index"),
            }
        )

    notes = [
        "LLM generation failed, so fallback template generation was used.",
        "You should refine field names, validation rules, and business logic."
    ]

    return {
        "task": task,
        "generation_type": generation_type,
        "target_file": target_file,
        "generated_code": generated_code,
        "references": references,
        "notes": notes
    }


def build_context_from_matches(matches: List[Dict], max_items: int = 5) -> str:
    context_blocks = []

    for idx, match in enumerate(matches[:max_items], start=1):
        file_path = match.get("file_path", "unknown")
        source_type = match.get("source_type", "unknown")
        chunk_index = match.get("chunk_index", "unknown")
        content = match.get("content", "")

        context_blocks.append(
            f"""[Reference {idx}]
File: {file_path}
Type: {source_type}
Chunk: {chunk_index}

{content}
"""
        )

    return "\n\n".join(context_blocks)


def choose_target_file(task: str) -> str:
    generation_type = infer_generation_type(task)
    slug = extract_meaningful_name(task)

    if generation_type == "fastapi_route":
        return "app/api/routes.py"

    return f"app/services/{slug}.py"


def build_generation_prompt(task: str, matches: List[Dict]) -> str:
    context_text = build_context_from_matches(matches)

    return f"""
You are an AI backend engineering assistant.

Your task is to generate code for this request:

{task}

Use the retrieved project context below as the style and structure reference.

Rules:
- Follow the coding style and architecture patterns from the context.
- Reuse naming conventions similar to the existing project.
- Generate only code, no explanation.
- Do not wrap the answer in markdown fences.
- If the task is about an API route, generate FastAPI-compatible code.
- If the task is about a function/service, generate Python code that matches existing helper/service style.

Retrieved context:
{context_text}
""".strip()


def build_generation_output(task: str, matches: List[Dict]) -> Dict:
    references = []
    for match in matches[:5]:
        references.append(
            {
                "file_path": match.get("file_path"),
                "source_type": match.get("source_type"),
                "chunk_index": match.get("chunk_index"),
            }
        )

    target_file = choose_target_file(task)

    try:
        prompt = build_generation_prompt(task, matches)
        generated_code = ask_huggingface_llm(prompt)

        return {
            "task": task,
            "generation_type": infer_generation_type(task),
            "target_file": target_file,
            "generated_code": generated_code,
            "references": references,
            "notes": [
                "This code was generated using retrieved project context and Hugging Face LLM output.",
                "Review carefully before saving into the codebase."
            ]
        }

    except Exception:
        return fallback_generation(task, matches)