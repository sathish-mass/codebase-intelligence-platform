import re
from typing import Dict, List


def to_snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "generated_item"


def infer_generation_type(task: str) -> str:
    task_lower = task.lower()

    if any(word in task_lower for word in ["route", "endpoint", "api"]):
        return "fastapi_route"

    if any(word in task_lower for word in ["service", "logic", "function"]):
        return "service_function"

    return "generic_python"


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


def build_generation_output(task: str, matches: List[Dict]) -> Dict:
    generation_type = infer_generation_type(task)
    slug = to_snake_case(task)[:50]

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
        "This is a grounded skeleton based on retrieved project context.",
        "You should refine field names, validation rules, and business logic.",
        "Later we can upgrade this to true LLM-based code generation."
    ]

    return {
        "task": task,
        "generation_type": generation_type,
        "target_file": target_file,
        "generated_code": generated_code,
        "references": references,
        "notes": notes
    }