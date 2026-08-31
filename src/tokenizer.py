import json
from typing import Any


def format_funcs(functions: list[dict[str, Any]]) -> str:
    result = []

    for func in functions:
        result.append(
            {
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters", {}),
            }
        )

    return json.dumps(result, indent=2)


def parse_funcs(base_text: str) -> dict[str, Any]:
    stripped = base_text.strip()

    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "", 1).strip()

    return json.loads(stripped)
