import json
from pathlib import Path
from typing import Any


def json_to_obj(path: str | Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as alert:
        raise FileNotFoundError(f"JSON file not found: {path}") from alert
    except PermissionError as alert:
        raise PermissionError(
            f"Permissions denied while opening: {path}") from alert
    except json.JSONDecodeError as alert:
        raise ValueError(f"Invalid JSON in: {path} - {alert}") from alert


def check_directory(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def obj_to_json(object: Any, path: str | Path) -> None:
    check_directory(path)

    with open(path, 'w', encoding="utf-8") as file:
        json.dump(object, file, indent=2)
        file.write('\n')
    print(f'{path} successfully written.')
