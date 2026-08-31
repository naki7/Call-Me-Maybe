import json
from pathlib import Path
from typing import Any

DIRECTORY = Path('./data/output')
BASE_FILE = DIRECTORY / "function_calling_results.json"
BACKUP_FILE = DIRECTORY / "function_calling_results_backup.json"


def check_directory() -> None:
    DIRECTORY.mkdir(parents=True, exist_ok=True)


def json_to_obj(file: str) -> Any:
    try:
        with open(file, "r", encoding="utf-8") as text:
            return json.load(text)
    except FileNotFoundError:
        print(f"JSON file not found: {file}")
        quit()
    except PermissionError:
        print(f"Permissions denied while opening: {file}")
        quit()
    except json.JSONDecodeError as alert:
        print(f"Invalid JSON in: {file} - {alert}")

    return {}


def obj_to_json(object: dict[Any, Any]) -> None:
    check_directory()

    try:
        with open(BASE_FILE, 'x', encoding="utf-8") as file:
            json.dump(object, file, indent=2)
            file.write('\n')
        print(f'{BASE_FILE} successfully written.')
    except FileExistsError:
        try:
            with open(BACKUP_FILE, 'x', encoding="utf-8") as file:
                json.dump(object, file, indent=2)
                file.write('\n')
            print(f'{BASE_FILE} already exists. {BACKUP_FILE} successfully',
                  'written')
        except FileExistsError:
            print(f'Both {BASE_FILE} file and {BACKUP_FILE} file exist',
                  '\nRemove or rename them before rerunning.')
