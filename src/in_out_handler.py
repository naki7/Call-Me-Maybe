import json
from typing import Any


def json_to_obj(file: str) -> dict[Any, Any]:
    dict_result: dict[Any, Any] = {}
    direct_text: str = ''

    try:
        with open(file, "rt") as text:
            direct_text = text.read()
    except FileNotFoundError:
        print("JSON file could not be found")
        quit()
    except PermissionError:
        print("JSON file permissions do not allow access")
        quit()

    dict_result = json.loads(direct_text)

    return dict_result


def obj_to_json(object: dict[Any, Any]) -> None:
    out_name: str = './data/output/function_calling_results.json'
    backup_name: str = './data/output/function_calling_results_backup.json'

    try:
        with open(out_name, 'x') as file:
            file.write(json.dumps(object))
            print('function_calling_results.json successfully written.')
    except FileExistsError:
        try:
            with open(backup_name, 'x') as file:
                file.write(json.dumps(object))
            print('function_calling_results_.json already exists. Switching',
                  'to function_calling_results_backup.json.\nPlease note,',
                  'there are no further backup files.\nPlease Move or Delete,',
                  'current files')
        except FileExistsError:
            print('Both the function_calling_results.json file and the',
                  'function_calling_results_backup.json file have already',
                  'been created and used.\nPlease Move or Delete these files',
                  'before running the program again.')
