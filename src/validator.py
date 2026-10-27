import json
from pydantic import ValidationError

from src.schemas import FunctionCall, FunctionDefinition


def load_registry(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            text = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Invalid file path in processing: {path}")

    validated_registry = [
        FunctionDefinition.model_validate(item) for item in text
    ]

    return [
        function.model_dump() for function in validated_registry
    ]


def validate_gened_json(generated_text: str) -> dict:
    try:
        call = FunctionCall.model_validate_json(generated_text)
    except ValidationError as alert:
        raise ValueError(
            f"Generated function call is invalid: {alert}"
        ) from alert

    return call.model_dump()


def validate_func_call(call: dict, registry: list[dict]) -> dict:
    """Validates the parsed function call compared to what is in the
    funciton registry"""

    # Validate Function
    func_name = call["name"]

    func = next(
        (
            function for function in registry if function["name"] == func_name
        ),
        None
    )

    if func is None:
        raise ValueError(f"Unknown function: {func_name}")

    # Validate parameters and arguments
    expected_params = func["parameters"]
    gen_params = call["parameters"]

    expected_vars = set(expected_params.keys())
    gen_vars = set(gen_params.keys())

    ignored_vars = expected_vars - gen_vars
    abnormal_vars = gen_vars - expected_vars

    if ignored_vars:
        raise ValueError(f"Missing parameters: {sorted(ignored_vars)}")

    if abnormal_vars:
        raise ValueError(f"Abnormal parameters: {sorted(abnormal_vars)}")

    # Validate argument typing
    for param_name, param_definition in expected_params.items():
        expected_type = param_definition["type"]
        value = gen_params[param_name]

        if expected_type == "number":
            valid = (isinstance(value, (int, float))
                     and not isinstance(value, bool))
        elif expected_type == "string":
            valid = isinstance(value, str)
        elif expected_type == "boolean":
            valid = isinstance(value, bool)
        elif expected_type == "list":
            valid = isinstance(value, list)
        elif expected_type == "dictionary":
            valid = isinstance(value, dict)
        else:
            raise ValueError(f"Unsupported parameter type "
                             f"'{expected_type}' for '{param_name}'")

        if not valid:
            raise ValueError(
                f"Parameter '{param_name}' must be "
                f"of type '{expected_type}', got {type(value).__name__}"
            )
    return call
