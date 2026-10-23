import json


def validate_gened_json(generated_text: str) -> dict:
    """Parse the generated JSON and validate its top-level structure."""
    required_keys = {"prompt", "name", "parameters"}

    try:
        result = json.loads(generated_text)
    except json.JSONDecodeError as alert:
        raise ValueError(f"Generated invalid JSON: {alert}") from alert

    if not isinstance(result, dict):
        raise ValueError("Generated output must be a JSON object")

    if set(result.keys()) != required_keys:
        raise ValueError(f"Expected exactly {required_keys}, "
                         f"got {set(result.keys())}")

    if not isinstance(result["prompt"], str):
        raise ValueError("'prompt' must be a string")

    if not isinstance(result["name"], str):
        raise ValueError("'name' must be a string")

    if not isinstance(result["parameters"], dict):
        raise ValueError("'parameters' must be an object")

    return result


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
