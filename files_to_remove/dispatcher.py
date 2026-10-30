from typing import Any
from src.functions import (
    fn_add_numbers,
    fn_greet,
    fn_reverse_string,
    fn_get_square_root,
    fn_substitute_string_with_regex
)

FUNCTION_MAP = {
    "fn_add_numbers": fn_add_numbers,
    "fn_greet": fn_greet,
    "fn_reverse_string": fn_reverse_string,
    "fn_get_square_root": fn_get_square_root,
    "fn_substitute_string_with_regex": fn_substitute_string_with_regex,
}


def dispatch_func(call: dict) -> Any:
    """Execute a validated function call using a trusted function map."""
    func_name = call["name"]
    params = call["parameters"]
    func = FUNCTION_MAP.get(func_name)

    if func is None:
        raise ValueError(f"No executable function registered for {func_name}")

    return func(**params)
