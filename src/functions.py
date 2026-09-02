import math
import re


def add_numbers(a: float, b: float) -> float:
    return float(a) + float(b)


def greet(name: str) -> str:
    return f"Hello, {name}!"


def reverse_string(s: str) -> str:
    return s[::-1]


def square_root(a: float) -> float:
    return math.sqrt(float(a))


def string_to_regex(source_string: str, regex: str, replacement: str) -> str:
    return re.sub(regex, replacement, source_string)


FUNC_DISPATCH = {
    "fn_add_numbers": add_numbers,
    "fn_greet": greet,
    "fn_reverse_string": reverse_string,
    "fn_get_square_root": square_root,
    "fn_substitute_string_with_regex": string_to_regex
}
