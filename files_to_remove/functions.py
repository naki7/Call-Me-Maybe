import re


def fn_add_numbers(a: float, b: float) -> float:
    result = a + b
    return result


def fn_greet(name: str) -> str:
    result = f"Hello, {name}!"
    return result


def fn_reverse_string(s: str) -> str:
    result = ""

    for char in s:
        result = f"{char}{result}"

    return result


def fn_get_square_root(a: float) -> float:
    result = a ** 0.5
    return result


def fn_substitute_string_with_regex(source_string: str, regex: str,
                                    replacement: str) -> str:
    result = re.sub(regex, replacement, source_string)
    return result
