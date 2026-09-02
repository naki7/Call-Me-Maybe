import json
import sys
import re
import ast
from typing import Any

from llm_sdk import Small_LLM_Model
from src.in_out_handler import json_to_obj, obj_to_json
import src.functions as funcs
from src.llms import generate


def func_select_prompt(user_input: str,
                       functions: list[dict[str, Any]]) -> str:
    registry = json.dumps(
        [
            {
                "name": func.get("name"),
                "description": func.get("description"),
                "parameters": func.get("parameters", {}),
            }
            for func in functions
        ],
        indent=2
    )

    expect_shape = json.dumps(
        {
            "function_name": "fn_add_numbers",
            "arguments": {"a": 2, "b": 3},
        },
        indent=2,
    )

    prompt_string = f"""
You are a tool-calling assistant.

Select exactly one function from the registry below that best matches the
user's request.
Return only valid JSON with this exact shape:
{expect_shape}

Return ONLY the JSON object between these markers (nothing else):
<<JSON_START>>
{expect_shape}
<<JSON_END>>

Rules:
- Use only function names from the registry.
- Use only parameter names exactly as defined.
- Do not include explanations or markdown.
- If a value is missing, use null.
- Return valid JSON only.
- Return ONLY the JSON object between the markers.
- Do NOT repeat the prompt, the registry, or the example.
- Do NOT include the markers in the JSON output (only the delimiter).

Registry:
{registry}

User request:
{user_input}
""".strip()

    return prompt_string


def parse_output(llm_output: str) -> dict[str, Any] | None:
    """
    - Try to directly translate through json.loads
    - Else look for <<JSON>> Markers
    - Else look for last bracket section
    - Else try fallbacks
    Lastly throw error if all fails
    """

    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    matches = list(re.finditer(r'<<JSON_START>>(.*?)<<JSON_END>>', llm_output,
                               re.S))
    attempt = matches[-1].group(1).strip() if matches else None

    if not attempt:
        def balanced_regions(output: str):
            regions = []

            for i, chr in enumerate(output):
                if chr != "{":
                    continue
                depth = 0
                for j in range(i, len(output)):
                    if output[j] == "{":
                        depth += 1
                    elif output[j] == "}":
                        depth -= 1
                        if depth == 0:
                            regions.append(output[i: j + 1])
                            break
                return regions

        regions = balanced_regions(llm_output)
        if regions:
            attempt = regions[-1].strip()

    if not attempt:
        raise ValueError(f"Could not find JSON object in LLM output: {
                         llm_output!r}")

    try:
        return json.loads(attempt)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(attempt)
        except Exception:
            try:
                return json.loads(attempt.replace("'", '"'))
            except Exception as alert:
                raise ValueError(f"Failed to parse: {alert}\n{attempt!r}")


def args_from_llm_output(output: str | dict[str, Any],
                         functions: dict[str, Any]) -> dict[str, Any]:
    if isinstance(output, str):
        try:
            obj = json.loads(output)
        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {output!r}")
    else:
        obj = output

    if not isinstance(obj, dict):
        raise ValueError(f"Parsed LLM output is not an object: {obj!r}")

    if "arguments" not in obj:
        raise ValueError(f"LLM response missing 'arguments': {obj}")

    args = obj["arguments"]
    if not isinstance(args, dict):
        raise ValueError(f"'arguments' must be an object: {args!r}")

    params = functions.get("parameters", {})

    for key in params:
        if key not in args:
            raise ValueError(f"Missing argument '{key}'- {functions['name']}")

    return args


def handle_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "fn_add_numbers":
        return {"a": float(args["a"]), "b": float(args["b"])}
    elif name == "fn_greet":
        return {"name": str(args["name"])}
    elif name == "fn_reverse_string":
        return {"s": str(args["s"])}
    elif name == "fn_get_square_root":
        return {"a": float(args["a"])}
    elif name == "fn_substitute_string_with_regex":
        return {
            "source_string": str(args["source_string"]),
            "regex": str(args["regex"]),
            "replacement": str(args["replacement"]),
        }
    else:
        print(f"Unsupported function: {name}")
        quit()


def test_function(func: str, args: dict[str, Any]) -> Any:
    function = funcs.FUNC_DISPATCH[func]
    return function(**args)


def process_prompts(prompt: str, registry: list[dict[str, Any]],
                    model: Small_LLM_Model) -> dict[str, Any]:
    # produce registry and run it against LLM to get output
    func_prompt = func_select_prompt(prompt, registry)
    llm_output = generate(func_prompt, 128, model)

    # parse output/handle errors
    try:
        result = parse_output(llm_output)
    except Exception as alert:
        print(f"Failed to parse LLM output: {alert!r}")
        print("Raw LLM output (truncated):", repr(llm_output)[:1000])
        quit()

    # verify parsed object structure
    if not isinstance(result, dict):
        print(f"Parse result is not an object: {result}")
        quit()

    # extract function name
    name = result.get("function_name")
    if not isinstance(name, str):
        print(f"Parsed result missing valid 'function_name': {result}")
        quit()
    if name not in funcs.FUNC_DISPATCH:
        print(f"LLM selected function not listed: {name}")
        quit()

    # extract definition
    definition = next((func for func in registry if func.get("name") == name),
                      None)
    if definition is None:
        print(f"No function definition in LLM selection: {name}")
        quit()

    # extract arguments
    try:
        arguments = args_from_llm_output(result, definition)
    except Exception as alert:
        print(f"Arguments couldn't be validated: {alert}")
        quit()

    # normalize arguments for function testing
    try:
        validated_args = handle_args(name, arguments)
    except Exception as alert:
        print(f"Arguments couldn't be normalized: {alert}")
        quit()

    # Test function
    try:
        final = test_function(name, validated_args)
    except Exception as alert:
        print(f"Function couldn't be executed: {alert}")
        quit()

    return {
        "function_name": name,
        "arguments": validated_args,
        "result": final,
    }


def main() -> None:
    if len(sys.argv) != 3:
        print("Run: python3 -m src.main <path_to_function_file>",
              "<path_to_test_file")
        sys.exit(1)

    func_path = sys.argv[1]
    test_path = sys.argv[2]
    registry = json_to_obj(func_path)
    tests = json_to_obj(test_path)
    model = Small_LLM_Model()
    all_results = []

    for test in tests:
        prompt = test["prompt"]
        result = process_prompts(prompt, registry, model)

        assert "function_name" in result
        assert "arguments" in result
        assert "result" in result

        func_def = next(
            func for func in registry
            if func["name"] == result["function_name"]
        )

        assert set(result["arguments"].keys()) == set(
            func_def["parameters"].keys())

        print(prompt)
        print(result)
        all_results.append(result)
    obj_to_json({"results": all_results})
    print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
