import json
import sys
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

    prompt_string = f"""
You are a tool-calling assistant.

Select exactly one function from the registry below that best matches the
user's request.
Return only valid JSON with this exact shape:
{{
    "function_name": "name_of_chosen_function",
    "arguments": {{ ... parameter names and values ... }}
}}

Rules:
- Use only function names from the registry.
- Use only parameter names exactly as defined.
- Do not include explanations or markdown.
- If a value is missing, use null.
- Return valid JSON only.

Registry:
{registry}

User request:
{user_input}
""".strip()

    return prompt_string


def args_from_llm_output(output: str,
                         functions: dict[str, Any]) -> dict[str, Any]:
    try:
        obj = json.loads(output)
    except json.JSONDecodeError:
        print(f"LLM returned invalid JSON: {output}")
        quit()

    if "arguments" not in obj:
        print(f"LLM response missing 'arguments': {output}")
        quit()

    args = obj["arguments"]
    params = functions.get("parameters", {})

    for key in params:
        if key not in args:
            print(f"Missing argument '{key}' for {functions['name']}")
            quit()

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
    func_prompt = func_select_prompt(prompt, registry)
    llm_output = generate(func_prompt, 128, model)

    try:
        result = json.loads(llm_output)
    except json.JSONDecodeError:
        # pull JSON from response
        start = llm_output.find("{")
        end = llm_output.find("}")

        if start == -1 or end == -1 or end < start:
            print(f"Could not parse LLM output to JSON: {llm_output}")
        result = json.loads(llm_output[start:end + 1])

    name = result.get("function_name")
    if name not in funcs.FUNC_DISPATCH:
        print(f"LLM selected function not recognized: {name}")
        quit()

    definition = next(func for func in registry if func.get("name") == name)

    arguments = args_from_llm_output(json.dumps(result), definition,)

    validated_args = handle_args(name, arguments)
    final = test_function(name, validated_args)

    return {
        "function_name": name,
        "arguments": validated_args,
        "result": final,
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Run: python3 -m src.main <path_to_json_file>")
        sys.exit(1)

    path = sys.argv[1]
    registry = json_to_obj(path)
    model = Small_LLM_Model()

    # EXAMPLE
    sample_prompt = "What is the sum of 2 and 3?"
    result = process_prompts(sample_prompt, registry, model)
    obj_to_json(result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
