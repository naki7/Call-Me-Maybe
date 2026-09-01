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
        print("Raw LLM output:", repr(llm_output))

        def extract_first_json(text: str):
            # find first balanced {...} and try to parse it
            for i, ch in enumerate(text):
                if ch != "{":
                    continue
                depth = 0
                for j in range(i, len(text)):
                    if text[j] == "{":
                        depth += 1
                    elif text[j] == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = text[i:j+1]
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                # try python literal fallback
                                try:
                                    return ast.literal_eval(candidate)
                                except Exception:
                                    # continue scanning for next
                                    # balanced region
                                    break
            return None

        # marker-based: take the last marker-delimited block if any
        matches = list(re.finditer(r'<<JSON_START>>(.*?)<<JSON_END>>',
                                   llm_output, re.S))
        candidate = None
        if matches:
            candidate = matches[-1].group(1).strip()

        if candidate:
            try:
                result = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    result = ast.literal_eval(candidate)
                except Exception:
                    try:
                        result = json.loads(candidate.replace("'", '"'))
                    except Exception as e:
                        print("Failed to parse candidate from markers:", e)
                        quit()
        else:
            # fallback: balanced-brace extractor
            # (your existing extract_first_json)
            result = extract_first_json(llm_output)
            if result is None:
                # last-resort naive region
                start = llm_output.find("{")
                end = llm_output.rfind("}")
                if start == -1 or end == -1 or end < start:
                    print(f"Could not find JSON object in LLM output: {
                        llm_output!r}")
                    quit()
                candidate = llm_output[start:end + 1]
                try:
                    result = json.loads(candidate)
                except Exception:
                    try:
                        result = ast.literal_eval(candidate)
                    except Exception as e:
                        print("Failed to parse candidate:", e)
                        quit()

        result = extract_first_json(llm_output)
        if result is None:
            # last resort: try naive single-quote -> double-quote replacement
            # on the first { .. } region found by simple find/rfind
            start = llm_output.find("{")
            end = llm_output.rfind("}")
            if start == -1 or end == -1 or end < start:
                print(f"Could not find JSON object in LLM output: {
                    llm_output!r}")
                quit()
            candidate = llm_output[start:end + 1]
            try:
                result = json.loads(candidate.replace("'", '"'))
            except Exception as alert:
                print("Failed to parse candidate as JSON/Python literal",
                      f"Error: {alert},\nFrom: {candidate!r}")
                quit()

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
