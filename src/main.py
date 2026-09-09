import sys
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model
from src.in_out_handler import json_to_obj, obj_to_json
from src.constrained import constrained_generate_function_call


def process_prompts(prompt: str, registry: list[dict[str, Any]],
                    model: Small_LLM_Model) -> dict[str, Any]:
    # minimal constrained JSON-state decoder:
    # choose function name with logits restricted to valid names in registry
    constrained = constrained_generate_function_call(prompt, registry, model)
    if constrained is not None:
        name = constrained["function_name"]
        args = constrained["arguments"]
        result = {
            "prompt": prompt,
            "function_name": name,
            "arguments": args,
        }
        return result


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

        assert "prompt" in result
        assert "function_name" in result
        assert "arguments" in result

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
    # print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
