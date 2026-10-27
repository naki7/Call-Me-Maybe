import sys

from llm_sdk.llm_sdk import Small_LLM_Model
from src.in_out_handler import json_to_obj, obj_to_json
from src.json_constrainer import constrained_decoder, load_vocab
from src.validator import (load_registry, validate_gened_json,
                           validate_func_call)
from src.dispatcher import dispatch_func


def main() -> None:
    if len(sys.argv) != 3:
        print("Using defaults for: uv run python -m src",
              "[--functions_definition <function_definition_file>]",
              "[--input <input_file>",
              "[--output <output_file>]")
        func_path = ".\\data\\input\\functions_definition.json"
        test_path = ".\\data\\input\\function_calling_tests.json"
        out_path = ".\\data\\output\\function_calls.json"
    else:
        func_path = sys.argv[1]
        test_path = sys.argv[2]
        out_path = sys.argv[3]
    try:
        registry = load_registry(func_path)
    except Exception as alert:
        print(alert)
        return None
    tests = json_to_obj(test_path)
    model = Small_LLM_Model()
    vocab = load_vocab(model)
    generated_results = []

    for test in tests:
        prompt = test["prompt"]
        if not registry:
            return None

        # add arrays and nested objects
        generated = constrained_decoder(model, prompt, registry, vocab)
        generated_text = model.decode(generated)
        call = validate_gened_json(generated_text)
        validated_call = validate_func_call(call, registry)

        generated_results.append(validated_call)
        print(call)
        # result = dispatch_func(call)
        # print(result)
    obj_to_json(generated_results, out_path)


if __name__ == '__main__':
    main()
