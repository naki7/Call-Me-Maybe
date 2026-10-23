import sys

from llm_sdk.llm_sdk import Small_LLM_Model
from src.in_out_handler import json_to_obj
from src.json_constrainer import constrained_decoder, load_vocab
from src.validator import validate_gened_json, validate_func_call
from src.dispatcher import dispatch_func


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
    vocab = load_vocab(model)

    generated = """
    {
        "prompt": "Add five and three",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 5,
            "b": 3}
    }
    """

    for test in tests:
        prompt = test["prompt"]
        if not registry:
            return None

        # add arrays and nested objects
        generated = constrained_decoder(model, prompt, registry, vocab)
        generated_text = model.decode(generated)
        call = validate_gened_json(generated_text)
        call = validate_func_call(call, registry)
        print(call)
        result = dispatch_func(call)
        print(result)


if __name__ == '__main__':
    main()
