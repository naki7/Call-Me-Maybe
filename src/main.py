from src.parser import parse_args
from llm_sdk.llm_sdk import Small_LLM_Model
from src.in_out_handler import json_to_obj, obj_to_json
from src.json_constrainer import constrained_decoder, load_vocab
from src.validator import (load_registry, validate_gened_json,
                           validate_func_call)
from src.dispatcher import dispatch_func


def main() -> None:
    arguments = parse_args()
    func_path = arguments.functions_definition
    test_path = arguments.input
    out_path = arguments.output
    registry = load_registry(func_path)
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
