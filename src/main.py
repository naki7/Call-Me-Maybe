import sys
import json

from llm_sdk.llm_sdk import Small_LLM_Model
from src.in_out_handler import json_to_obj
from src.json_constrainer import constrained_decoder, load_vocab


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

    for test in tests:
        prompt = json.dumps(test["prompt"])
        if not registry:
            return None

        tester = constrained_decoder(model, prompt, registry, vocab)
        print(model.decode(tester).rstrip())

        # context = build_model_context(prompt, registry)
        # print(context)

        # input_ids = encode_text(model, context)
        # print(model.decode(input_ids) == context)

        # print(model.decode(tester))
        # load_vocab(model)
    #     return

    #     result = process_prompts(prompt, registry, model)
    #     return

    #     assert "prompt" in result
    #     assert "name" in result
    #     assert "parameters" in result

    #     func_def = next(
    #         func for func in registry
    #         if func["name"] == result["name"]
    #     )

    #     assert set(result["parameters"].keys()) == set(
    #         func_def["parameters"].keys())

    #     print(prompt)
    #     print(result)
    #     all_results.append(result)

    # obj_to_json({"results": all_results})
    # print(json.dumps(all_results, indent=2))


if __name__ == '__main__':
    main()
