import sys
import json

from llm_sdk.llm_sdk import Small_LLM_Model
from src.in_out_handler import json_to_obj
from src.json_constrainer import constrained_decoder, load_vocab
from src.json_constrainer import encode_text
from src.constrained import get_next_logits, decode_text


def test_number_extraction(model, prompt: str, steps: int = 10):
    input_ids = encode_text(model, prompt)
    generated_ids = []

    for _ in range(steps):
        print("Input token count:", len(input_ids))
        print("Input token IDs (last 40):", input_ids[-40:])
        print("Input text (last 250 chars):", repr(decode_text(model, input_ids)[-250:]))
        logits = get_next_logits(model, input_ids)

        # Show the model's preferred next tokens without JSON masking.
        top_ids = sorted(
            range(len(logits)),
            key=lambda token_id: float(logits[token_id]),
            reverse=True
        )[:10]

        print([
            (model.decode([token_id]), float(logits[token_id]))
            for token_id in top_ids
        ])

        next_id = top_ids[0]
        generated_ids.append(next_id)
        input_ids.append(next_id)

    print("Generated continuation:", repr(model.decode(generated_ids)))


def main() -> None:
    # model = Small_LLM_Model()
    # print("=======\nTESTING - First number:\n=======\n\n\n")
    # test_number_extraction(
    #     model,
    #     """User request: What is the sum of -2.53 and -0.12?
    #         Function: fn_add_numbers
    #         Extract the numbers from the user request in their original order.
    #         Preserve each number's sign and decimal digits exactly.
    #         First number:"""
    # )
    # print("\n\n\n=======\nTESTING - Return the arguments as JSON:{\"a\":\n=======\n\n\n")
    # test_number_extraction(
    #     model,
    #     """User request: What is the sum of -2.53 and -0.12?
    #         Function: fn_add_numbers
    #         Extract the numbers from the user request in their original order.
    #         Preserve each number's sign and decimal digits exactly.
    #         Return the arguments as JSON:
    #         {"a":"""
    # )
    # print("\n\n\n=======\nTESTING - Copy the first numeric value from the user request exactly,including its sign and decimal digits. Do not calculate.{\"a\":\n=======\n\n\n")
    # test_number_extraction(
    #         model,
    #         """Available functions:
    # Function: fn_add_numbers
    # Description: Add two numbers together and return their sum.
    # Parameters:
    # - a: number
    # - b: number

    # Function: fn_greet
    # Description: Generate a greeting message for a person by name.
    # Parameters:
    # - name: string

    # Function: fn_reverse_string
    # Description: Reverse a string and return the reversed result.
    # Parameters:
    # - s: string

    # Function: fn_get_square_root
    # Description: Calculate the square root of a number.
    # Parameters:
    # - a: number

    # Function: fn_substitute_string_with_regex
    # Description: Replace all occurrences matching a regex pattern in a string.
    # Parameters:
    # - source_string: string
    # - regex: string
    # - replacement: string

    # User request: What is the sum of -2.53 and -0.12?


    # {"prompt":"What is the sum of -2.53 and -0.12?","name":"fn_add_numbers","parameters":{"a":"""
    #     )
    # from src.json_constrainer import build_model_context
    # print("=======\nTESTING - using build_model_context:\n=======\n\n\n")
    # func_path = sys.argv[1]
    # test_path = sys.argv[2]
    # registry = json_to_obj(func_path)
    # tests = json_to_obj(test_path)
    # for test in tests:
    #     prompt = test["prompt"]
    #     context = build_model_context(prompt, registry)
    #     escaped_prompt = json.dumps(prompt)
    #     output_prefix = '{"prompt":' + escaped_prompt + ','
    #     output_prefix = (
    #         '{"prompt":'
    #         + json.dumps(prompt)
    #         + ',"name":"fn_add_numbers","parameters":{"a":'
    #     )

    #     test_number_extraction(
    #         model,
    #         context + "\n\n" + output_prefix
    #     )
    # return
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
        prompt = test["prompt"]
        if not registry:
            return None

        tester = constrained_decoder(model, prompt, registry, vocab)
        print(model.decode(tester).rstrip())
        result = json.loads(model.decode(tester))
        print(repr(result))

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
