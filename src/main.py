from src.parser import parse_args
from llm_sdk.llm_sdk import Small_LLM_Model
from src.in_out_handler import json_to_obj, obj_to_json
from src.json_constrainer import constrained_decoder, load_vocab
from src.validator import (load_registry, validate_gened_json,
                           validate_func_call)


def main() -> None:
    arguments = parse_args()
    func_path = arguments.functions_definition
    prompt_path = arguments.input
    out_path = arguments.output

    try:
        registry = load_registry(func_path)
        if not registry:
            raise ValueError("Registry is empty.")

        prompts = json_to_obj(prompt_path)
        if not isinstance(prompts, list):
            raise ValueError("The input file must contain a JSON array.")

        model = Small_LLM_Model()
        vocab = load_vocab(model)
        generated_results = []

        for index, prompt in enumerate(prompts):
            if not isinstance(prompt, dict):
                raise ValueError(
                    f"Prompt at index {index} must be a JSON object.")

            curr_prompt = prompt.get("prompt")
            if not isinstance(curr_prompt, str):
                raise ValueError(
                    f"Prompt at index {index} must contain a string prompt.")

            gen_tokens = constrained_decoder(model, curr_prompt, registry,
                                             vocab)
            generated_text = model.decode(gen_tokens)

            call = validate_gened_json(generated_text)
            validated_call = validate_func_call(call, registry)

            generated_results.append(validated_call)
            print(validated_call)

        obj_to_json(generated_results, out_path)
    except (OSError, ValueError) as alert:
        print(f"Error: {alert}")


if __name__ == '__main__':
    main()
