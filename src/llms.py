from llm_sdk import Small_LLM_Model
import torch


def generate(prompt: str, tokens: int, model: Small_LLM_Model) -> str:
    inputs = model.encode(prompt)
    with torch.no_grad():
        outputs = model._model.generate(
            inputs,
            max_new_tokens=tokens,
            do_sample=False,
            pad_token_id=model._tokenizer.eos_token_id,
        )
        return model.decode(outputs[0])
