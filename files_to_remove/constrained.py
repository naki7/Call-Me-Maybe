# from typing import List

# from llm_sdk.llm_sdk import Small_LLM_Model


# def encode_text(model: Small_LLM_Model, text: str) -> List[int]:
#     ids = model.encode(text)
#     try:
#         return list(ids.tolist())
#     except Exception:
#         try:
#             return list(ids)
#         except Exception:
#             return [int(ids)]


# def decode_text(model: Small_LLM_Model, ids: List[int]) -> str:
#     try:
#         return model.decode(ids)
#     except Exception:
#         return ""


# def get_next_logits(model: Small_LLM_Model, input_ids: List[int]) -> List[float]:
#     logits = model.get_logits_from_input_ids(input_ids)
#     if hasattr(logits, "__len__") and logits and hasattr(logits[0], "__len__"):
#         logits = logits[0]
#     return [float(x) for x in logits]


# def json_quote(value: str) -> str:
#     escaped = value.replace("\\", "\\\\").replace('"', '\\"')
#     return '"' + escaped + '"'
