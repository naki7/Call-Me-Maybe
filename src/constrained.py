import json
import re
from typing import Any, Dict, List, Optional, Tuple

from llm_sdk.llm_sdk import Small_LLM_Model


def encode_text(model: Small_LLM_Model, text: str) -> List[int]:
    ids = model.encode(text)
    try:
        return list(ids.tolist())
    except Exception:
        try:
            return list(ids)
        except Exception:
            return [int(ids)]


def decode_text(model: Small_LLM_Model, ids: List[int]) -> str:
    try:
        return model.decode(ids)
    except Exception:
        return ""


def get_next_logits(model: Small_LLM_Model, input_ids: List[int]) -> List[float]:
    logits = model.get_logits_from_input_ids(input_ids)
    if hasattr(logits, "__len__") and logits and hasattr(logits[0], "__len__"):
        logits = logits[0]
    return [float(x) for x in logits]


def json_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


def literal_value_score(model: Small_LLM_Model, prefix: str, candidate: str) -> float:
    """
    Minimal probability-based constrained score:
    encode the prefix followed by candidate, then score the final token id that
    would continue the candidate text.
    """
    enc = encode_text(model, prefix + candidate)
    if len(enc) < 2:
        return -1e9
    logits = get_next_logits(model, enc[:-1])
    final_id = enc[-1]
    if final_id >= len(logits):
        return -1e9
    return float(logits[final_id])


def allowed_function_names(functions: List[Dict[str, Any]]) -> List[str]:
    return [f["name"] for f in functions if isinstance(f, dict) and "name" in f]


class JSONState:
    """
    Tiny schema-aware state machine for the target JSON object shape:

    {
      "function_name": "<name>",
      "arguments": {
        "<param>": <value>,
        ...
      }
    }

    This is enough to constrain the JSON format while still keeping the implementation
    compact and practical.
    """

    def __init__(self, function_name: str, schema: Dict[str, Any]):
        self.function_name = function_name
        self.schema = schema
        self.params = schema.get("parameters", {})
        self.param_names = list(self.params.keys())
        self.idx = 0
        self.in_args = False
        self.in_args_key = False
        self.expecting = "start"
        self.done = False

    def next_state_after_value(self) -> None:
        if self.idx < len(self.param_names) - 1:
            self.expecting = "comma_then_next_key"
        else:
            self.expecting = "close_args"
        self.in_args_key = False

    def allowed_tokens_for_key(self) -> List[str]:
        if self.expecting == "start":
            return ["{"]

        if self.expecting == "function_name_key":
            return ['"function_name"']

        if self.expecting == "function_name_value":
            return [json_quote(self.function_name)]

        if self.expecting == "after_function_name":
            return [","]

        if self.expecting == "arguments_key":
            return ['"arguments"']

        if self.expecting == "arguments_start":
            return ["{"]

        if self.expecting == "key_name":
            if self.idx < len(self.param_names):
                return [json_quote(self.param_names[self.idx])]
            return []

        if self.expecting == "colon":
            return [":"]

        if self.expecting == "value":
            param_name = self.param_names[self.idx]
            ptype = self.params[param_name].get("type", "string")
            if ptype == "number":
                return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "."]
            if ptype == "string":
                return ['"']
            if ptype == "boolean":
                return ["true", "false"]
            return ['"']

        if self.expecting == "close_value":
            return [",", "}"]

        if self.expecting == "comma_then_next_key":
            return [","]

        if self.expecting == "close_args":
            return ["}"]

        return []

    def next_token_candidates(self, current_json: str) -> List[str]:
        # Very compact state simulation based on current partial JSON text.
        if not current_json:
            return ['{']

        if current_json == "{":
            return ['"function_name"']

        if current_json.endswith('"function_name"'):
            return [":"]
        if current_json.endswith(':"'):
            return [json_quote(self.function_name)]

        if current_json.endswith(json_quote(self.function_name)):
            return [","]

        if current_json.endswith(','):
            return ['"arguments"']

        if current_json.endswith('"arguments"'):
            return [":"]
        if current_json.endswith(':'):
            return ["{"]

        if current_json.endswith("{"):
            return [json_quote(self.param_names[0])] if self.param_names else ["}"]

        if self.param_names and current_json.endswith(json_quote(self.param_names[self.idx])):
            return [":"]
        if self.param_names and current_json.endswith(":"):
            param_name = self.param_names[self.idx]
            ptype = self.params[param_name].get("type", "string")
            if ptype == "number":
                return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "."]
            if ptype == "string":
                return ['"']
            if ptype == "boolean":
                return ["true", "false"]
            return ['"']

        if current_json.endswith('"'):
            # We are inside a string parameter value; allow a closing quote followed by separator
            return ['"', ",", "}"]

        if current_json.endswith(","):
            if self.idx < len(self.param_names) - 1:
                return [json_quote(self.param_names[self.idx + 1])]
            return ["}"]

        if current_json.endswith("}"):
            return []

        return []


def constrained_generate_function_call(prompt: str, registry: List[Dict[str, Any]], model: Small_LLM_Model) -> Optional[Dict[str, Any]]:
    """
    Full minimal constrained JSON-state decoder.
    It does not implement arbitrary general JSON; it specifically enforces the project's
    target object shape:
      {"function_name": "<name>", "arguments": {...}}
    and only allows the names and parameter names in the registry.
    """
    if not registry:
        return None

    best_name = None
    best_score = -1e9

    for func in registry:
        name = func.get("name")
        if not isinstance(name, str):
            continue

        # Score candidate function name using logits on the exact JSON prefix.
        candidate_prefix = prompt + '\n{"function_name":'
        candidate_suffix = json_quote(name) + ',"arguments":'
        ids = encode_text(model, candidate_prefix + candidate_suffix)
        logits = get_next_logits(model, ids[0][:-1])
        last_id = ids[0][-1]
        if last_id >= len(logits):
            continue
        score = float(logits[last_id])
        # print(f"{score} - {name}")
        if score > best_score:
            best_score = score
            best_name = name

    if best_name is None:
        return None

    selected = next((f for f in registry if f.get("name") == best_name), None)
    if selected is None:
        return None

    schema = selected
    params = schema.get("parameters", {})
    if not params:
        return {"function_name": best_name, "arguments": {}}

    args: Dict[str, Any] = {}

    for param_name, param_schema in params.items():
        ptype = param_schema.get("type", "string")

        # minimal extraction from the prompt:
        # - numbers: extract the first numeric token
        # - strings: extract quoted strings from prompt, or the raw value
        # - bools: parse true/false if present
        v: Any = None

        if ptype == "number":
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", prompt)
            if nums:
                v = float(nums[0])
            else:
                v = 0.0

        elif ptype == "string":
            # first, look for quoted content
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
            if quoted:
                v = quoted[0]
            else:
                # fallback: use the clean prompt text or parameter name
                v = prompt.strip()

        elif ptype == "boolean":
            lower = prompt.lower()
            v = "true" in lower

        else:
            v = prompt.strip()

        args[param_name] = v

    return {
        "function_name": best_name,
        "arguments": args,
    }
