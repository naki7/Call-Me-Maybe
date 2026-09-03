from typing import List, Dict, Optional, Any
from llm_sdk import Small_LLM_Model
import math
import re

# very small cache to avoid repeated encodes for same prompt/function
_prefix_ids_cache: Dict[str, List[int]] = {}
_first_token_cache: Dict[str, Optional[int]] = {}


def to_id_list(model: Small_LLM_Model, prompt: str) -> List[Any]:
    """
    Encode prompt to a Python list - token ids from tokenizer
    """
    encoded = model.encode(prompt)

    # Handle either a Tensor or List type return from encode
    try:
        return list(encoded.tolist())
    except Exception:
        try:
            return list(encoded)
        except Exception:
            return [int(encoded)]


def logits_for_next(model: Small_LLM_Model,
                    input_ids: List[int]) -> List[float]:
    """
    Prepares logits for the next token call
    """
    logits = model.get_logits_from_input_ids(list(input_ids))

    if logits and hasattr(logits, "__len__") and hasattr(logits[0], "__len__"):
        logits = logits[0]
    if not isinstance(logits, list):
        return [logits]
    else:
        return list(logits)


def build_name_first_token_map(functions: List[Dict[str, Any]],
                               model: Small_LLM_Model,) -> Dict[
                                   str, Optional[int]]:
    """
    Precompute the first-token id for JSON-quoted function names.
    Returns {name: first_token_id_or_None}
    """
    map: Dict[str, Optional[int]] = {}
    for func in functions:
        name = func.get("name")
        if not isinstance(name, str):
            map[name] = None
            continue
        try:
            seq = '"' + name + '"'
            ids = to_id_list(model, seq)
            map[name] = ids[0] if ids else None
        except Exception:
            map[name] = None
    # seed global cache for quicker reuse
    _first_token_cache.update(map)
    return map


def _jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    uni = sa | sb
    return len(inter) / len(uni) if uni else 0.0


def _is_greeting_intent(prompt: str) -> bool:
    return bool(re.search(r"\b(greet|say hello|say hi|hello|hi|hey|greeting|good morning|good evening)\b", prompt, re.I))


def _is_substitute_intent(prompt: str) -> bool:
    return bool(re.search(r"\b(substitute|replace|replace all|swap|change|replace the word|replace the substring)\b", prompt, re.I))


def _mentions_numbers(prompt: str) -> bool:
    # checks for explicit number-replacement intent or numeric tokens present
    if re.search(r"\b(number|numbers|digits|\\d|NUMBERS)\b", prompt, re.I):
        return True
    if re.search(r"\d", prompt):
        # presence of digits in the user example string
        return True
    return False


def select_function(prompt: str, functions: List[Dict[str, Any]], model,
                    name_first_token_map: Optional[
                        Dict[str, Optional[int]]] = None,
                    *, alpha: float = 1.0, beta: float = 6.0,
                    gamma: float = 3.0) -> Optional[str]:
    """
    Hybrid deterministic selector:
      - prefix logit score of the first token of the JSON-quoted function name
      - jaccard similarity between prompt and function description
      - small rule-based bonuses for greeting-like prompts

    Returns best function name or None if model logits API not available.
    """
    # cache prefix ids per prompt
    if prompt in _prefix_ids_cache:
        prefix_ids = _prefix_ids_cache[prompt]
    else:
        try:
            prefix_ids = to_id_list(model, prompt)
        except Exception:
            return None
        _prefix_ids_cache[prompt] = prefix_ids

    # compute logits once for the current prefix
    try:
        logits = logits_for_next(model, prefix_ids)
    except Exception:
        return None

    # ensure we have first-token ids for names
    if name_first_token_map is None:
        name_first_token_map = _first_token_cache

    best_name = None
    best_score = -math.inf
    greeting = _is_greeting_intent(prompt)
    sub_intent = _is_substitute_intent(prompt)
    numbers_mentioned = _mentions_numbers(prompt)

    for f in functions:
        name = f.get("name")
        if not isinstance(name, str):
            continue

        first_id = None
        if name_first_token_map and name in name_first_token_map:
            first_id = name_first_token_map[name]
        elif name in _first_token_cache:
            first_id = _first_token_cache[name]
        else:
            try:
                seq = '"' + name + '"'
                ids = to_id_list(model, seq)
                first_id = ids[0] if ids else None
            except Exception:
                first_id = None
            _first_token_cache[name] = first_id

        if first_id is None or first_id >= len(logits):
            token_logit = -1e9
        else:
            token_logit = float(logits[first_id])

        # normalized token logit (scale to ~[-1,1] roughly)
        # use tanh to squish large logits
        token_score = math.tanh(token_logit / 20.0)

        # cheap semantic score between prompt and function description/name
        desc = " ".join(filter(None, [f.get("description", ""), name]))
        desc_sim = _jaccard(prompt, desc)

        # rule-based bonus/penalty
        bonus = 0.0
        lname = name.lower()
        if greeting and "greet" in lname:
            bonus += 1.5
        if sub_intent and ("substitut" in lname or "regex" in lname or "substitute" in lname or "replace" in lname):
            bonus += 2.0
        if sub_intent and "add" in lname:
            bonus -= 2.0
        if numbers_mentioned and sub_intent:
            # likely a numbers-replacement intent -> favor substitute
            if "substitut" in lname or "regex" in lname:
                bonus += 1.5
        # small heuristic: if prompt contains 'square root' favor sqrt
        if re.search(r"\b(square root|sqrt|root of)\b", prompt, re.I) and "square" in lname:
            bonus += 2.0

        # combine with weights (alpha, beta, gamma)
        # beta is higher to prioritize description similarity for ambiguous
        # token logits
        score = alpha * token_score + beta * desc_sim + gamma * bonus

        if score > best_score:
            best_score = score
            best_name = name

    return best_name
