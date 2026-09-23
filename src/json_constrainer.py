import json
from enum import Enum, auto

from llm_sdk.llm_sdk import Small_LLM_Model


class JSON_State(Enum):
    EXPECT_OPEN_OBJ = auto()

    EXPECT_NAME_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_FUNCTION_NAME = auto()

    EXPECT_PARAMETERS_KEY = auto()
    EXPECT_PARAMETER_KEY = auto()
    EXPECT_PARAMETER_COLON = auto()

    EXPECT_STRING_OPEN = auto()
    IN_STRING = auto()

    EXPECT_NUMBER = auto()
    IN_DIGIT = auto()
    IN_DECIMAL = auto()

    EXPECT_BOOLEAN = auto()

    EXPECT_COMMA_OR_OBJ_PARAMETERS = auto()
    EXPECT_OBJ_END = auto()

    DONE = auto()


class JSON_Machine:
    def __init__(self, model: Small_LLM_Model, registry: list[dict]):
        self.model = model
        self.registry = registry

    def encode(self, text: str) -> list[int]:
        return encode_text(self.model, text)

    def valid_first_tokens(self, texts: list[str]) -> set[int]:
        valid = set()

        for text in texts:
            ids = self.encode(text)
            if ids:
                valid.add(ids[0])

        return valid

    def verify_sequence(self, gen_ids: list[int], expected: list[int]) -> bool:
        if len(gen_ids) < len(expected):
            return False
        return gen_ids[-len(expected):] == expected

    def valid_tokens(self, state: JSON_State, gen_ids: list[int],
                     curr_func: dict | None) -> set[int]:

        if state == JSON_State.EXPECT_OPEN_OBJ:
            return self.valid_first_tokens(["{"])

        elif state == JSON_State.EXPECT_NAME_KEY:
            return self.valid_first_tokens(['"name"'])

        elif state == JSON_State.EXPECT_COLON:
            return self.valid_first_tokens([":"])

        elif state == JSON_State.EXPECT_FUNCTION_NAME:
            func_names = [func["name"] for func in self.registry]
            return self.valid_first_tokens(
                [f'"{func_name}"' for func_name in func_names])

        elif state == JSON_State.EXPECT_PARAMETERS_KEY:
            return self.valid_first_tokens(['"parameters"'])

        elif state == JSON_State.EXPECT_PARAMETER_KEY:
            if curr_func is None:
                return set()

            param_names = curr_func["parameters"].keys()
            return self.valid_first_tokens(
                [f'"{param}"' for param in param_names])

        elif state == JSON_State.EXPECT_PARAMETER_COLON:
            return self.valid_first_tokens([":"])

        elif state == JSON_State.EXPECT_STRING_OPEN:
            return set(self.encode('"'))

        elif state == JSON_State.EXPECT_MORE_STRING:
            pass
            return self.valid_first_tokens(['"'])

        elif state == JSON_State.EXPECT_NUMBER:
            pass

        elif state == JSON_State.EXPECT_BOOLEAN:
            return self.valid_first_tokens(["true", "false"])

        elif state == JSON_State.EXPECT_COMMA:
            return self.valid_first_tokens([","])

        elif state == JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS:
            return self.valid_first_tokens([",", "}"])

        elif state == JSON_State.DONE:
            return set()

        return set()

    def update_state(self, state: JSON_State, next_token: int,
                     gen_ids: list[int]) -> JSON_State:

        if state == JSON_State.EXPECT_OPEN_OBJ:
            return JSON_State.EXPECT_NAME_KEY

        elif state == JSON_State.EXPECT_NAME_KEY:
            expect_ids = self.encode('"name"')
            if self.verify_sequence(gen_ids, expect_ids):
                return JSON_State.EXPECT_COLON

        elif state == JSON_State.EXPECT_COLON:
            return JSON_State.EXPECT_FUNCTION_NAME

        elif state == JSON_State.EXPECT_FUNCTION_NAME:
            return JSON_State.EXPECT_PARAMETERS_KEY

        elif state == JSON_State.EXPECT_PARAMETERS_KEY:
            return JSON_State.EXPECT_PARAMETER_KEY

        elif state == JSON_State.EXPECT_PARAMETER_KEY:
            return JSON_State.EXPECT_PARAMETER_COLON

        elif state == JSON_State.EXPECT_PARAMETER_COLON:
            return JSON_State.EXPECT_BOOLEAN

        elif state == JSON_State.EXPECT_STRING_OPEN:
            pass

        elif state == JSON_State.EXPECT_MORE_STRING:
            pass

        elif state == JSON_State.EXPECT_NUMBER:
            pass

        elif state == JSON_State.EXPECT_BOOLEAN:
            return JSON_State.EXPECT_COMMA

        elif state == JSON_State.EXPECT_COMMA:
            return JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS

        elif state == JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS:
            return JSON_State.DONE

        return state


def constrained_decoder(model: Small_LLM_Model, registry: list[dict]) -> list[int]:
    state = JSON_State.EXPECT_OPEN_OBJ
    state_machine = JSON_Machine(model, registry)
    curr_func = None
    gen_ids = encode_text(model,
                          '{"name":"fn_add_numbers","parameters":{"a":{"type":"number"},"b":{"type":"number"}}')
    print(gen_ids)
    while state != JSON_State.DONE:
        logits = model.get_logits_from_input_ids(gen_ids)
        if hasattr(logits, "__len__") and logits and hasattr(logits[0], "__len__"):
            logits = logits[0]

        valid_ids = state_machine.valid_tokens(state, gen_ids, curr_func)
        # HANDLE GRACEFULLY LATER
        if not valid_ids:
            print(f"No valid tokens for state {state}")
        else:
            for id in range(len(logits)):
                if id not in valid_ids:
                    logits[id] = float("-inf")

            next_token = max(valid_ids, key=lambda id: logits[id])

            gen_ids.append(next_token)
        print(model.decode(gen_ids))

        state = state_machine.update_state(state, next_token, gen_ids)

    return gen_ids

def encode_text(model: Small_LLM_Model, text: str) -> list[list[int]]:
    ids = model.encode(text)

    try:
        ids = ids.tolist()
    except AttributeError:
        ids = list(ids)

    if ids and isinstance(ids[0], list):
        ids = ids[0]

    return [int(token_id) for token_id in ids]


def script_exp(model: Small_LLM_Model, vocab: dict[str, int],
               examples: list[str]) -> None:
    print(f"Vocab size: {len(vocab)}")

    for text in examples:
        token_ids = encode_text(model, text)

        print("\n" + "=" * 50)
        print(f"TEXT: \"{text}\"")
        print(f"TOKENS: {token_ids}")

        for token_id in token_ids:
            print(f"  {token_id} -> {model.decode(token_id)}")


def load_vocab(model: Small_LLM_Model) -> None:
    vocab_path = model.get_path_to_vocab_file()
    vocab = {}
    print(vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        vocab = json.load(vocab_file)

    examples = [
        # "{",
        # "}",
        # ":",
        # ",",
        # "\"",
        # "\"name\"",
        # "\"name\":",
        # "fn_add_numbers",
        # "\"fn_add_numbers\"",
        # "{\"name\":\"fn_add_numbers\"}",
        # "{\"name\": \"fn_add_numbers\"}",
        # " fn_add_numbers",
        # "\nfn_add_numbers",
        # "40",
        # "40.5",
        # "-40",
        # "true",
        # "false",
        "hello",
        "hello world",
        " hello",
        "fn_",
        "fn_add",
        "\"hello",
        "\"hello world",
        "{",
        '{"',
        '{"name',
        '{"name"',
        '{"name":',
        '{"name":"',
        '{"name":"fn_add_numbers',
        '{"name":"fn_add_numbers"',
        '{"name":"fn_add_numbers",',
        '{"name":"fn_add_numbers","parameters":',
        '{"name":"fn_add_numbers","parameters":{"a":{"type":"number"},"b":{"type":"number"}}',
        '"hello"',
        '"hello world"',
        "40",
        "40.5",
        "-40",
        "-40.5",
        "true",
        "false",
        "null",
        '"',
        '"n',
        '"na',
        '"nam',
        '"name',
        '"name"',
        '"fn_',
        '"fn_a',
        '"fn_add',
        '"fn_add_',
        '"fn_add_numbers',
    ]
    script_exp(model, vocab, examples)
