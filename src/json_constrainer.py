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
        self.func_names = [func["name"] for func in self.registry]

        self.curr_func = None
        self.curr_param = None
        self.prev_token = None

        self.expected_sequence = []
        self.sequence_i = 0

        self.func_sequences = []
        self.func_seq_i = 0

        self.param_sequences = []
        self.param_seq_i = 0

    def encode(self, text: str) -> list[int]:
        return encode_text(self.model, text)

    def valid_first_tokens(self, texts: list[str]) -> set[int]:
        valid = set()

        for text in texts:
            ids = self.encode(text)
            if ids:
                valid.add(ids[0])

        return valid

    def init_sequence(self, text: str) -> None:
        self.expected_sequence = self.encode(text)
        self.sequence_i = 0

    def next_token(self) -> set[int]:
        if self.sequence_i >= len(self.expected_sequence):
            return set()
        return {self.expected_sequence[self.sequence_i]}

    def increment_index(self, curr_token: int) -> bool:
        if self.sequence_i >= len(self.expected_sequence):
            return False

        if curr_token != self.expected_sequence[self.sequence_i]:
            return False

        self.sequence_i += 1

        return self.sequence_i == len(self.expected_sequence)

    def verify_sequence(self, gen_ids: list[int], expected: list[int]) -> bool:
        if len(gen_ids) < len(expected):
            return False
        return gen_ids[-len(expected):] == expected

    def init_func_seq(self) -> None:
        self.func_sequences = []

        for func in self.registry:
            func_tokens = self.encode(f'"{func["name"]}"')
            self.func_sequences.append(func_tokens)

        self.func_seq_i = 0

    def next_func_token(self) -> set[int]:
        valid = set()

        for func in self.func_sequences:
            if self.func_seq_i < len(func):
                valid.add(func[self.func_seq_i])

        return valid

    def increment_func(self, curr_token: int) -> bool:
        updated_funcs = []

        for func in self.func_sequences:
            if self.func_seq_i < len(func):
                if func[self.func_seq_i] == curr_token:
                    updated_funcs.append(func)

        self.func_sequences = updated_funcs
        self.func_seq_i += 1

        if not self.func_sequences:
            return False

        return all(self.func_seq_i >= len(func)
                   for func in self.func_sequences)

    def find_func(self, gen_ids: list[int]) -> dict | None:
        for func in self.registry:
            func_token = self.encode(f'"{func["name"]}"')
            if self.verify_sequence(gen_ids, func_token):
                return func

        return None

    def init_param_seq(self) -> None:
        self.param_sequences = []

        if self.curr_func is None:
            return False

        for param in self.curr_func["parameters"]:
            param_tokens = self.encode(f'"{param}"')
            self.param_sequences.append(param_tokens)

        self.param_seq_i = 0

    def next_param_token(self) -> set[int]:
        valid = set()

        for param in self.param_sequences:
            if self.param_seq_i < len(param):
                valid.add(param[self.param_seq_i])

        return valid

    def increment_param(self, curr_token: int) -> bool:
        updated_params = []

        for param in self.param_sequences:
            if self.param_seq_i < len(param):
                if param[self.param_seq_i] == curr_token:
                    updated_params.append(param)

        self.param_sequences = updated_params
        self.param_seq_i += 1

        if not self.param_sequences:
            return False

        return all(self.param_seq_i >= len(param)
                   for param in self.param_sequences)

    def find_param(self, gen_ids: list[int]) -> dict | None:
        if self.curr_func is None:
            return None

        for param_name, param_type in self.curr_func["parameters"].items():
            param_token = self.encode(f'"{param_name}"')
            if self.verify_sequence(gen_ids, param_token):
                return param_type

        return None

    def valid_tokens(self, state: JSON_State, gen_ids: list[int]) -> set[int]:

        if state == JSON_State.EXPECT_OPEN_OBJ:
            return self.valid_first_tokens(["{"])

        elif state == JSON_State.EXPECT_NAME_KEY:
            if len(self.expected_sequence) <= 0:
                self.init_sequence('"name"')
            return self.next_token()

        elif state == JSON_State.EXPECT_COLON:
            return self.valid_first_tokens([":"])

        elif state == JSON_State.EXPECT_FUNCTION_NAME:
            if not self.func_sequences:
                self.init_func_seq()
            return self.next_func_token()

        elif state == JSON_State.EXPECT_PARAMETERS_KEY:
            if len(self.expected_sequence) <= 0:
                self.init_sequence(',"parameters":')
            return self.next_token()

        elif state == JSON_State.EXPECT_PARAMETER_KEY:
            if not self.param_sequences:
                self.init_param_seq()
            return self.next_param_token()

        elif state == JSON_State.EXPECT_PARAMETER_COLON:
            return self.valid_first_tokens([":"])

        elif state == JSON_State.EXPECT_STRING_OPEN:
            return set(self.encode('"'))

        elif state == JSON_State.IN_STRING:
            return self.valid_first_tokens(['"'])

        elif state == JSON_State.EXPECT_NUMBER:
            pass

        elif state == JSON_State.EXPECT_BOOLEAN:
            return self.valid_first_tokens(["true", "false"])

        elif state == JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS:
            return self.valid_first_tokens([",", "}"])

        elif state == JSON_State.EXPECT_OBJ_END:
            return self.valid_first_tokens(["}"])

        elif state == JSON_State.DONE:
            return set()

        return set()

    def update_state(self, state: JSON_State, curr_token: int,
                     gen_ids: list[int]) -> JSON_State:
        self.prev_token = curr_token

        if state == JSON_State.EXPECT_NAME_KEY:
            if self.increment_index(curr_token):
                self.expected_sequence = []
                self.sequence_i = 0
                return JSON_State.EXPECT_COLON

        elif state == JSON_State.EXPECT_COLON:
            return JSON_State.EXPECT_FUNCTION_NAME

        elif state == JSON_State.EXPECT_FUNCTION_NAME:
            if self.increment_func(curr_token):
                func = self.find_func(gen_ids)

                if func is not None:
                    self.curr_func = func
                    self.func_sequences = []
                    self.func_seq_i = 0
                return JSON_State.EXPECT_PARAMETERS_KEY

        elif state == JSON_State.EXPECT_PARAMETERS_KEY:
            if self.increment_index(curr_token):
                self.expected_sequence = []
                self.sequence_i = 0
                return JSON_State.EXPECT_OPEN_OBJ

        if state == JSON_State.EXPECT_OPEN_OBJ:
            return JSON_State.EXPECT_PARAMETER_KEY

        elif state == JSON_State.EXPECT_PARAMETER_KEY:
            if self.increment_param(curr_token):
                param = self.find_param(gen_ids)

                if param is not None:
                    self.curr_param = param
                    self.param_sequences = []
                    self.param_seq_i = 0
                return JSON_State.EXPECT_PARAMETER_COLON

        elif state == JSON_State.EXPECT_PARAMETER_COLON:
            if self.curr_param["type"] == "number":
                return JSON_State.EXPECT_NUMBER
            if self.curr_param["type"] == "string":
                return JSON_State.EXPECT_STRING_OPEN
            if self.curr_param["type"] == "boolean":
                return JSON_State.EXPECT_BOOLEAN

        elif state == JSON_State.EXPECT_STRING_OPEN:
            return JSON_State.IN_STRING

        elif state == JSON_State.IN_STRING:
            pass

        elif state == JSON_State.EXPECT_NUMBER:
            pass

        elif state == JSON_State.EXPECT_BOOLEAN:
            return JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS

        elif state == JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS:
            if self.token_is(curr_token, ","):
                return JSON_State.EXPECT_PARAMETER_KEY

            elif self.token_is(curr_token, "}"):
                return JSON_State.EXPECT_OBJ_END

        elif state == JSON_State.EXPECT_OBJ_END:
            return JSON_State.DONE

        return state


def constrained_decoder(model: Small_LLM_Model, prompt: str, registry: list[dict]) -> list[int]:
    state = JSON_State.EXPECT_NAME_KEY
    state_machine = JSON_Machine(model, registry)
    input = '{"prompt":"' + prompt + '",'
    gen_ids = encode_text(model, input)

    while state != JSON_State.DONE:
        logits = model.get_logits_from_input_ids(gen_ids)
        if hasattr(logits, "__len__") and logits and hasattr(logits[0], "__len__"):
            logits = logits[0]

        valid_ids = state_machine.valid_tokens(state, gen_ids)
        # HANDLE GRACEFULLY LATER
        if not valid_ids:
            print(f"No valid tokens for state {state}")
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
