import json
from enum import Enum, auto
from typing import Any

from llm_sdk.llm_sdk import Small_LLM_Model
from src.contextualizer import build_model_context


def encode_text(model: Small_LLM_Model, text: str) -> list[int]:
    ids = model.encode(text)

    try:
        ids = ids.tolist()
    except AttributeError:
        ids = list(ids)

    if ids and isinstance(ids[0], list):
        ids = ids[0]

    return [int(token_id) for token_id in ids]


def load_vocab(model: Small_LLM_Model) -> Any:
    vocab_path = model.get_path_to_vocab_file()
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        vocab = json.load(vocab_file)

    return vocab


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
    IN_STRING_ESCAPE = auto()
    IN_STRING_UNICODE_HEX = auto()

    EXPECT_NUMBER = auto()
    EXPECT_NUMBER_DIGIT = auto()
    EXPECT_DECIMAL_DIGIT = auto()
    IN_ZERO = auto()
    IN_INTEGER = auto()
    IN_DECIMAL = auto()
    EXPECT_EXPONENT = auto()
    EXPECT_EXPONENT_SIGN_OR_DIGIT = auto()
    IN_EXPONENT = auto()

    EXPECT_BOOLEAN = auto()
    EXPECT_NULL = auto()

    EXPECT_COMMA_OR_OBJ_PARAMETERS = auto()
    EXPECT_OBJ_END = auto()

    DONE = auto()


class JSON_Machine:
    def __init__(self, model: Small_LLM_Model, registry: list[dict[Any, Any]],
                 vocab: dict[Any, Any]):
        self.model = model
        self.registry = registry
        self.vocab = vocab
        self.func_names = [func["name"] for func in self.registry]

        self._string_token_cache: tuple[set[int], set[int]] | None = None

        self.curr_func: dict[Any, Any] = {}
        self.curr_param: dict[Any, Any] = {}
        self.used_param: set[Any] = set()
        self.prev_token: None | int = None

        self.expected_sequence: list[Any] = []
        self.sequence_i = 0

        self.func_sequences: list[Any] = []
        self.func_seq_i = 0

        self.param_sequences: list[Any] = []
        self.param_seq_i = 0

        self.bool_sequences: list[Any] = []
        self.bool_seq_i = 0

        self.null_sequences: list[Any] = []
        self.null_seq_i = 0

    def encode(self, text: str) -> list[int]:
        return encode_text(self.model, text)

    def decode(self, token_id: int) -> Any:
        return self.model.decode([token_id])

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

    def find_func(self, gen_ids: list[int]) -> dict[Any, Any] | None:
        for func in self.registry:
            func_token = self.encode(f'"{func["name"]}"')
            if self.verify_sequence(gen_ids, func_token):
                return func

        return None

    def init_param_seq(self) -> None:
        self.param_sequences = []

        if self.curr_func is None:
            return

        for param in self.curr_func["parameters"]:
            if param in self.used_param:
                continue
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

    def find_param(self, gen_ids: list[int]) -> tuple[
            str, dict[Any, Any]] | None:
        if self.curr_func is None:
            return None

        for param_name, param_type in self.curr_func["parameters"].items():
            if param_name in self.used_param:
                continue

            param_token = self.encode(f'"{param_name}"')
            if self.verify_sequence(gen_ids, param_token):
                return param_name, param_type

        return None

    def init_bool_seq(self) -> None:
        self.bool_sequences = [
            self.encode("true"),
            self.encode("false")
        ]
        self.bool_seq_i = 0

    def next_bool_token(self) -> set[int]:
        valid = set()

        for bool_value in self.bool_sequences:
            if self.bool_seq_i < len(bool_value):
                valid.add(bool_value[self.bool_seq_i])

        return valid

    def increment_bool(self, curr_token: int) -> bool:
        updated_bools = []

        for bool_value in self.bool_sequences:
            if self.bool_seq_i < len(bool_value):
                if bool_value[self.bool_seq_i] == curr_token:
                    updated_bools.append(bool_value)

        self.bool_sequences = updated_bools
        self.bool_seq_i += 1

        if not self.bool_sequences:
            return False

        return all(self.bool_seq_i >= len(bool_value)
                   for bool_value in self.bool_sequences)

    def init_null_seq(self) -> None:
        self.null_sequences = [self.encode("null")]
        self.null_seq_i = 0

    def next_null_token(self) -> set[int]:
        valid = set()
        for null_value in self.null_sequences:
            if self.null_seq_i < len(null_value):
                valid.add(null_value[self.null_seq_i])
        return valid

    def increment_null(self, curr_token: int) -> bool:
        updated_nulls = []
        for null_value in self.null_sequences:
            if self.null_seq_i < len(null_value):
                if null_value[self.null_seq_i] == curr_token:
                    updated_nulls.append(null_value)

        self.null_sequences = updated_nulls
        self.null_seq_i += 1

        if not self.null_sequences:
            return False

        return all(self.null_seq_i >= len(null_value)
                   for null_value in self.null_sequences)

    def has_unused_params(self) -> bool:
        if self.curr_func is None:
            return False

        return any(param not in self.used_param
                   for param in self.curr_func["parameters"])

    def digit_tokens(self) -> set[int]:
        digits = set()

        for digit in "0123456789":
            digits.update(self.encode(digit))

        return digits

    def token_is(self, curr_token: int, text: str) -> bool:
        tokens = self.encode(text)
        return len(tokens) == 1 and curr_token == tokens[0]

    def token_is_digit(self, curr_token: int) -> bool:
        possible_digits = self.digit_tokens()

        for digit in possible_digits:
            if curr_token == digit:
                return True

        return False

    def token_valid_from_string(self, text: str,
                                escape_pending: bool = False) -> bool:
        state = JSON_State.IN_STRING
        escape_mode = escape_pending
        unicode_digits = 0

        for char in text:
            if state == JSON_State.IN_STRING:
                if escape_mode:
                    if char in '\\/bfnrt"':
                        escape_mode = False
                        continue
                    if char == 'u':
                        state = JSON_State.IN_STRING_UNICODE_HEX
                        escape_mode = False
                        unicode_digits = 0
                        continue
                    return False

                if char == "\\":
                    escape_mode = True
                    continue

                if char == '"':
                    if self.has_unused_params():
                        state = JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS
                    else:
                        state = JSON_State.EXPECT_OBJ_END
                    continue

                if ord(char) < 0x20:
                    return False

            elif state == JSON_State.IN_STRING_UNICODE_HEX:
                if char in "0123456789abcdefABCDEF":
                    unicode_digits += 1
                    if unicode_digits == 4:
                        state = JSON_State.IN_STRING
                    continue
                return False

            elif state == JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS:
                if char == ",":
                    state = JSON_State.EXPECT_PARAMETER_KEY
                elif char == "}":
                    state = JSON_State.EXPECT_OBJ_END
                else:
                    return False

            elif state == JSON_State.EXPECT_PARAMETER_KEY:
                return False

            elif state == JSON_State.EXPECT_OBJ_END:
                if char == "}":
                    state = JSON_State.DONE
                else:
                    return False

            elif state == JSON_State.DONE:
                return False

        return not escape_mode

    def _build_token_cache(self) -> tuple[set[int], set[int]]:
        valid_tokens: set[int] = set()
        valid_esc_tokens: set[int] = set()

        for token_id in range(len(self.vocab)):
            token_text = self.decode(token_id)

            if self.token_valid_from_string(token_text):
                valid_tokens.add(token_id)

            if self.token_valid_from_string(token_text, escape_pending=True):
                valid_esc_tokens.add(token_id)

        return valid_tokens, valid_esc_tokens

    def valid_string_tokens(self) -> set[int]:
        if self._string_token_cache is None:
            self._string_token_cache = self._build_token_cache()

        return self._string_token_cache[0]

    def valid_string_escape_tokens(self) -> set[int]:
        if self._string_token_cache is None:
            self._string_token_cache = self._build_token_cache()

        return self._string_token_cache[1]

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
            if not self.has_unused_params():
                return self.valid_first_tokens(["}"])

            if not self.param_sequences:
                self.init_param_seq()
            return self.next_param_token()

        elif state == JSON_State.EXPECT_PARAMETER_COLON:
            return self.valid_first_tokens([":"])

        elif state == JSON_State.EXPECT_STRING_OPEN:
            return set(self.encode('"'))

        elif state == JSON_State.IN_STRING:
            return self.valid_string_tokens()

        elif state == JSON_State.IN_STRING_ESCAPE:
            return self.valid_string_escape_tokens()

        elif state == JSON_State.EXPECT_NUMBER:
            digits = self.digit_tokens()
            digits.update(self.encode("-"))
            return digits

        elif state == JSON_State.EXPECT_NUMBER_DIGIT:
            return self.digit_tokens()

        elif state == JSON_State.IN_ZERO:
            chars = set()
            chars.update(self.encode("."))
            chars.update(self.encode("e"))
            chars.update(self.encode("E"))

            if self.has_unused_params():
                chars.update(self.encode(","))
            else:
                chars.update(self.encode("}"))
            return chars

        elif state == JSON_State.IN_INTEGER:
            digits = self.digit_tokens()
            digits.update(self.encode("."))
            digits.update(self.encode("e"))
            digits.update(self.encode("E"))

            if self.has_unused_params():
                digits.update(self.encode(","))
            else:
                digits.update(self.encode("}"))

            return digits

        elif state == JSON_State.EXPECT_DECIMAL_DIGIT:
            return self.digit_tokens()

        elif state == JSON_State.IN_DECIMAL:
            digits = self.digit_tokens()
            digits.update(self.encode("e"))
            digits.update(self.encode("E"))

            if self.has_unused_params():
                digits.update(self.encode(","))
            else:
                digits.update(self.encode("}"))

            return digits

        elif state == JSON_State.EXPECT_EXPONENT:
            exponents = set(self.encode("e"))
            exponents.update(self.encode("E"))
            return exponents

        elif state == JSON_State.EXPECT_EXPONENT_SIGN_OR_DIGIT:
            tokens = self.digit_tokens()
            tokens.update(self.encode("+"))
            tokens.update(self.encode("-"))
            return tokens

        elif state == JSON_State.IN_EXPONENT:
            digits = self.digit_tokens()
            if self.has_unused_params():
                digits.update(self.encode(","))
            else:
                digits.update(self.encode("}"))
            return digits

        elif state == JSON_State.EXPECT_BOOLEAN:
            if not self.bool_sequences:
                self.init_bool_seq()
            return self.next_bool_token()

        elif state == JSON_State.EXPECT_NULL:
            if not self.null_sequences:
                self.init_null_seq()
            return self.next_null_token()

        elif state == JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS:
            if self.has_unused_params():
                return self.valid_first_tokens([","])
            return self.valid_first_tokens(["}"])

        elif state == JSON_State.EXPECT_OBJ_END:
            return self.valid_first_tokens(["}"])

        elif state == JSON_State.DONE:
            return set()

        return set()

    def update_state(self, state: JSON_State, curr_token: int,
                     gen_ids: list[int]) -> JSON_State:
        self.prev_token = curr_token
        token_text = self.model.decode([curr_token])

        if state in (JSON_State.IN_STRING, JSON_State.IN_STRING_ESCAPE,
                     JSON_State.IN_STRING_UNICODE_HEX):
            unicode_count = 0
            for char in token_text:
                if state == JSON_State.IN_STRING:
                    if char == "\\":
                        state = JSON_State.IN_STRING_ESCAPE
                        continue
                    if char == '"':
                        state = JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS
                        continue
                    continue

                elif state == JSON_State.IN_STRING_ESCAPE:
                    if char in '"\\/bfnrt':
                        state = JSON_State.IN_STRING
                        continue
                    if char == 'u':
                        state = JSON_State.IN_STRING_UNICODE_HEX
                        unicode_count = 0
                        continue
                    return state

                elif state == JSON_State.IN_STRING_UNICODE_HEX:
                    if char in "0123456789abcdefABCDEF":
                        unicode_count += 1
                        if unicode_count == 4:
                            state = JSON_State.IN_STRING
                            unicode_count = 0
                        continue
                    return state

                elif state == JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS:
                    if char == ",":
                        state = JSON_State.EXPECT_PARAMETER_KEY
                    elif char == "}":
                        state = JSON_State.EXPECT_OBJ_END
                    else:
                        return state

                elif state == JSON_State.EXPECT_OBJ_END:
                    if char == "}":
                        state = JSON_State.DONE
                    else:
                        return state

                elif state == JSON_State.DONE:
                    break

            return state

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
            if self.token_is(curr_token, "{"):
                return JSON_State.EXPECT_PARAMETER_KEY

        elif state == JSON_State.EXPECT_PARAMETER_KEY:
            if self.token_is(curr_token, "}"):
                return JSON_State.EXPECT_OBJ_END

            if not self.param_sequences:
                self.init_param_seq()

            if self.increment_param(curr_token):
                param = self.find_param(gen_ids)

                if param is not None:
                    param_name, param_type = param
                    self.curr_param = param_type
                    self.used_param.add(param_name)

                    self.param_sequences = []
                    self.param_seq_i = 0
                    return JSON_State.EXPECT_PARAMETER_COLON

        elif state == JSON_State.EXPECT_PARAMETER_COLON:
            if self.curr_param["type"] == "number" or self.curr_param[
                    "type"] == "integer":
                return JSON_State.EXPECT_NUMBER
            if self.curr_param["type"] == "string":
                return JSON_State.EXPECT_STRING_OPEN
            if self.curr_param["type"] == "boolean":
                return JSON_State.EXPECT_BOOLEAN
            if self.curr_param["type"] == "list":
                return JSON_State.EXPECT_OPEN_OBJ
            if self.curr_param["type"] == "dictionary":
                return JSON_State.EXPECT_OPEN_OBJ

        elif state == JSON_State.EXPECT_STRING_OPEN:
            return JSON_State.IN_STRING

        elif state == JSON_State.EXPECT_NUMBER:
            if self.token_is(curr_token, "-"):
                return JSON_State.EXPECT_NUMBER_DIGIT
            if self.token_is(curr_token, "0"):
                return JSON_State.IN_ZERO
            if self.token_is_digit(curr_token):
                return JSON_State.IN_INTEGER

        elif state == JSON_State.EXPECT_NUMBER_DIGIT:
            if self.token_is_digit(curr_token):
                return JSON_State.IN_INTEGER

        elif state == JSON_State.IN_ZERO:
            if self.token_is(curr_token, "."):
                return JSON_State.EXPECT_DECIMAL_DIGIT
            if self.token_is(curr_token, "e") or self.token_is(
                    curr_token, "E"):
                return JSON_State.EXPECT_EXPONENT_SIGN_OR_DIGIT
            if self.token_is(curr_token, ",") and self.has_unused_params():
                return JSON_State.EXPECT_PARAMETER_KEY
            if self.token_is(curr_token, "}"):
                return JSON_State.EXPECT_OBJ_END

        elif state == JSON_State.IN_INTEGER:
            if self.token_is_digit(curr_token):
                return JSON_State.IN_INTEGER
            if self.token_is(curr_token, "."):
                return JSON_State.EXPECT_DECIMAL_DIGIT
            if self.token_is(curr_token, "e") or self.token_is(
                    curr_token, "E"):
                return JSON_State.EXPECT_EXPONENT_SIGN_OR_DIGIT
            if self.token_is(curr_token, ",") and self.has_unused_params():
                return JSON_State.EXPECT_PARAMETER_KEY
            if self.token_is(curr_token, "}"):
                return JSON_State.EXPECT_OBJ_END

        elif state == JSON_State.EXPECT_DECIMAL_DIGIT:
            if self.token_is_digit(curr_token):
                return JSON_State.IN_DECIMAL

        elif state == JSON_State.IN_DECIMAL:
            if self.token_is_digit(curr_token):
                return JSON_State.IN_DECIMAL
            if self.token_is(curr_token, "e") or self.token_is(
                    curr_token, "E"):
                return JSON_State.EXPECT_EXPONENT_SIGN_OR_DIGIT
            if self.token_is(curr_token, ",") and self.has_unused_params():
                return JSON_State.EXPECT_PARAMETER_KEY
            if self.token_is(curr_token, "}"):
                return JSON_State.EXPECT_OBJ_END

        elif state == JSON_State.EXPECT_EXPONENT_SIGN_OR_DIGIT:
            if self.token_is(curr_token, "+") or self.token_is(
                    curr_token, "-"):
                return JSON_State.IN_EXPONENT
            if self.token_is_digit(curr_token):
                return JSON_State.IN_EXPONENT

        elif state == JSON_State.IN_EXPONENT:
            if self.token_is_digit(curr_token):
                return JSON_State.IN_EXPONENT
            if self.token_is(curr_token, ",") and self.has_unused_params():
                return JSON_State.EXPECT_PARAMETER_KEY
            if self.token_is(curr_token, "}"):
                return JSON_State.EXPECT_OBJ_END

        elif state == JSON_State.EXPECT_BOOLEAN:
            if self.increment_bool(curr_token):
                self.bool_sequences = []
                self.bool_seq_i = 0
                return JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS

        elif state == JSON_State.EXPECT_NULL:
            if self.increment_null(curr_token):
                self.null_sequences = []
                self.null_seq_i = 0
                return JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS

        elif state == JSON_State.EXPECT_COMMA_OR_OBJ_PARAMETERS:
            if self.token_is(curr_token, ","):
                return JSON_State.EXPECT_PARAMETER_KEY

            elif self.token_is(curr_token, "}"):
                return JSON_State.EXPECT_OBJ_END

        elif state == JSON_State.EXPECT_OBJ_END:
            return JSON_State.DONE

        return state


def constrained_decoder(model: Small_LLM_Model, prompt: str,
                        registry: list[dict[Any, Any]],
                        vocab: dict[str, int]) -> list[int]:
    state = JSON_State.EXPECT_NAME_KEY
    state_machine = JSON_Machine(model, registry, vocab)

    context = build_model_context(prompt, registry)
    escaped_prompt = json.dumps(prompt)
    output_prefix = '{"prompt":' + escaped_prompt + ','
    input_text = context + "\n\n" + output_prefix

    gen_ids = encode_text(model, input_text)
    gen_result_ids = encode_text(model, output_prefix)

    next_token: int | None = None

    while state != JSON_State.DONE:
        logits = model.get_logits_from_input_ids(gen_ids)
        if hasattr(logits, "__len__") and logits and hasattr(
                logits[0], "__len__"):
            if isinstance(logits[0], list):
                logits = logits[0]

        valid_ids = state_machine.valid_tokens(state, gen_ids)
        if not valid_ids:
            raise RuntimeError(f"No valid tokens for state {state.name}")

        valid_logits = {
            token_id: logits[token_id]
            for token_id in valid_ids
            if 0 <= token_id < len(logits)
        }
        if not valid_logits:
            raise RuntimeError(
                f"No valid token IDs remain in logits for state {state.name}"
            )

        next_token = max(valid_logits, key=lambda token: valid_logits[token])
        gen_ids.append(next_token)
        gen_result_ids.append(next_token)
        state = state_machine.update_state(state, next_token, gen_ids)
        print(model.decode(gen_result_ids))

    return gen_result_ids
