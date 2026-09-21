import json
from enum import Enum, auto

from llm_sdk.llm_sdk import Small_LLM_Model


class JSONState(Enum):
    EXPECT_OPEN_OBJ = auto()
    EXPECT_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_FUNCTION_NAME = auto()
    EXPECT_PARAMETERS_KEY = auto()
    EXPECT_PARAMETER_KEY = auto()
    EXPECT_PARAMETER_COLON = auto()
    EXPECT_STRING = auto()
    EXPECT_NUMBER = auto()
    EXPECT_BOOLEAN = auto()
    EXPECT_COMMA_OR_OBJ_END = auto()
    EXPECT_COMMA_OR_OBJ_PARAMETERS = auto()
    DONE = auto()


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
        # "hello",
        # "hello world",
        # " hello",
        # "fn_",
        # "fn_add",
        # "\"hello",
        # "\"hello world",
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
        '"hello"',
        '"hello world"',
        "40",
        "40.5",
        "-40",
        "-40.5",
        "true",
        "false",
        "null",
    ]
    script_exp(model, vocab, examples)

    # funcs = ''
    # with open('./data/input/functions_definition.json', 'r',
    #           encoding="utf-8") as func_file:
    #     funcs = func_file.read()
    # g_funcs = funcs.replace(" ", "Ġ")
    # print(g_funcs)

    # funcs_as_ids = funcs.split()
    # state = JSONState()
    # print(funcs_as_ids)

    # ids = state.input_ids(vocab, funcs_as_ids)
    # print(model.encode("a.\""))
    # print(model.encode("a"))
    # print(model.encode(" b"))
    # print(model.encode("b"))
    # print(model.encode("greet"))
    # print(model.encode("_greet"))
    # print(model.encode("substitute"))
    # print(vocab.get("Ġname"))
    # print(len(vocab))
    # for key in vocab:
    #     if vocab[key] == 70 or vocab[key] == 3744:
    #         print(key)
    #     if vocab[key] == 1966 or vocab[key] == 7660:
    #         print(key)
    # print(ids)
    # print(model.decode(ids))
    # print(vocab.get("_name"))
    # print(vocab.get("_numbers"))
    # for char in funcs:
    #     print(char)
    # print(funcs_as_ids)

    # tokens = state.valid_tokens(ids)
    # print(tokens)

    # for_show = []
    # for token in tokens:
    #     for_show.append(token)
    # print(model.decode(for_show))
