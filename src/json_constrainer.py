import json

from llm_sdk.llm_sdk import Small_LLM_Model


class JSONState():
    def input_ids(self, vocab: dict[str, int], input: list[str]) -> list[int]:
        ids: list[int | list[int]] = []
        for token in input:
            token.replace(" ", "Ġ")
            temp = None
            temp = vocab.get(token)
            if temp is None:
                temp = []
                stripped = token.strip(".\":,")
                if token.count("_") > 0:
                    split = stripped.split("_")
                    temp.append(vocab.get(split[0]))
                    for word in split[1:]:
                        word = f"_{word}"
                        temp.append(vocab.get(word))
                    temp.append(vocab.get("\""))
                    temp.append(vocab.get(","))
                elif token.startswith("\"") or token.endswith(
                        "\":") or token.endswith("\","):
                    if token.startswith("\""):
                        temp.append(vocab.get("\""))
                    temp.append(vocab.get(stripped))
                    if token.endswith("\":"):
                        temp.append(vocab.get("\""))
                        temp.append(vocab.get(":"))
                    elif token.endswith("\","):
                        temp.append(vocab.get("\""))
                        temp.append(vocab.get(","))
                    elif token.endswith(".\""):
                        temp.append(vocab.get("."))
                        temp.append(vocab.get("\""))
            if isinstance(temp, list):
                for item in temp:
                    if item is None:
                        continue
                    ids.append(item)
            elif temp is None:
                continue
            else:
                ids.append(temp)
        return ids

    def valid_tokens(self, ids: list[int]) -> set[int]:
        all_tokens: set[int] = {}
        for id in ids:
            all_tokens.add(id)
        return all_tokens



def produce_vocab(model: Small_LLM_Model) -> None:
    vocab_path = model.get_path_to_vocab_file()
    vocab = {}
    with open(vocab_path, "r") as vocab_file:
        vocab = json.load(vocab_file)
    funcs = ''
    with open('./data/input/functions_definition.json', 'r') as func_file:
        funcs = func_file.read()
    funcs.replace(" ", "Ġ")
    funcs_as_ids = funcs.split()
    state = JSONState()
    print(funcs_as_ids)
    ids = state.input_ids(vocab, funcs_as_ids)
    print(model.encode(" a"))
    print(model.encode("a"))
    print(model.encode(" b"))
    print(model.encode("b"))
    print(model.encode(" c"))
    print(model.encode("c"))
    print(vocab.get("Ġname"))
    print(len(vocab))
    for key in vocab:
        if vocab[key] == 264 or vocab[key] == 64:
            print(key)
    print(ids)
    print(model.decode(ids))
    # print(vocab.get("_name"))
    # print(vocab.get("_numbers"))
    # for char in funcs:
    #     print(char)
    # print(funcs_as_ids)
