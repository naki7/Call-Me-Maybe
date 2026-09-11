import json

from llm_sdk.llm_sdk import Small_LLM_Model


class JSONState():
    def input_ids(self, vocab: dict[str, int], input: list[str]) -> list[int]:
        ids: list[int | list[int]] = []
        space_time = 'waiting'
        for token in input:
            temp = None
            temp = vocab.get(token)
            if temp and space_time == "active":
                temp = vocab.get(f"Ġ{token}")
            if temp is None:
                temp = []
                stripped = token.strip(".\":,")
                if token.count("_") > 0:
                    split = stripped.split("_")
                    temp.append(vocab.get(split[0]))
                    for word in split[1:]:
                        trial = vocab.get(f"_{word}")
                        if trial is None:
                            if word == "greet":
                                temp.append(vocab.get("_g"))
                                temp.append(vocab.get("reet"))
                            elif word == "substitute":
                                temp.append(vocab.get("_sub"))
                                temp.append(vocab.get("stitute"))
                            else:
                                temp.append(vocab.get("_"))
                                temp.append(vocab.get(word))
                        else:
                            temp.append(trial)
                    temp.append(vocab.get("\""))
                    temp.append(vocab.get(","))
                elif token.startswith("\"") or token.endswith(
                        "\":") or token.endswith(
                            "\",") or token.endswith("\""):
                    if space_time == 'waiting':
                        if token.startswith("\""):
                            temp.append(vocab.get("\""))
                            space_time = 'ready'
                    if token.count(".") == 1:
                        temp.append(vocab.get(f"Ġ{stripped}"))
                        temp.append(vocab.get(".\","))
                        space_time = 'waiting'
                    else:
                        temp.append(vocab.get(stripped))
                        if token.endswith("\"") or token.endswith(
                                ":") or token.endswith(","):
                            if token.endswith("\":"):
                                temp.append(vocab.get("\""))
                                temp.append(vocab.get(":"))
                            elif token.endswith("\","):
                                temp.append(vocab.get("\","))
                            elif token.endswith("\""):
                                temp.append(vocab.get("\""))
                            space_time = 'waiting'
                        elif space_time == "ready":
                            space_time = 'active'
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
        all_tokens: set[int] = {None}
        for id in ids:
            all_tokens.add(id)
        all_tokens.remove(None)
        return all_tokens



def produce_vocab(model: Small_LLM_Model) -> None:
    vocab_path = model.get_path_to_vocab_file()
    vocab = {}
    with open(vocab_path, "r") as vocab_file:
        vocab = json.load(vocab_file)
    funcs = ''
    with open('./data/input/functions_definition.json', 'r') as func_file:
        funcs = func_file.read()
    # g_funcs = funcs.replace(" ", "Ġ")
    # print(g_funcs)
    funcs_as_ids = funcs.split()
    state = JSONState()
    # print(funcs_as_ids)
    ids = state.input_ids(vocab, funcs_as_ids)
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
    print(model.decode(ids))
    # print(vocab.get("_name"))
    # print(vocab.get("_numbers"))
    # for char in funcs:
    #     print(char)
    # print(funcs_as_ids)
    tokens = state.valid_tokens(ids)
    print(tokens)
    for_show = []
    for token in tokens:
        for_show.append(token)
    print(model.decode(for_show))
