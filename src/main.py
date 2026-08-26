from in_out_handler import json_to_obj, obj_to_json
from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Any
import sys


def main() -> None:
    obj_data: dict[Any, Any] = {}
    test_llm = Small_LLM_Model()
    tokens = ''

    if len(sys.argv) == 2:
        obj_data = json_to_obj(sys.argv[1])
        # obj_to_json(obj_data)
        tokens = test_llm.encode(obj_data)
        print(tokens)
    # print(obj_data)


if __name__ == '__main__':
    main()
