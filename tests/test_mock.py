import unittest
from unittest.mock import patch

from src import main as main_module

class TestMain(unittest.TestCase):

    @patch("src.main.obj_to_json")
    @patch("src.main.validate_func_call")
    @patch("src.main.validate_gened_json")
    @patch("src.main.constrained_decoder")
    @patch("src.main.load_vocab")
    @patch("src.main.Small_LLM_Model")
    @patch("src.main.json_to_obj")
    @patch("src.main.load_registry")
    @patch("src.main.parse_args")
    def test_processes_multiple_prompts(
        self,
        mock_args,
        mock_registry,
        mock_json,
        mock_model_class,
        mock_vocab,
        mock_decoder,
        mock_validate_json,
        mock_validate_call,
        mock_write,
    ):
        """Runs a mock test of the main.py file to verify multiple variables"""
        mock_args.return_value.functions_definition = "functions.json"
        mock_args.return_value.input = "input.json"
        mock_args.return_value.output = "output.json"

        mock_registry.return_value = [{"name": "fn_greet"}]
        mock_json.return_value = [
            {"prompt": "Greet Alex"},
            {"prompt": "Greet Sam"},
        ]

        mock_model = mock_model_class.return_value
        mock_model.decode.return_value = "generated JSON"
        mock_vocab.return_value = {}

        mock_decoder.return_value = [1, 2, 3]
        mock_validate_json.return_value = {
            "prompt": "Greet Alex",
            "name": "fn_greet",
            "parameters": {"name": "Alex"},
        }
        mock_validate_call.side_effect = lambda call, registry: call

        main_module.main()

        self.assertEqual(mock_decoder.call_count, 2)
        mock_write.assert_called_once()
        saved_results = mock_write.call_args.args[0]
        self.assertEqual(len(saved_results), 2)

if __name__ == "__main__":
    unittest.main()
