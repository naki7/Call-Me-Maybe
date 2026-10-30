import json
import unittest
from pathlib import Path

from src.validator import validate_gened_json, validate_func_call


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "input" / "functions_definition.json"


class TestFunctionCallValidation(unittest.TestCase):
    """Test JSON parsing and function-call validation."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the actual function registry once."""
        with REGISTRY_PATH.open("r", encoding="utf-8") as file:
            cls.registry = json.load(file)

    def make_call(
        self,
        name: str = "fn_add_numbers",
        parameters: dict | None = None,
        prompt: str = "What is the sum of 2 and 3?",
    ) -> dict:
        """Create a function-call object for testing."""
        if parameters is None:
            parameters = {"a": 2.0, "b": 3.0}

        return {
            "prompt": prompt,
            "name": name,
            "parameters": parameters,
        }

    def test_valid_json_call(self) -> None:
        """Accept a valid call with all required top-level keys."""
        call = self.make_call()
        result = validate_gened_json(json.dumps(call))

        self.assertEqual(set(result), {"prompt", "name", "parameters"})
        self.assertEqual(result["name"], "fn_add_numbers")

    def test_reject_invalid_json(self) -> None:
        """Reject malformed JSON."""
        with self.assertRaises(ValueError):
            validate_gened_json('{"prompt": "hello"')

    def test_reject_non_object_json(self) -> None:
        """Reject a JSON array instead of an object."""
        with self.assertRaises(ValueError):
            validate_gened_json('[{"prompt": "hello"}]')

    def test_reject_missing_top_level_key(self) -> None:
        """Reject a call that omits a required key."""
        call = self.make_call()
        del call["parameters"]

        with self.assertRaises(ValueError):
            validate_gened_json(json.dumps(call))

    def test_reject_extra_top_level_key(self) -> None:
        """Reject unexpected top-level keys."""
        call = self.make_call()
        call["extra"] = "not allowed"

        with self.assertRaises(ValueError):
            validate_gened_json(json.dumps(call))

    def test_reject_wrong_prompt_type(self) -> None:
        """Reject a non-string prompt."""
        call = self.make_call()
        call["prompt"] = 123

        with self.assertRaises(ValueError):
            validate_gened_json(json.dumps(call))

    def test_reject_non_object_parameters(self) -> None:
        """Reject parameters that are not a JSON object."""
        call = self.make_call()
        call["parameters"] = ["2", "3"]

        with self.assertRaises(ValueError):
            validate_gened_json(json.dumps(call))

    def test_reject_unknown_function(self) -> None:
        """Reject a function absent from the registry."""
        call = self.make_call(name="fn_does_not_exist")

        with self.assertRaises(ValueError):
            validate_func_call(call, self.registry)

    def test_reject_missing_parameter(self) -> None:
        """Reject a call missing a required argument."""
        call = self.make_call(parameters={"a": 2.0})

        with self.assertRaises(ValueError):
            validate_func_call(call, self.registry)

    def test_reject_extra_parameter(self) -> None:
        """Reject arguments not declared in the function registry."""
        call = self.make_call(
            parameters={"a": 2.0, "b": 3.0, "c": 4.0}
        )

        with self.assertRaises(ValueError):
            validate_func_call(call, self.registry)

    def test_reject_boolean_as_number(self) -> None:
        """Reject booleans as numeric arguments."""
        call = self.make_call(parameters={"a": True, "b": 3.0})

        with self.assertRaises(ValueError):
            validate_func_call(call, self.registry)

    def test_reject_string_as_number(self) -> None:
        """Reject a string where a number is required."""
        call = self.make_call(parameters={"a": "2", "b": 3.0})

        with self.assertRaises(ValueError):
            validate_func_call(call, self.registry)

    def test_accept_large_numbers(self) -> None:
        """Accept large numeric values without truncating them."""
        call = self.make_call(
            parameters={"a": 1_000_000_000, "b": 2_000_000_000}
        )

        result = validate_func_call(call, self.registry)

        self.assertEqual(result["parameters"]["a"], 1_000_000_000)
        self.assertEqual(result["parameters"]["b"], 2_000_000_000)

    def test_accept_decimal_numbers(self) -> None:
        """Accept decimal numeric arguments."""
        call = self.make_call(parameters={"a": 2.5, "b": 3.25})

        result = validate_func_call(call, self.registry)

        self.assertEqual(result["parameters"]["a"], 2.5)
        self.assertEqual(result["parameters"]["b"], 3.25)

    def test_accept_negative_numbers(self) -> None:
        """Accept negative numeric arguments."""
        call = self.make_call(parameters={"a": -12.5, "b": -8.0})

        result = validate_func_call(call, self.registry)

        self.assertEqual(result["parameters"]["a"], -12.5)
        self.assertEqual(result["parameters"]["b"], -8.0)

    def test_accept_empty_string(self) -> None:
        """Accept an empty string for a string parameter."""
        call = self.make_call(
            name="fn_reverse_string",
            parameters={"s": ""},
            prompt='Reverse the string ""',
        )

        result = validate_func_call(call, self.registry)

        self.assertEqual(result["parameters"]["s"], "")

    def test_preserve_special_characters(self) -> None:
        """Preserve quotes, backslashes, and Unicode in strings."""
        value = 'hello "world" \\ café 🐈'
        call = self.make_call(
            name="fn_reverse_string",
            parameters={"s": value},
            prompt="Reverse a string with special characters",
        )

        encoded = json.dumps(call, ensure_ascii=False)
        parsed = validate_gened_json(encoded)
        result = validate_func_call(parsed, self.registry)

        self.assertEqual(result["parameters"]["s"], value)

    def test_regex_function_requires_all_three_parameters(self) -> None:
        """Validate a function with three required string arguments."""
        call = self.make_call(
            name="fn_substitute_string_with_regex",
            parameters={
                "source_string": "cats and cats",
                "regex": "cats",
                "replacement": "dogs",
            },
            prompt="Replace cats with dogs",
        )

        result = validate_func_call(call, self.registry)

        self.assertEqual(
            set(result["parameters"]),
            {"source_string", "regex", "replacement"},
        )

    def test_regex_function_rejects_wrong_parameter_type(self) -> None:
        """Reject a non-string regex replacement argument."""
        call = self.make_call(
            name="fn_substitute_string_with_regex",
            parameters={
                "source_string": "cats",
                "regex": "cats",
                "replacement": 123,
            },
        )

        with self.assertRaises(ValueError):
            validate_func_call(call, self.registry)


if __name__ == "__main__":
    unittest.main()
