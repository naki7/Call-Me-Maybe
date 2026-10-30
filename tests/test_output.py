import tempfile
import json
import unittest
from pathlib import Path


class TestOutputFile(unittest.TestCase):
    """Test the required output JSON format."""

    def test_output_is_a_json_array_of_function_calls(self) -> None:
        """Write and read the expected output structure."""
        results = [
            {
                "prompt": "What is the sum of 2 and 3?",
                "name": "fn_add_numbers",
                "parameters": {"a": 2.0, "b": 3.0},
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "nested" / "results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(results, file, indent=2)
                file.write("\n")

            with output_path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

        self.assertIsInstance(loaded, list)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            set(loaded[0]),
            {"prompt", "name", "parameters"},
        )
        self.assertIs(type(loaded[0]["parameters"]["a"]), float)
        self.assertEqual(loaded[0]["parameters"]["a"], 2.0)


if __name__ == "__main__":
    unittest.main()
