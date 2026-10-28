import argparse


def parse_args() -> argparse.Namespace:
    """Get command-line arguments and parse them"""
    parsed = argparse.ArgumentParser(
        description="Generate function calls using an LLM like Qwen")

    parsed.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to JSON file with functions that would be called.",
    )

    parsed.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to JSON file which contains the prompts.",
    )

    parsed.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path to the JSON file that the output is written to.",
    )

    return parsed.parse_args()
