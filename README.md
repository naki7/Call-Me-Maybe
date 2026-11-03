*This project has been created as part of the 42 curriculum by joshde-s*

# Call Me Maybe

## Description

This project explores function calling with large language models. Instead of
    asking the model to answer the question directly, the program asks it to
    choose a valid function and produce the corresponding arguments in a strict
    JSON structure. The goal is to generate machine-readable calls such as:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The implementation combines a function registry, prompt parsing, a constrained
    decoder, and validation so that only schema-compliant calls are emitted.

## Instructions

### Installation

```bash
uv sync
```

If you want to run the project in the local virtual environment manually:

```bash
source .venv/bin/activate
```

### Execution

```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

You can also use the built-in Makefile aliases:

```bash
make install
make run
```

### Example Usage

```bash
uv run python -m src
```

This reads the default prompt and function files and writes the result to
    `data/output/function_calling_results.json`.

## Algorithm Explanation

The project follows a constrained-decoding workflow:

1. The prompt and registry are converted into a model context.
2. The program builds a JSON state machine to enforce the expected call
    structure.
3. At each generation step the decoder filters logits to allow only tokens
    that keep the output valid.
4. Function names and parameter keys are restricted to values declared in the
    `functions_definitions input file`.
5. Each generated output is validated against the registry before being saved.

This is important because the model alone is not reliable enough to produce
    valid JSON by chance. The decoder narrows the token space so the output
    always remains parseable and aligned with the expected schema.

## Design Decisions

- The CLI uses a file-based approach so prompts and function registries can be
    changed without modifying code.
- The JSON validator enforces exact keys and required arguments.
- Numeric parameters are normalized to Python `float` values, which matches the
    project contract and keeps JSON compatible.
- When the vocab is used to create strings which can vary a lot in required
    logits, a local cache is created to store all the important tokens for both
    strings and string escape symbols so that the program doesn't have to waste
    resources re-checking the vocab file.

## Performance Analysis

The project is designed to prioritize reliability over raw model output. With
    constrained decoding, the JSON generation is kept structurally valid, which
    avoids the most common failure mode: malformed function calls. The approach
    is appropriate for smaller models because it reduces the search space
    dramatically.

## Challenges Faced

The main challenges were ensuring valid JSON generation and matching the
    registry schema exactly. In particular, function names, parameter names,
    and numerical types needed to remain consistent with the project
    specification. The solution was to add guarding logic at both generation
    and validation stages.

## Testing Strategy

The project includes tests for JSON validity, function validation, output
    schema checks, and edge cases like large numbers, negatives, empty
    strings, and special characters.

```bash
uv run python -m unittest discover -s tests -v
```

## Resources

### Technical References

- [JSON Schema](https://json-schema.org/)
- [Pydantic](https://docs.pydantic.dev/)
- [Qwen model docs](https://huggingface.co/Qwen)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

### AI Usage

AI was used to help reason about constrained decoding, token masks, schema
    validation, and debugging logic for prompt handling.
