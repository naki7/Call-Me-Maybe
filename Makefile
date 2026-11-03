VENV    := .venv
PYTHON  := $(VENV)/bin/python
UV      := uv

.PHONY: install run debug clean lint lint-strict

all: install

install:
	mkdir -p /sgoinfre/joshde-s/.cache/uv
	export UV_CACHE_DIR=/sgoinfre/joshde-s/.cache/uv
	mkdir -p /sgoinfre/joshde-s/.cache/huggingface
	export HF_HOME=/sgoinfre/joshde-s/.cache/huggingface
	$(UV) sync

run:
	$(UV) run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

debug:
	$(UV) run python -m pdb -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json

clean:
	rm -rf $(VENV) build dist *.egg-info .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	flake8 . --exclude ./llm_sdk
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude ./llm_sdk --follow-imports=silent

lint-strict:
	flake8 . --exclude ./llm_sdk
	mypy . --strict --exclude ./llm_sdk --follow-imports=silent
