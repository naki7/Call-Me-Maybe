VENV    := venv
UV		:= uv run python -m
PYTHON  := $(VENV)/bin/python3
PIP     := $(VENV)/bin/pip
MAIN    := ./src/main.py

.PHONY: install run debug clean lint lint-strict

all: install

install:
	python3 -m venv $(VENV)
	source .venv/bin/activate
	unzip llm_sdk.zip
	unzip llm_sdk.zip

run:
	$(UV) src

debug:
	$(PYTHON) -m pdb $(MAIN) $(FILE) $(OUTPUT)

clean:
	rm -rf $(VENV) build dist *.egg-info .mypy_cache maps
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
