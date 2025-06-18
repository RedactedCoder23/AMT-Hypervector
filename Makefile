install:
	pip install -e .[dev]

lint:
	black --check src tests
	flake8 src tests

typecheck:
	mypy src

test:
	pytest

docs:
	mkdocs build --strict

docs-build: docs

build:
	python -m build

.PHONY: install lint typecheck test docs docs-build build
