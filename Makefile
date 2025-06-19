install:
	pip install -e .[dev,examples,ui]

lint:
	black --check src tests examples
	flake8 src tests examples

typecheck:
	mypy src

test:
	pytest

docs:
	mkdocs build --strict

build:
	python -m build

.PHONY: install lint typecheck test docs build
