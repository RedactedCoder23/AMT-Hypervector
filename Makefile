install:
	pip install -e .[dev,examples,ui]

lint:
	black --check src tests
	flake8 src tests

typecheck:
	mypy src

test:
	pytest

docs:
	mkdocs build --strict

build:
	python -m build

.PHONY: install lint typecheck test docs build
