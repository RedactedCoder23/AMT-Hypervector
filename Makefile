install:
	pip install -e .[dev]
lint:
	black --check src tests
	flake8 src tests
typecheck:
	mypy src
test:
	pytest
docs-build:
	mkdocs build --strict

build:
	python -m build
