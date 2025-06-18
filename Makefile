install:
	pip install -e .[dev]
lint:
	flake8 src tests
typecheck:
	mypy src
test:
	pytest
build:
	python -m build
