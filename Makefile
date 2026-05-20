.PHONY: setup test lint

setup:
	uv sync --group dev
	uv run pre-commit install

test:
	uv run pytest --cov tests -v

lint:
	uv run ruff check --fix --config pyproject.toml
	uv run ruff format --config pyproject.toml
