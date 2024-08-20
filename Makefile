.PHONY: tests lint install mypy update

tests:
	poetry run pytest

lint:
	poetry run ruff check motion/* --fix

install:
	pip install poetry
	poetry install --all-extras

mypy:
	poetry run mypy

update:
	poetry update