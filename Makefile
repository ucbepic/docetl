.PHONY: tests tests-basic lint install mypy update

tests:
	poetry run pytest

tests-basic:
	poetry run pytest tests/test_basic.py

lint:
	poetry run ruff check motion/* --fix

install:
	pip install poetry
	poetry install --all-extras

mypy:
	poetry run mypy

update:
	poetry update