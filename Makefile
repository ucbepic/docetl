.PHONY: tests tests-basic lint install mypy update

tests:
	poetry run pytest

tests-basic:
	poetry run pytest -n auto \
	tests/test_map.py \
		tests/test_map_parallel.py \
		tests/test_filter.py \
		tests/test_api.py

lint:
	poetry run ruff check docetl/* --fix

install:
	pip install poetry
	poetry install --all-extras

mypy:
	poetry run mypy

update:
	poetry update