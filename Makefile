.PHONY: tests tests-basic lint install mypy update

tests:
	poetry run pytest

tests-basic:
	poetry run pytest tests/basic/test_basic_map.py
	poetry run pytest tests/basic/test_basic_reduce_resolve.py
	poetry run pytest tests/basic/test_basic_parallel_map.py
	poetry run pytest tests/basic/test_basic_filter_split_gather.py

lint:
	poetry run ruff check docetl/* --fix

install:
	pip install poetry
	poetry install --all-extras

mypy:
	poetry run mypy

update:
	poetry update