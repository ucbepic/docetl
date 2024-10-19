.PHONY: tests tests-basic lint install mypy update ui-install ui-run

# Existing commands
tests:
	poetry run pytest

tests-basic:
	poetry run pytest tests/basic
	poetry run pytest tests/test_api.py
	poetry run pytest tests/test_runner_caching.py

lint:
	poetry run ruff check docetl/* --fix

install:
	pip install poetry
	poetry install --all-extras

mypy:
	poetry run mypy

update:
	poetry update

# New UI-related commands
UI_DIR := ./website 

install-ui:
	cd $(UI_DIR) && npm install

run-ui-dev:
	@echo "Starting server..."
	@python server/app/main.py & \
	echo "Starting UI development server..." && \
	cd $(UI_DIR) && npm run dev

run-ui:
	@echo "Starting server..."
	@python server/app/main.py & \
	echo "Building UI..." && \
	cd $(UI_DIR) && npm run build && npm run start

# Help command
help:
	@echo "Available commands:"
	@echo "  make tests        : Run all tests"
	@echo "  make tests-basic  : Run basic tests"
	@echo "  make lint         : Run linter"
	@echo "  make install      : Install dependencies using Poetry"
	@echo "  make mypy         : Run mypy for type checking"
	@echo "  make update       : Update dependencies"
	@echo "  make install-ui   : Install UI dependencies"
	@echo "  make run-ui-dev   : Run UI development server"
	@echo "  make run-ui-prod  : Run UI production server"
	@echo "  make help         : Show this help message"