# Load environment variables from .env file
include .env

.PHONY: tests tests-basic lint install mypy update ui-install ui-run docker

# Existing commands
tests:
	poetry run pytest

tests-basic:
	poetry run pytest tests/basic
	poetry run pytest -s tests/test_api.py
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

# UI-related commands
UI_DIR := ./website 

install-ui:
	cd $(UI_DIR) && npm install

run-ui-dev:
	@echo "Starting server..."
	@python server/app/main.py & \
	echo "Starting UI development server..." && \
	cd $(UI_DIR) && HOST=${FRONTEND_HOST}  PORT=${FRONTEND_PORT} npm run dev

run-ui:
	@echo "Starting server..."
	@python server/app/main.py & \
	echo "Building UI..." && \
	cd $(UI_DIR) && npm run build && HOST=${FRONTEND_HOST}  PORT=${FRONTEND_PORT} NEXT_PUBLIC_FRONTEND_ALLOWED_HOSTS=${FRONTEND_ALLOWED_HOSTS} npm run start

# Single Docker command to build and run
docker:
	docker volume create docetl-data && \
	docker build -t docetl . && \
	docker run --rm -it \
		-p 3000:3000 \
		-p 8000:8000 \
		-v docetl-data:/docetl-data \
		-e FRONTEND_HOST=0.0.0.0 \
		-e FRONTEND_PORT=3000 \
		-e BACKEND_HOST=0.0.0.0 \
		-e BACKEND_PORT=8000 \
		docetl

# Add new command for cleaning up docker resources
docker-clean:
	docker volume rm docetl-data

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
	@echo "  make run-ui       : Run UI production server"
	@echo "  make docker       : Build and run docetl in Docker"
	@echo "  make docker-clean : Remove docetl Docker volume"
	@echo "  make help         : Show this help message"