# Load environment variables from .env file
include .env

.PHONY: tests tests-basic lint install mypy update install-ui run-ui-dev run-ui docker docker-clean test-aws docs-serve help

# Test commands
tests:
	uv run pytest --ignore=tests/ranking --ignore=tests/test_ollama.py

tests-basic:
	uv run pytest -s tests/basic
	uv run pytest -s tests/test_api.py
	uv run pytest -s tests/test_runner_caching.py
	uv run pytest -s tests/test_pandas_accessors.py
	uv run pytest -s tests/test_output_modes.py

lint:
	uv run ruff check docetl/* --fix

install:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH=$$HOME/.local/bin:$$PATH && uv sync --all-groups --all-extras
	export PATH=$$HOME/.local/bin:$$PATH && uv run pre-commit install

mypy:
	uv run mypy

update:
	uv lock --upgrade

# UI-related commands
UI_DIR := ./website

install-ui:
	cd $(UI_DIR) && npm install

run-ui-dev:
	@echo "Starting server..."
	@export PATH=$$HOME/.local/bin:$$PATH; \
	uv sync --all-extras; \
	uv run python server/app/main.py & \
	echo "Starting UI development server..." && \
	cd $(UI_DIR) && HOST=$${FRONTEND_HOST:-127.0.0.1} PORT=$${FRONTEND_PORT:-3000} npm run dev

run-ui:
	@echo "Starting server..."
	@export PATH=$$HOME/.local/bin:$$PATH; \
	uv sync --all-extras; \
	uv run python server/app/main.py & \
	echo "Building UI..." && \
	cd $(UI_DIR) && npm run build && HOST=$${FRONTEND_HOST:-127.0.0.1} PORT=$${FRONTEND_PORT:-3000} NEXT_PUBLIC_FRONTEND_ALLOWED_HOSTS=$${FRONTEND_ALLOWED_HOSTS} npm run start

# Docker commands
docker:
	docker volume create docetl-data
	docker build -t docetl .
	@if [ -n "$${AWS_PROFILE}" ]; then \
		echo "[INFO] Detected AWS_PROFILE — including AWS credentials."; \
		DOCKER_AWS_FLAGS="-v ~/.aws:/root/.aws:ro \
			-e AWS_PROFILE=$${AWS_PROFILE} \
			-e AWS_REGION=$${AWS_REGION:-us-west-2}"; \
	else \
		echo "[INFO] No AWS_PROFILE set — skipping AWS credentials."; \
		DOCKER_AWS_FLAGS=""; \
	fi && \
	docker run --rm -it \
		-p 3000:3000 \
		-p 8000:8000 \
		-v docetl-data:/docetl-data \
		$$DOCKER_AWS_FLAGS \
		-e FRONTEND_HOST=0.0.0.0 \
		-e FRONTEND_PORT=3000 \
		-e BACKEND_HOST=0.0.0.0 \
		-e BACKEND_PORT=8000 \
		docetl

docker-clean:
	docker volume rm docetl-data

# Documentation commands
docs-serve:
	uv run mkdocs serve -a localhost:8001

# Test AWS connectivity
test-aws:
	@if ! command -v aws > /dev/null; then \
		echo "[WARNING] AWS CLI is not installed locally. Skipping local AWS credentials test."; \
	else \
		if [ ! -d ~/.aws ]; then \
			echo "[ERROR] AWS directory ~/.aws not found!"; \
			echo "👉 Configure AWS CLI first: https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html"; \
			exit 1; \
		fi; \
		if [ -z "$${AWS_PROFILE}" ]; then \
			echo "[WARNING] AWS_PROFILE is not set, using 'default' profile."; \
		fi; \
		if [ -z "$${AWS_REGION}" ]; then \
			echo "[WARNING] AWS_REGION is not set, using 'us-west-2' region."; \
		fi; \
		AWS_PROFILE_FLAG=$${AWS_PROFILE:+--profile $$AWS_PROFILE}; \
		echo "[INFO] Testing AWS credentials locally via cli..."; \
		if ! aws sts get-caller-identity $$AWS_PROFILE_FLAG > /dev/null; then \
			echo "[ERROR] Local AWS credentials test failed!"; \
			exit 1; \
		else \
			echo "[SUCCESS] Local AWS credentials are valid."; \
		fi; \
	fi
	@echo "[INFO] Testing AWS credentials inside Docker..."; \
	if ! docker run --rm -it \
		-v ~/.aws:/root/.aws:ro \
		-e AWS_PROFILE=$${AWS_PROFILE:-default} \
		-e AWS_REGION=$${AWS_REGION:-us-west-2} \
		amazon/aws-cli sts get-caller-identity > /dev/null; then \
		echo "[ERROR] Docker AWS credentials test failed!"; \
		exit 1; \
	else \
		echo "[SUCCESS] Docker AWS credentials are valid."; \
	fi

# Help command
help:
	@echo "Available commands:"
	@echo "  make tests        : Run all tests"
	@echo "  make tests-basic  : Run basic tests"
	@echo "  make lint         : Run linter"
	@echo "  make install      : Install dependencies using uv"
	@echo "  make mypy         : Run mypy for type checking"
	@echo "  make update       : Update dependencies"
	@echo "  make install-ui   : Install UI dependencies"
	@echo "  make run-ui-dev   : Run UI development server"
	@echo "  make run-ui       : Run UI production server"
	@echo "  make docker       : Build and run docetl in Docker"
	@echo "  make docker-clean : Remove docetl Docker volume"
	@echo "  make docs-serve   : Serve documentation locally on port 8001"
	@echo "  make test-aws     : Test AWS credentials configuration"
	@echo "  make help         : Show this help message"