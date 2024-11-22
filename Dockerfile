# Build stage for Python dependencies
FROM python:3.11-slim AS python-builder

RUN pip install poetry==1.4.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    DOCETL_HOME_DIR="/docetl-data"

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY docetl/ ./docetl/
COPY server/ ./server/
COPY tests/ ./tests/
RUN touch README.md

# Install with --no-root first for dependencies, then install with root for entrypoints
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --all-extras --no-root && \
    poetry install --all-extras

# Build stage for Node.js dependencies
FROM node:20-alpine AS node-builder

WORKDIR /app/website

# Update DOCETL_HOME_DIR to match final location
ENV DOCETL_HOME_DIR="/docetl-data"

COPY website/package*.json ./
RUN npm install
COPY website/ ./
RUN npm run build

# Final runtime stage
FROM python:3.11-slim AS runtime

# Install Node.js
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python virtual environment from builder
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    DOCETL_HOME_DIR="/docetl-data"

COPY --from=python-builder /app/.venv ${VIRTUAL_ENV}

# Copy Python application files
COPY docetl/ ./docetl/
COPY server/ ./server/
COPY tests/ ./tests/
COPY pyproject.toml poetry.lock ./
COPY .env ./

# Copy Node.js dependencies and application files
COPY --from=node-builder /app/website ./website

ENV PORT=3000

# Create data directory with appropriate permissions
RUN mkdir -p /docetl-data && chown -R nobody:nogroup /docetl-data && chmod 777 /docetl-data

# Define volume AFTER creating and setting permissions
VOLUME ["/docetl-data"]

# Expose ports for frontend and backend
EXPOSE 3000 8000

# Start both servers
CMD ["sh", "-c", "python3 server/app/main.py & cd website && npm run start"]