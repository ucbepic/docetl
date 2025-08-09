# Build stage for Python dependencies
FROM python:3.11-slim AS python-builder

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV DOCETL_HOME_DIR="/docetl-data"

WORKDIR /app

COPY pyproject.toml ./
COPY docetl/ ./docetl/
COPY server/ ./server/
COPY tests/ ./tests/
RUN touch README.md

# Create venv and sync dependencies (including extras)
RUN uv venv && \
    uv sync --all-extras --all-groups

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
COPY pyproject.toml ./
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