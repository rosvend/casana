# syntax=docker/dockerfile:1.7

# Provider-agnostic image. Same artifact runs on Render, Cloud Run, EC2,
# Fly, App Runner — the host only injects PORT, DATABASE_URL, and the
# API keys.

###########################
# Stage 1: build frontend #
###########################
FROM node:20-slim AS frontend

WORKDIR /app/src/frontend

COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY src/frontend/ ./
RUN npm run build


############################
# Stage 2: Python runtime  #
############################
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Pin uv via the official image (much faster + smaller than `pip install uv`).
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

# Install Python deps first, without the project itself, so this layer
# caches across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Install Chromium plus its OS-level dependencies. --with-deps shells out
# to apt-get; clean its cache afterwards to keep the layer tight.
RUN uv run playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# Application source (data/ is the DANE geography lookup — runtime needs it).
COPY src/ ./src/
COPY data/ ./data/
COPY README.md ./

# Install the project itself now that the source is present.
RUN uv sync --frozen --no-dev

# Built SPA from stage 1 — main.py auto-mounts it when the dir exists.
COPY --from=frontend /app/src/frontend/dist ./src/frontend/dist

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
