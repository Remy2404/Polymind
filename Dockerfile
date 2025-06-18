# syntax=docker/dockerfile:1.4

FROM python:3.11-slim-bookworm AS base

FROM base AS builder
# Copy uv binaries from the distroless uv image
COPY --from=ghcr.io/astral-sh/uv:python3.11-bookworm-slim /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

# Install build dependencies
RUN apt-get update \
  && apt-get install -y --no-install-recommends gcc g++ libffi-dev \
  && rm -rf /var/lib/apt/lists/*

# Install dependencies (without installing project) using cached uv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy app source and install project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Remove build tools
RUN apt-get remove -y gcc g++ libffi-dev \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

##########
# Final #
##########
FROM base
WORKDIR /app

# Copy virtual environment and uv binaries
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
