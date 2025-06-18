FROM python:3.11-slim-bookworm AS builder

# 1. Install uv binary (pin specific version for reproducibility)
COPY --from=ghcr.io/astral-sh/uv:python3.11-bookworm-slim /uv /uvx /bin/uv

# 2. Set build environment variables for caching and bytecode compilation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

# 3. Install system build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev g++ && \
    rm -rf /var/lib/apt/lists/*

# 4. Copy only lock and project file to install dependencies separately
COPY pyproject.toml uv.lock* ./

# 5. Sync dependencies (without installing our project), reuse uv cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 6. Copy full source and install the project itself (non-editable mode recommended)
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# 7. Clean up build tools to reduce image size
RUN apt-get remove -y gcc g++ libffi-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy environment produced by uv
COPY --from=builder /app/.venv /app/.venv

# Make uv binaries available if needed at runtime
COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /bin/uvx /bin/uvx

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Copy application source
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
