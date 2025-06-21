# syntax=docker/dockerfile:1.4

#################
# Builder stage #
#################
FROM python:3.11-slim AS builder

# 1️⃣ Copy uv binaries from the distroless 'uv:latest' image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# 2️⃣ Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

# 3️⃣ Install system dependencies (including FFmpeg for voice processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Copy dependency files and install dependencies
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# 5️⃣ Copy application code and install the project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# 5.5️⃣ Download spaCy language model
RUN /app/.venv/bin/python -m ensurepip && \
    /app/.venv/bin/python -m pip install --upgrade pip && \
    /app/.venv/bin/python -m spacy download en_core_web_sm

# 6️⃣ Don't remove FFmpeg - it's needed for voice processing in production
RUN apt-get remove -y gcc g++ libffi-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#################
FROM python:3.11-slim
#################

WORKDIR /app

# 🎤 Install runtime dependencies for voice processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 7️⃣ Copy virtual env and uv binaries from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# 8️⃣ Copy source code and define entrypoint
COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
