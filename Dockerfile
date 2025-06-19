# syntax=docker/dockerfile:1.4

#################
# Builder stage #
#################
FROM python:3.11-slim-bookworm AS builder

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

# 5️⃣ Download Faster-Whisper model (large-v3)
# This step pre-downloads the model to the HF_HOME cache directory set above.
# It runs after dependencies are installed.
# Use the compute_type that matches your production environment (int8_float16 from logs)
RUN python -c "from faster_whisper import WhisperModel; import logging; logging.basicConfig(level=logging.INFO); print('Downloading large-v3 model...'); model = WhisperModel('large-v3', device='cpu', compute_type='int8_float16'); print('Download complete.')"

# 6️⃣ Copy application code and install the project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# 6️⃣ Don't remove FFmpeg - it's needed for voice processing in production
RUN apt-get remove -y gcc g++ libffi-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# 7️⃣ Don't remove FFmpeg - it's needed for voice processing in production
#################
# Final stage   #
#################
FROM python:3.11-slim-bookworm

WORKDIR /app

# 🎤 Install runtime dependencies for voice processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 8️⃣ Copy virtual env, uv binaries, and the Hugging Face cache from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

ENV HF_HOME=/app/.cache/huggingface
# 9️⃣ Copy source code and define entrypoint
COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
