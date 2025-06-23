# syntax=docker/dockerfile:1.4

#################
# Builder stage #
#################
FROM python:3.11-slim AS builder

# 1️⃣ Get uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# 2️⃣ Environment setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

# 3️⃣ Install dependencies: dev + Mermaid requirements
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev ffmpeg curl \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \
        libgbm1 libxss1 libasound2 libatspi2.0-0 \
        fonts-freefont-ttf fonts-liberation fonts-noto-color-emoji fonts-dejavu && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @mermaid-js/mermaid-cli && \
    rm -rf /var/lib/apt/lists/*

# 4️⃣ Python deps
COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# 5️⃣ Clean build deps but keep ffmpeg
RUN apt-get purge -y gcc g++ libffi-dev && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

#################
# Runtime stage #
#################
FROM python:3.11-slim

WORKDIR /app

# 🎤 Runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg curl \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
        libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 \
        libgbm1 libxss1 libasound2 libatspi2.0-0 \
        fonts-freefont-ttf fonts-liberation fonts-noto-color-emoji fonts-dejavu && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @mermaid-js/mermaid-cli && \
    rm -rf /var/lib/apt/lists/*

# 🐍 Bring in venv + uv
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
