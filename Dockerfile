# syntax=docker/dockerfile:1.4

# Base stage: minimal image for runtime
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

# Builder stage: installs build-time deps, builds everything
FROM base AS builder

# Install build tools + node + npm packages, cleanup in one RUN
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc g++ libffi-dev ffmpeg curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @mermaid-js/mermaid-cli @smithery/cli @upstash/context7-mcp @modelcontextprotocol/server-sequential-thinking && \
    apt-get purge -y gcc g++ libffi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy lock files first for caching
COPY pyproject.toml uv.lock* ./

# Use cache mount for uv cache to speed up builds
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy rest of source
COPY . .

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Final stage: only runtime stuff
FROM base AS final

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates libnss3 libatk-bridge2.0-0 libxcomposite1 libxdamage1 \
      libxrandr2 libxss1 libasound2 libxkbcommon0 libdrm2 libgbm1 \
      libatk1.0-0 libcups2 libnspr4 fonts-dejavu-core fonts-liberation chromium && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @mermaid-js/mermaid-cli puppeteer @smithery/cli @upstash/context7-mcp @modelcontextprotocol/server-sequential-thinking && \
    npm cache clean --force && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Bring over built artifacts
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:/usr/local/bin:$PATH" \
    PORT=8000 \
    INSIDE_DOCKER="true" 

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
