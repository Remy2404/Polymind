FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

FROM base AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev ffmpeg curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @mermaid-js/mermaid-cli @smithery/cli @upstash/context7-mcp @modelcontextprotocol/server-sequential-thinking && \
    npm cache clean --force && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN apt-get update && \
    apt-get purge -y gcc g++ libffi-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

FROM base

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg curl ca-certificates libnss3 libatk-bridge2.0-0 libxcomposite1 libxdamage1 \
    libxrandr2 libxss1 libasound2 libxkbcommon0 libdrm2 libgbm1 \
    libatk1.0-0 libcups2 libnspr4 fonts-dejavu-core fonts-liberation chromium && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    npm install -g @mermaid-js/mermaid-cli puppeteer @smithery/cli @upstash/context7-mcp @modelcontextprotocol/server-sequential-thinking && \
    # Verify MCP packages are installed correctly
    npx @smithery/cli --version && \
    npx @upstash/context7-mcp --version && \
    npx @modelcontextprotocol/server-sequential-thinking --version && \
    npm cache clean --force && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:/usr/local/bin:$PATH" \
    PORT=8000 \
    INSIDE_DOCKER="true"

COPY . .

# Make scripts executable
RUN chmod +x scripts/init-mcp.sh scripts/health-check-mcp.sh

# Add health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD ./scripts/health-check-mcp.sh || exit 1

EXPOSE 8000
CMD ["./scripts/init-mcp.sh", "--start-app", "uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]