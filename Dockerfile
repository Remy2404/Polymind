FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/root/.cache/uv

WORKDIR /app

# Builder stage: installs build-time deps, builds everything
FROM base AS builder

# Install build tools + node + npm packages with cache mounts for speed
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc g++ libffi-dev ffmpeg curl ca-certificates && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get purge -y gcc g++ libffi-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install npm packages globally with cache mount for faster rebuilds
RUN --mount=type=cache,target=/root/.npm \
    npm install -g @mermaid-js/mermaid-cli @smithery/cli @upstash/context7-mcp \
    @modelcontextprotocol/server-sequential-thinking @modelcontextprotocol/inspector

# Copy uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy lock files first for caching
COPY pyproject.toml uv.lock* ./

# Use cache mount for uv cache to speed up builds
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Pre-install commonly used uvx tools for MCP
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uvx --help > /dev/null 2>&1 && \
    uvx mcp-server-fetch --help > /dev/null 2>&1 || true

# Copy rest of source
COPY . .

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Final stage: only runtime stuff
FROM base AS final

# Install only runtime dependencies (NO Node.js - we copy from builder)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates libnss3 libatk-bridge2.0-0 libxcomposite1 libxdamage1 \
      libxrandr2 libxss1 libasound2 libxkbcommon0 libdrm2 libgbm1 \
      libatk1.0-0 libcups2 libnspr4 chromium && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy Node.js and npm packages from builder instead of reinstalling
COPY --from=builder /usr/bin/node /usr/bin/node
COPY --from=builder /usr/bin/npm /usr/bin/npm
COPY --from=builder /usr/bin/npx /usr/bin/npx
COPY --from=builder /usr/lib/node_modules /usr/lib/node_modules
COPY --from=builder /usr/include/node /usr/include/node

# Install .NET runtime for Spire.Doc support
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -fsSL https://packages.microsoft.com/config/debian/11/packages-microsoft-prod.deb -o packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends libicu && \
    apt-get install -y --no-install-recommends dotnet-runtime-8.0 fonts-liberation fonts-dejavu fonts-dejavu-core fonts-dejavu-extra && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Bring over built artifacts
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx
COPY --from=builder /app/.venv /app/.venv

# Create non-root user for security
RUN groupadd -r appuser && useradd --no-log-init -r -g appuser appuser && \
    chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:/usr/local/bin:$PATH" \
    PORT=8000 \
    INSIDE_DOCKER="true" \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
    DOTNET_NOLOGO=1 

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
