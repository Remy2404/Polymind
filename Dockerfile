FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/app/.cache/uv

WORKDIR /app

# Builder stage: only build Python dependencies
FROM base AS builder

# Install only build-time dependencies needed for Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy UV binaries first
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy lock files for cache
COPY pyproject.toml uv.lock* ./

# Build dependencies with cache mount
RUN --mount=type=cache,id=uv-cache,target=/app/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy source and install project
COPY . .

RUN --mount=type=cache,id=uv-cache,target=/app/.cache/uv \
    uv sync --locked --no-dev

# ===== Final stage: runtime only =====
FROM base AS final

# Create non-root user and set up home directory
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser && \
    mkdir -p /home/appuser/.config /home/appuser/.local/share /home/appuser/.cache && \
    chown -R appuser:appuser /home/appuser

# Install runtime dependencies in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install npm packages
RUN npm install -g \
      @mermaid-js/mermaid-cli \
      @smithery/cli \
      @upstash/context7-mcp \
      @modelcontextprotocol/server-sequential-thinking \
      @modelcontextprotocol/inspector && \
    npm cache clean --force

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      fonts-liberation \
      fonts-dejavu \
      fonts-dejavu-core \
      fonts-dejavu-extra && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy built artifacts from builder
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Set environment variables
ENV PATH="/app/.venv/bin:/usr/local/bin:$PATH" \
    PORT=8000 \
    INSIDE_DOCKER="true"

RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]