FROM python:3.11-slim
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV UV_CACHE_DIR=/tmp/uv-cache

# Install system dependencies and uv
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev g++ ffmpeg curl && \
    pip install --no-cache-dir uv && \
    apt-get remove -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev && \
    uv cache clean

# Remove build dependencies
RUN apt-get update && \
    apt-get remove -y gcc g++ libffi-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/uv-cache

# Copy application code
COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]