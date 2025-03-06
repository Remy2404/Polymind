FROM python:3.11-slim-bookworm AS builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies and compile wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Create final image
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install runtime dependencies including curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install wheels - simplified approach
COPY --from=builder /app/wheels /app/wheels
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels /* && \
    rm -rf /app/wheels

# Copy application code - exclude unnecessary files
COPY src/ /app/src/
COPY app.py requirements.txt .env* ./

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Use a simpler command without conditionals
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]