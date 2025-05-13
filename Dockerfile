# syntax=docker/dockerfile:1.4
# Enable BuildKit features for faster builds

# Stage 1: Build dependencies
FROM python:3.11-alpine AS builder

# Set work directory
WORKDIR /app

# Set build arguments and environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies in a single layer
RUN apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    libffi-dev \
    make \
    python3-dev \
    cairo-dev \
    pango-dev \
    gdk-pixbuf-dev \
    jpeg-dev \
    zlib-dev

# Copy and install dependencies separately to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-alpine AS runtime

# Set work directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install runtime dependencies only in a single layer
RUN apk add --no-cache \
    ffmpeg \
    libstdc++ \
    libgomp \
    libffi \
    cairo \
    pango \
    gdk-pixbuf \
    jpeg \
    tzdata

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Create a non-root user and directories in one layer
RUN addgroup -S appgroup && \
    adduser -S appuser -G appgroup && \
    mkdir -p /app/data/memory /app/logs && \
    chown -R appuser:appgroup /app

# Copy application files - ordered by change frequency (least to most frequent)
# Copy static assets first
COPY --chown=appuser:appgroup assets/ assets/

# Copy application code
COPY --chown=appuser:appgroup src/ src/

# Copy main application file (changes less frequently than individual modules)
COPY --chown=appuser:appgroup app.py .

# Switch to non-root user
USER appuser

# Explicitly expose port 8000
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]