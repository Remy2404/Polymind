FROM python:alpine AS builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies for Alpine and compile wheels
RUN apk add --no-cache \
  gcc \
  python3-dev \
  musl-dev \
  libffi-dev \
  make \
  g++ \
  && \
  pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# Create optimized final image
FROM python:alpine

WORKDIR /app

# Install runtime dependencies for Alpine
# WeasyPrint and other PDF-related dependencies
RUN apk add --no-cache \
  ffmpeg \
  curl \
  cairo \
  pango \
  gdk-pixbuf \
  shared-mime-info \
  tzdata \
  # Set up locales through musl-locales for Alpine
  && apk add --no-cache musl-locales musl-locales-lang \
  && cp /usr/share/zoneinfo/UTC /etc/localtime

# Set locale environment variables for Alpine
ENV LANG=en_US.UTF-8 \
  LANGUAGE=en_US:en \
  LC_ALL=en_US.UTF-8 \
  TZ=UTC

# Copy wheels and install dependencies more efficiently
COPY --from=builder /app/wheels /app/wheels
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels /app/wheels/* && \
  rm -rf /app/wheels

# Create non-root user for better security
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
RUN mkdir -p /app/data/memory /app/data/knowledge_graph && \
  chown -R appuser:appgroup /app

# Copy application code
COPY --chown=appuser:appgroup . /app/

# Set environment variables
ENV PORT=8000 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/app

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Switch to non-root user
USER appuser

# Use a more flexible command with proper configuration
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]