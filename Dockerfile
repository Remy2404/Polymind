FROM python:3.11.12-slim-bookworm AS builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies and compile wheels
RUN apt-get update && \
  apt-get install -y --no-install-recommends gcc python3-dev && \
  pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt && \
  rm -rf /var/lib/apt/lists/*


FROM python:3.11.12-slim-bookworm

WORKDIR /app

# Install runtime dependencies including curl for healthcheck and locales for Unicode support
RUN apt-get update && \
  apt-get install -y --no-install-recommends ffmpeg curl locales && \
  sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
  sed -i -e 's/# km_KH UTF-8/km_KH UTF-8/' /etc/locale.gen && \
  dpkg-reconfigure --frontend=noninteractive locales && \
  update-locale LANG=en_US.UTF-8 && \
  rm -rf /var/lib/apt/lists/*

# Set locale environment variables
ENV LANG=en_US.UTF-8 \
  LANGUAGE=en_US:en \
  LC_ALL=en_US.UTF-8

# Copy wheels and install dependencies
COPY --from=builder /app/wheels /app/wheels
RUN pip install --no-cache-dir --no-index --find-links=/app/wheels /app/wheels/* && \
  rm -rf /app/wheels

# Copy application code
COPY . /app/

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Use a simpler command without conditionals
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]