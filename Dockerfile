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

<<<<<<< HEAD
# Install runtime dependencies including curl for healthcheck and locales for Unicode support
=======
# Install runtime dependencies
>>>>>>> b6ce3f4bf02c0e1b6e292a535d84a30b2f904dff
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
<<<<<<< HEAD

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ || exit 1

# Use a simpler command without conditionals
=======
ENV DEV_SERVER=uvicorn
>>>>>>> b6ce3f4bf02c0e1b6e292a535d84a30b2f904dff
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]