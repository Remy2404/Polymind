FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHON_PATH=/app

# Keep WORKDIR at the app root since app.py is there
# Instead of: WORKDIR /app/src

# Change the CMD to use app.py in the root directory
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]