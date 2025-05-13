FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
COPY requirements.txt .
RUN apt-get update && \
  apt-get install -y --no-install-recommends gcc libffi-dev g++ ffmpeg && \
  pip install --no-cache-dir -r requirements.txt && \
  apt-get remove -y gcc g++ libffi-dev && \
  apt-get autoremove -y && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]