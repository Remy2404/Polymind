FROM python:3.11-slim

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

# Initialize database and start bot
CMD ["sh", "-c", "python database/init_db.py && python main.py"] 