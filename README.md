# Telegram Gemini Bot

A powerful Telegram bot leveraging Google's Gemini and DeepSeek AI & OpenRouter for conversational assistance, media processing, and document management.

<div align="center">
  <img src="assets/templates/Project_report_group5.png" alt="Telegram Gemini Bot" width="200" />
</div>

## Table of Contents
- [Telegram Gemini Bot](#telegram-gemini-bot)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Commands](#commands)
  - [Project Structure](#project-structure)
  - [Docker Deployment](#docker-deployment)
  - [Contributing](#contributing)
  - [License](#license)

## Features
- AI-powered conversations with persistent memory  
- Image & video generation and analysis  
- PDF and DOCX processing with semantic search  
- Voice transcription and multilingual support  
- Modular design supporting multiple AI models  
- Robust error handling: rate limiting, caching, retries  
- Secure credential management via environment variables  

## Prerequisites
- Python 3.11+  
- MongoDB instance  
- API keys: Telegram Bot, Google Gemini, OpenRouter, Together AI, etc.  

## Installation
```bash
git clone https://github.com/Remy2404/Telegram-Gemini-Bot.git
cd Telegram-Gemini-Bot
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -r requirements.txt
```

## Configuration
Create a `.env` in the project root:
```env
DATABASE_URL=your_mongo_uri
TELEGRAM_BOT_TOKEN=your_telegram_token
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
TEXT_TO_IMAGE_API_KEY=...
TEXT_TO_VIDEO_API_KEY=...
TOGETHER_API_KEY=...
BOT_MODE=webhook
WEBHOOK_URL=your_webhook_url
PORT=8000
DEV_MODE=true
LOGS_DIR=logs
```

## Usage
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Commands
| Command           | Description                          |
|-------------------|--------------------------------------|
| `/start`          | Initialize the bot                   |
| `/help`           | List available commands              |
| `/generate_image` | Generate an image from a prompt      |
| `/generate_video` | Create a video from a prompt         |
| `/reset`          | Clear conversation history           |
| `/stats`          | Show usage statistics                |
| `/switchmodel`    | Switch AI model                      |
| `/exportdoc`      | Export chat to PDF/DOCX              |

## Project Structure
```
src/
├── database/        # Database schemas and connections
├── handlers/        # Message & callback handlers
├── services/        # AI model wrappers & business logic
├── utils/           # Logging, config, utilities
└── main.py          # Entry point
app.py               # FastAPI server setup
```

## Docker Deployment
Build and run:
```bash
docker build -t telegram-gemini-bot .
docker run -d -p 8000:8000 --env-file .env telegram-gemini-bot
```
With Docker Compose:
```bash
docker-compose up -d
```

## Contributing
Contributions are welcome. Fork the repo, create a feature branch, commit your changes, and open a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
