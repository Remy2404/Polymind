import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Telegram Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = logging.INFO

GENERATION_CONFIG = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

# Bot Messages
WELCOME_MESSAGE = """
Welcome to Gemini AI Bot! 🤖✨
I'm here to help you with:
- Text generation
- Code assistance
- Image analysis

Use /help to see available commands.
"""

HELP_MESSAGE = """
Available commands:
/start - Start the bot
/help - Show this help message
/settings - Configure your preferences
/code - Generate code (follow with language and prompt)
/feedback - Send feedback
"""