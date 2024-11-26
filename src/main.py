import os, sys
import tempfile
import logging
from dotenv import load_dotenv
from telegram import Update, Message, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from pydub import AudioSegment
import speech_recognition as sr
from database.connection import get_database, close_database_connection
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from handlers.command_handlers import CommandHandlers
from handlers.text_handlers import TextHandler
from utils.telegramlog import telegram_logger
from utils.telegramlog import TelegramLogger
from handlers.message_handlers import MessageHandlers

# Load environment variables
load_dotenv()

logger = TelegramLogger()

# Update logging configuration to handle Unicode
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(sys.stdout)]
)

file_handler = logging.FileHandler('your_log_file.log', encoding='utf-8')
logging.getLogger().addHandler(file_handler)

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Example usage
try:
    # Your code here
    logger.info("This is an info message")
except Exception as e:
    logger.error(f"An error occurred: {str(e)}")

class TelegramBot:
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Establish database connection
        self.db, self.client = get_database()
        if self.db is None:
            self.logger.error("Failed to connect to the database")
            raise ConnectionError("Failed to connect to the database")
        self.logger.info("Connected to MongoDB successfully")

        # Get tokens from .env file
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')  # Changed to match your .env variable name
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

        # Verify Gemini API key is present
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize components
        self.gemini_api = GeminiAPI()
        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger
        self.pdf_handler = None  # Initialize this properly
        self.command_handler = CommandHandlers(self.user_data_manager)  # Add this
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)  # Add this
        self.message_handler = MessageHandlers(self.gemini_api, self.user_data_manager, self.pdf_handler, self.telegram_logger)

        # Initialize application
        self.application = Application.builder().token(self.token).build()
        self._setup_handlers()

    def shutdown(self):
        close_database_connection(self.client)

    def run(self) -> None:
        """Start the bot."""
        try:
            self.logger.info("Starting bot")
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
            raise

    def _setup_handlers(self):
        # Register command handlers
        self.command_handler.register_handlers(self.application)
        
        # Register text handlers
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)

        # Register message handlers
        self.application.add_handler(MessageHandlers(filters.VOICE, self.message_handler._handle_voice_message))
        self.application.add_handler(MessageHandlers(filters.Document.PDF, self.message_handler._handle_pdf_document))
        self.application.add_handler(MessageHandlers(filters.TEXT & ~filters.COMMAND, self.message_handler._handle_text_message))
        self.application.add_handler(MessageHandlers(filters.PHOTO, self.message_handler._handle_image_message))
        self.application.add_handler(MessageHandlers(filters.CommandHandler, self.message_handler._handle_command))

        # Register PDF handlers if pdf_handler exists
        if self.pdf_handler:
            self.application.add_handler(CommandHandler("pdf_info", self.pdf_handler.handle_pdf_info))
            self.application.add_handler(self.pdf_handler.get_conversation_handler())
        
        # Register history handler
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))
        
        # Register error handler
        self.application.add_error_handler(self.message_handler._error_handler)

if __name__ == '__main__':
    main_bot = TelegramBot()
    try:
        main_bot.run()
    finally:
        main_bot.shutdown()