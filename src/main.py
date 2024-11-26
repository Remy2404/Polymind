import os
import sys
import logging
from telegram import Update
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from database.connection import get_database, close_database_connection
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from handlers.command_handlers import CommandHandlers
from handlers.text_handlers import TextHandler
from handlers.message_handlers import MessageHandlers
from utils.telegramlog import TelegramLogger, telegram_logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Establish database connection
        self.db, self.client = get_database()
        if self.db is None:
            self.logger.error("Failed to connect to the database")
            raise ConnectionError("Failed to connect to the database")
        self.logger.info("Connected to MongoDB successfully")

        # Get tokens from .env file
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

        # Verify Gemini API key is present
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize components
        self.gemini_api = GeminiAPI()
        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger
        self.pdf_handler = None  # Initialize this properly if needed

        # Initialize handlers
        self.command_handler = CommandHandlers(self.gemini_api, self.user_data_manager)
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)
        self.message_handler = MessageHandlers(self.gemini_api, self.user_data_manager, self.telegram_logger, self.pdf_handler)

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
        self.application.add_handler(MessageHandler(filters.VOICE, self.message_handler._handle_voice_message))
        self.application.add_handler(MessageHandler(filters.Document.PDF, self.message_handler._handle_pdf_document))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler._handle_text_message))
        self.application.add_handler(MessageHandler(filters.PHOTO, self.message_handler._handle_image_message))

        # Register PDF handlers if pdf_handler exists
        if self.pdf_handler:
            self.application.add_handler(MessageHandler(filters.Document.PDF, self.message_handler._handle_pdf_document))
            self.application.add_handler(MessageHandler(filters.VOICE, self.message_handler._handle_pdf_followup))

        # Register error handler
        self.application.add_error_handler(self.message_handler._error_handler)

if __name__ == '__main__':
    main_bot = TelegramBot()
    try:
        main_bot.run()
    finally:
        main_bot.shutdown()