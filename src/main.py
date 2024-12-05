import os
import sys
import logging
import os
import sys
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, 
    CommandHandler, 
    PicklePersistence
)

# Import custom modules
from database.connection import get_database, close_database_connection
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from services.rate_limiter import RateLimiter
from services.reminder_manager import ReminderManager
from handlers.command_handlers import CommandHandlers
from handlers.text_handlers import TextHandler
from handlers.message_handlers import MessageHandlers
from utils.language_manager import LanguageManager
from utils.telegramlog import telegram_logger
from utils.pdf_handler import PDFHandler
import google.generativeai as genai
#FastAPI
from fastapi import FastAPI, Request
#uvicorn
import uvicorn

# Configure logging
def setup_logging():
    """
    Set up comprehensive logging configuration.
    Logs to both console and file with rotating file handler.
    """
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bot.log', encoding='utf-8', mode='a')
        ]
    )
    
    # Optional: Add file rotation to prevent log file from growing too large
    try:
        import logging.handlers as log_handlers
        file_handler = log_handlers.RotatingFileHandler(
            'bot.log', 
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
    except ImportError:
        logging.warning("Could not set up rotating file handler")

app = FastAPI()

class TelegramBot:
    """
    Main Telegram Bot class managing bot initialization, 
    webhook setup, and core functionality.
    """
    
    def __init__(self):
        """
        Initialize the Telegram Bot with all necessary components.
        Sets up database, API connections, handlers, and persistence.
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Validate required environment variables
        self._validate_env_variables()
        
        # Establish database connection
        self._setup_database()
        
        # Initialize Gemini API
        self._setup_gemini_api()
        
        # Create bot application with persistence
        self._create_bot_application()
        
        # Initialize supporting services and handlers
        self._initialize_services()
        
        # Setup message handlers
        self._setup_message_handlers()

    def _validate_env_variables(self):
        """
        Validate and extract essential environment variables.
        Raise exceptions for missing critical configurations.
        """
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.webhook_url = os.getenv('WEBHOOK_URL')
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env")
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        
        if not self.webhook_url:
            raise ValueError("WEBHOOK_URL not found in .env")

    def _setup_database(self):
        """
        Establish connection to the database.
        Handle potential connection errors.
        """
        try:
            self.db, self.client = get_database()
            if self.db is None:
                raise ConnectionError("Failed to connect to MongoDB")
            self.logger.info("Connected to MongoDB successfully")
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise

    def _setup_gemini_api(self):
        """
        Configure Gemini API with rate limiting and model initialization.
        """
        try:
            # Configure Gemini API key
            genai.configure(api_key=self.gemini_api_key)
            
            # Initialize vision model with rate limiting
            vision_model = genai.GenerativeModel("gemini-1.5-flash")
            rate_limiter = RateLimiter(requests_per_minute=20)
            
            self.gemini_api = GeminiAPI(
                vision_model=vision_model, 
                rate_limiter=rate_limiter
            )
        except Exception as e:
            self.logger.error(f"Gemini API setup failed: {e}")
            raise

    def _create_bot_application(self):
        """
        Create Telegram bot application with persistence.
        """
        try:
            self.application = (
                Application.builder()
                .token(self.token)
                .persistence(PicklePersistence(filepath='conversation_states.pickle'))
                .build()
            )
        except Exception as e:
            self.logger.error(f"Bot application creation failed: {e}")
            raise

    def _initialize_services(self):
        """
        Initialize supporting services and managers.
        """
        # User Data Manager
        self.user_data_manager = UserDataManager(self.db)
        
        # Telegram Logger
        self.telegram_logger = telegram_logger
        
        # Text Handler
        self.text_handler = TextHandler(
            self.gemini_api, 
            self.user_data_manager, 
            self.telegram_logger
        )
        
        # PDF Handler
        self.pdf_handler = PDFHandler(
            text_handler=self.text_handler,
            telegram_logger=self.telegram_logger
        )
        
        # Command Handlers
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api, 
            user_data_manager=self.user_data_manager
        )
        
        # Message Handlers
        self.message_handlers = MessageHandlers(
            self.gemini_api,
            self.user_data_manager,
            self.telegram_logger,
            self.pdf_handler
        )
        
        # Additional Services
        self.reminder_manager = ReminderManager(self.application.bot)
        self.language_manager = LanguageManager()

    def _setup_message_handlers(self):
        """
        Register all message and command handlers.
        """
        # Register command handlers
        self.command_handler.register_handlers(self.application)

        # Register text handlers
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)

        # Register PDF handlers
        for handler in self.pdf_handler.get_handlers():
            self.application.add_handler(handler)

        # Register message handlers
        self.message_handlers.register_handlers(self.application)

        # Additional specific command handlers
        self.application.add_handler(
            CommandHandler("remind", self.reminder_manager.set_reminder)
        )
        self.application.add_handler(
            CommandHandler("language", self.language_manager.set_language)
        )
        self.application.add_handler(
            CommandHandler("history", self.text_handler.show_history)
        )

        # Error handler
        self.application.add_error_handler(
            self.message_handlers._error_handler
        )

    async def setup_webhook(self):
        """
        Configure webhook for Telegram bot.
        Ensures secure HTTPS webhook URL.
        """
        try:
            webhook_path = f"/webhook/{self.token}"
            webhook_url = f"{self.webhook_url}{webhook_path}"
            
            if not webhook_url.startswith("https://"):
                raise ValueError("WEBHOOK_URL must start with 'https://'")

            self.logger.info(f"Setting webhook to {webhook_url}")
            await self.application.initialize()
            await self.application.bot.set_webhook(url=webhook_url)
            
            self.logger.info(f"Webhook successfully set up at {webhook_url}")
            return True
        except Exception as e:
            self.logger.error(f"Webhook setup failed: {e}")
            return False

    async def process_update(self, update_data: dict):
        """
        Process incoming webhook updates.
        
        Args:
            update_data (dict): Incoming update from Telegram webhook
        """
        try:
            update = Update.de_json(update_data, self.application.bot)
            await self.application.process_update(update)
        except Exception as e:
            self.logger.error(f"Update processing error: {e}")
            raise

    def shutdown(self):
        """
        Graceful shutdown of bot resources.
        Close database connections and stop bot.
        """
        try:
            close_database_connection(self.client)
            self.logger.info("Resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

@app.post("/webhook/{token}")
async def webhook_handler(token: str, request: Request):
    bot = TelegramBot()
    update_data = await request.json()
    await bot.process_update(update_data)
    return {"status": "ok"}

def main():
    """
    Main entry point for the Telegram bot application.
    Handles bot initialization, webhook setup, and error management.
    """
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize bot
        bot = TelegramBot()
        
        # Setup webhook
        webhook_path = f"/webhook/{bot.token}"
        webhook_url = f"{bot.webhook_url}{webhook_path}"
        
        if not webhook_url.startswith("https://"):
            raise ValueError("WEBHOOK_URL must start with 'https://'")

        logger.info(f"Setting webhook to {webhook_url}")
        
        # Run FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.critical(f"Failed to initialize bot: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()