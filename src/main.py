import os
import sys
import logging
import asyncio
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from telegram import Update
import traceback
from telegram.ext import (
    Application, 
    CommandHandler, 
    PicklePersistence,
    filters
)
from database.connection import get_database, close_database_connection
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from handlers.command_handlers import CommandHandlers
from handlers.text_handlers import TextHandler
from handlers.message_handlers import MessageHandlers  # Ensure this is your custom handler
from utils.telegramlog import TelegramLogger, telegram_logger
from utils.pdf_handler import PDFHandler
from threading import Thread
from services.reminder_manager import ReminderManager
from utils.language_manager import LanguageManager 
from services.rate_limiter import RateLimiter
import google.generativeai as genai
from services.flux_lora_img import flux_lora_image_generator
# Add these imports
from waitress import serve  # or gunicorn for Linux
from gevent import monkey 
monkey.patch_all() # or eventlet for Linux

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)



class TelegramBot:
    """Main class for the Telegram Bot."""

    def __init__(self):
        """Initialize the TelegramBot instance."""
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
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        # Initialize application with persistence
        self.application = (
            Application.builder()
            .token(self.token)
            .persistence(PicklePersistence(filepath='conversation_states.pickle'))
            .build()
        )

        # Initialize Gemini API **before** using it in handlers
        vision_model = genai.GenerativeModel("gemini-1.5-flash")
        rate_limiter = RateLimiter(requests_per_minute=10)
        self.gemini_api = GeminiAPI(vision_model=vision_model, rate_limiter=rate_limiter)



        # Initialize User Data Manager
        self.user_data_manager = UserDataManager(self.db)
        # Initialize Telegram Logger
        self.telegram_logger = telegram_logger

        # Initialize TextHandler **before** PDFHandler
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)

        # Initialize CommandHandlers with the initialized Gemini API and User Data Manager
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api, 
            user_data_manager=self.user_data_manager,
            telegram_logger=self.telegram_logger,
            flux_lora_image_generator=flux_lora_image_generator,
        )

        # Now initialize PDFHandler with text_handler
        self.pdf_handler = PDFHandler(
            text_handler=self.text_handler,
            telegram_logger=self.telegram_logger
        )

        # Initialize MessageHandlers after other handlers
        self.message_handlers = MessageHandlers(
            self.gemini_api,
            self.user_data_manager,
            self.telegram_logger,
            self.pdf_handler
        )

        self.reminder_manager = ReminderManager(self.application.bot)
        self.language_manager = LanguageManager()

        self._setup_handlers()

    def shutdown(self):
        """Clean up resources."""
        close_database_connection(self.client)
        logger.info("Shutdown complete. Database connection closed.")

    def _setup_handlers(self):
        """Set up all message handlers."""
       # Register command handlers
        self.command_handler.register_handlers(self.application)

        # Register handlers from TextHandler
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)

        # Register handlers from PDFHandler
        for handler in self.pdf_handler.get_handlers():
            self.application.add_handler(handler)

        # Register handlers from MessageHandlers
        self.message_handlers.register_handlers(self.application)

        # Register specific command handlers
        self.application.add_handler(CommandHandler("remind", self.reminder_manager.set_reminder))
        self.application.add_handler(CommandHandler("language", self.language_manager.set_language))
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))

        self.application.add_error_handler(self.message_handlers._error_handler)

        self.application.run_webhook = self.run_webhook
    async def setup_webhook(self):
        """Set up webhook for the bot."""
        webhook_path = f"/webhook/{self.token}"
        webhook_url = f"{os.getenv('WEBHOOK_URL')}{webhook_path}"

        if not webhook_url.startswith("https://"):
            self.logger.error("WEBHOOK_URL must start with 'https://'")
            raise ValueError("Invalid WEBHOOK_URL format.")

        self.logger.info(f"Setting webhook to {webhook_url}")
        await self.application.initialize()
        await self.application.bot.set_webhook(url=webhook_url)
        self.logger.info(f"Webhook set up at {webhook_url}")

    async def process_update(self, update_data: dict):
        """Process updates received from webhook."""
        try:
            update = Update.de_json(update_data, self.application.bot)
            self.logger.debug(f"Processed Update object: {update}")
            await self.application.process_update(update)
            self.logger.debug("Awaited process_update successfully.")
        except Exception as e:
            self.logger.error(f"Error in process_update: {e}")
            raise

    def run_webhook(self, loop):
        """Start the bot in webhook mode."""
        try:
            self.logger.info("Starting bot in webhook mode")

            @app.route(f"/webhook/{self.token}", methods=['POST'])
            def webhook_handler():
                try:
                    update_data = request.get_json(force=True)
                    asyncio.run_coroutine_threadsafe(self.process_update(update_data), loop)
                    return jsonify({"status": "ok"}), 200
                except Exception as e:
                    self.logger.error(f"Webhook handler error: {e}")
                    return jsonify({"status": "error", "message": str(e)}), 500
        except Exception as e:
            self.logger.error(f"Error in webhook setup: {str(e)}")

async def start_bot(bot: TelegramBot):
    """Initialize and start the Telegram bot."""
    try:
        await bot.application.initialize()
        await bot.application.start()
        logger.info("Bot started successfully.")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def create_app():
    """Create and configure the Flask app."""
    app = Flask(__name__)
    main_bot = TelegramBot()
    
    @app.route('/')
    def index():
        return jsonify({"status": "ok", "message": "Service is running"}), 200

    @app.route(f"/webhook/{main_bot.token}", methods=['POST'])
    def webhook_handler():
        try:
            update_data = request.get_json(force=True)
            asyncio.run(main_bot.process_update(update_data))
            return jsonify({"status": "ok"}), 200
        except Exception as e:
            main_bot.logger.error(f"Webhook handler error: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.before_first_request
    def setup_webhook():
        asyncio.run(main_bot.setup_webhook())
        asyncio.run(start_bot(main_bot))

    return app

app = create_app()

if __name__ == '__main__':
    main_bot = TelegramBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = create_app(main_bot, loop)  # Rename flask_app to app

    # For development server
    if os.environ.get('DEV_SERVER') == 'uvicorn':
        # Let uvicorn handle the event loop
        app.run(host="0.0.0.0", port=8000)
    else:
        try:
            # Original production code
            if not os.getenv('WEBHOOK_URL'):
                logger.error("WEBHOOK_URL not set in .env")
                sys.exit(1)

            loop.create_task(main_bot.setup_webhook())
            loop.create_task(start_bot(main_bot))

            def run_flask():
                port = int(os.environ.get("PORT", 8000))
                serve(app, host="0.0.0.0", port=port)

            flask_thread = Thread(target=run_flask)
            flask_thread.start()
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            main_bot.logger.error(f"Unhandled exception: {str(e)}")
        finally:
            close_database_connection(main_bot.client)
            loop.stop()
            loop.close()