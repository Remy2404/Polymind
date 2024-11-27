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
    MessageHandler as TeleMessageHandler, 
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

app = Flask(__name__)

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

        # Initialize Gemini API **before** using it in handlers
        self.gemini_api = GeminiAPI()

        # Initialize User Data Manager
        self.user_data_manager = UserDataManager(self.db)

        # Initialize Telegram Logger
        self.telegram_logger = telegram_logger

        # Initialize TextHandler **before** PDFHandler
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)

        # Initialize CommandHandlers with the initialized Gemini API and User Data Manager
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api, 
            user_data_manager=self.user_data_manager
        )

        # Now initialize PDFHandler with text_handler
        self.pdf_handler = PDFHandler(self.gemini_api, self.text_handler)

        # Initialize MessageHandlers after other handlers
        self.message_handlers = MessageHandlers(
            self.gemini_api,
            self.user_data_manager,
            self.pdf_handler,
            self.telegram_logger
        )

        # Initialize application
        self.application = Application.builder().token(self.token).build()
        self._setup_handlers()

    def shutdown(self):
        """Clean up resources."""
        close_database_connection(self.client)
        logger.info("Shutdown complete. Database connection closed.")

    def _setup_handlers(self):
        """Set up all message handlers."""
        # Register command handlers
        self.command_handler.register_handlers(self.application)
        
        # Register text handlers
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)

        # Register message handlers using telegram.ext.MessageHandler
        self.application.add_handler(TeleMessageHandler(filters.VOICE, self.message_handlers._handle_voice_message))
        self.application.add_handler(TeleMessageHandler(filters.Document.PDF, self.message_handlers._handle_pdf_document))
        self.application.add_handler(TeleMessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handlers._handle_pdf_followup))
        self.application.add_handler(TeleMessageHandler(filters.PHOTO, self.message_handlers._handle_image_message))

        # Register PDF handlers if pdf_handler exists
        if self.pdf_handler:
            self.application.add_handler(CommandHandler("pdf_info", self.pdf_handler.handle_pdf_info))
            self.application.add_handler(self.pdf_handler.get_conversation_handler())

        # Register history handler
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))

        # Register error handler
        self.application.add_error_handler(self.message_handlers._error_handler)

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

    async def run_polling_async(self):
        """Async method to run the bot in polling mode."""
        try:
            self.logger.info("Starting bot in polling mode")
            await self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
            raise

    def run_polling(self):
        """Start the bot in polling mode."""
        try:
            asyncio.create_task(self.run_polling_async())
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
            raise

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

def create_app(bot: TelegramBot, loop):
    """Create and configure the Flask app."""
    bot.run_webhook(loop)
    return app

if __name__ == '__main__':
    main_bot = TelegramBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    flask_app = create_app(main_bot, loop)

    def run_flask():
        """Run the Flask app."""
        port = int(os.environ.get("PORT", 8000))
        flask_app.run(host="0.0.0.0", port=port)

    try:
        mode = os.getenv('BOT_MODE', 'polling').lower()
        
        if mode == 'webhook':
            # Ensure 'WEBHOOK_URL' is set correctly in .env
            if not os.getenv('WEBHOOK_URL'):
                logger.error("WEBHOOK_URL not set in .env")
                sys.exit(1)
            
            # Initialize the bot's webhook and start bot
            loop.create_task(main_bot.setup_webhook())
            loop.create_task(start_bot(main_bot))
            
            # Start Flask in a separate thread
            flask_thread = Thread(target=run_flask)
            flask_thread.start()
            
            # Run the event loop
            loop.run_forever()
        elif mode == 'polling':
            loop.run_until_complete(start_bot(main_bot))
            main_bot.run_polling()
            loop.run_forever()
        else:
            main_bot.logger.error(f"Invalid BOT_MODE: {mode}. Use 'webhook' or 'polling'.")
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        main_bot.logger.error(f"Unhandled exception: {str(e)}")
    finally:
        main_bot.shutdown()
        loop.stop()
        loop.close()