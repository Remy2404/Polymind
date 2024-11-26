import os
import sys
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from telegram import Update
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
from handlers.message_handlers import CustomMessageHandler  # Ensure this is your custom handler
from utils.telegramlog import TelegramLogger, telegram_logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.FileHandler('bot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

app = FastAPI()

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
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize components
        self.gemini_api = GeminiAPI()
        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger
        self.pdf_handler = None  # Initialize this properly if needed

        # Initialize handlers
        self.command_handler = CommandHandlers(self.gemini_api, self.user_data_manager)
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)
        self.message_handler = CustomMessageHandler(
            self.gemini_api, self.user_data_manager, self.telegram_logger, self.pdf_handler
        )

        # Initialize application
        self.application = Application.builder().token(self.token).build()
        self._setup_handlers()

    def shutdown(self):
        close_database_connection(self.client)

    def _setup_handlers(self):
        # Register command handlers
        self.command_handler.register_handlers(self.application)
        
        # Register text handlers
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)

        # Register message handlers using telegram.ext.MessageHandler
        self.application.add_handler(TeleMessageHandler(filters.VOICE, self.message_handler._handle_voice_message))
        self.application.add_handler(TeleMessageHandler(filters.Document.PDF, self.message_handler._handle_pdf_document))
        self.application.add_handler(TeleMessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler._handle_text_message))
        self.application.add_handler(TeleMessageHandler(filters.PHOTO, self.message_handler._handle_image_message))

        # Register PDF handlers if pdf_handler exists
        if self.pdf_handler:
            self.application.add_handler(CommandHandler("pdf_info", self.pdf_handler.get_pdf_info))
            self.application.add_handler(self.pdf_handler.get_conversation_handler())

        # Register error handler
        self.application.add_error_handler(self.message_handler._error_handler)

    async def setup_webhook(self):
        """Set up webhook for the bot"""
        webhook_path = f"/webhook/{self.token}"
        webhook_url = f"{os.getenv('WEBHOOK_URL')}{webhook_path}"
        
        self.logger.info(f"Setting webhook to {webhook_url}")
        await self.application.initialize()
        await self.application.bot.set_webhook(url=webhook_url)
        self.logger.info(f"Webhook set up at {webhook_url}")

    async def process_update(self, update_data: dict):
        """Process updates received from webhook"""
        update = Update.de_json(update_data, self.application.bot)
        await self.application.process_update(update)

    def run_webhook(self):
        """Start the bot in webhook mode"""
        try:
            self.logger.info("Starting bot in webhook mode")
            
            @app.post(f"/webhook/{self.token}")
            async def webhook_handler(request: Request):
                update_data = await request.json()
                await self.process_update(update_data)
                return JSONResponse(content={"status": "ok"})
        except Exception as e:
            self.logger.error(f"Error in webhook setup: {str(e)}")

    def run_polling(self):
        """Start the bot in polling mode"""
        try:
            self.logger.info("Starting bot in polling mode")
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
            raise

if __name__ == '__main__':
    main_bot = TelegramBot()
    try:
        mode = os.getenv('BOT_MODE', 'polling').lower()
        
        if mode == 'webhook':
            asyncio.run(main_bot.setup_webhook())
            
            # Use the PORT environment variable provided by Vercel
            port = int(os.environ.get("PORT", 8000))
            import uvicorn
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=port
            )
        elif mode == 'polling':
            main_bot.run_polling()
        else:
            main_bot.logger.error(f"Invalid BOT_MODE: {mode}. Use 'webhook' or 'polling'.")
    except Exception as e:
        main_bot.logger.error(f"Unhandled exception: {str(e)}")
    finally:
        main_bot.shutdown()