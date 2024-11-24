import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    ConversationHandler
)
from database.connection import get_database, close_database_connection
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from utils.telegramlog import telegram_logger
from utils.pdf_handler import PDFHandler
from handlers.text_handlers import TextHandler
from handlers.command_handlers import CommandHandlers
from handlers.message_handlers import MessageHandlers
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)

file_handler = logging.FileHandler('telegram_bot.log', encoding='utf-8')
logging.getLogger().addHandler(file_handler)

app = FastAPI()

class TelegramBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Database setup
        self.db, self.client = get_database()
        if self.db is None:
            self.logger.error("Failed to connect to the database")
            raise ConnectionError("Failed to connect to the database")
        self.logger.info("Connected to MongoDB successfully")

        # Environment variables
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.webhook_url = os.getenv('WEBHOOK_URL')
        self.webhook_path = f"/webhook/{self.token}"
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize components
        self.gemini_api = GeminiAPI()
        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger
        
        # Initialize handlers
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api, 
            user_data_manager=self.user_data_manager
        )
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)
        self.pdf_handler = PDFHandler(self.gemini_api, self.text_handler)
        self.message_handlers = MessageHandlers(
            self.gemini_api,
            self.user_data_manager,
            self.pdf_handler,
            self.telegram_logger
        )

        # Initialize application
        self.application = Application.builder().token(self.token).build()
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up all message handlers"""
        # Register command handlers
        self.command_handler.register_handlers(self.application)
        
        # Register text handlers
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)

        # Register message handlers
        self.application.add_handler(MessageHandler(filters.VOICE, self.message_handlers._handle_voice_message))
        self.application.add_handler(MessageHandler(filters.Document.PDF, self.message_handlers._handle_pdf_document))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handlers._handle_pdf_followup))
        
        # Register PDF handlers
        self.application.add_handler(CommandHandler("pdf_info", self.pdf_handler.handle_pdf_info))
        self.application.add_handler(self.pdf_handler.get_conversation_handler())
        
        # Register history handler
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))
        
        # Register error handler
        self.application.add_error_handler(self.message_handlers._error_handler)

    async def setup_webhook(self):
        """Set up webhook for the bot"""
        webhook_url = f"{self.webhook_url}{self.webhook_path}"
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
            
            @app.post(self.webhook_path)
            async def webhook_handler(request: Request):
                update_data = await request.json()
                await self.process_update(update_data)
                return JSONResponse(content={"status": "ok"})

            @app.get("/")
            async def root():
                return {"status": "Bot is running"}

            import uvicorn
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000
            )
        except Exception as e:
            self.logger.error(f"Fatal error in webhook mode: {str(e)}")
            raise

    def run_polling(self):
        """Start the bot in polling mode"""
        try:
            self.logger.info("Starting bot in polling mode")
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            self.logger.error(f"Fatal error in polling mode: {str(e)}")
            raise

    def shutdown(self):
        """Clean up resources"""
        close_database_connection(self.client)

if __name__ == '__main__':
    main_bot = TelegramBot()
    try:
        # Determine the mode (webhook or polling) based on environment variable
        mode = os.getenv('BOT_MODE', 'polling').lower()
        
        if mode == 'webhook':
            asyncio.run(main_bot.setup_webhook())
            
            # Modify this part to use the PORT environment variable
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