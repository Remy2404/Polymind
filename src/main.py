import os
import sys
import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
import uvicorn


load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get('/health')
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)

class TelegramBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db, self.client = get_database()
        if self.db is None:
            self.logger.error("Failed to connect to the database")
            raise ConnectionError("Failed to connect to the database")
        self.logger.info("Connected to MongoDB successfully")

        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        self.application = (
            Application.builder()
            .token(self.token)
            .persistence(PicklePersistence(filepath='conversation_states.pickle'))
            .build()
        )

        vision_model = genai.GenerativeModel("gemini-1.5-flash")
        rate_limiter = RateLimiter(requests_per_minute=10)
        self.gemini_api = GeminiAPI(vision_model=vision_model, rate_limiter=rate_limiter)

        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api, 
            user_data_manager=self.user_data_manager,
            telegram_logger=self.telegram_logger,
            flux_lora_image_generator=flux_lora_image_generator,
        )
        self.pdf_handler = PDFHandler(
            text_handler=self.text_handler,
            telegram_logger=self.telegram_logger
        )
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
        close_database_connection(self.client)
        logger.info("Shutdown complete. Database connection closed.")

    def _setup_handlers(self):
        self.command_handler.register_handlers(self.application)
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)
        for handler in self.pdf_handler.get_handlers():
            self.application.add_handler(handler)
        self.message_handlers.register_handlers(self.application)
        self.application.add_handler(CommandHandler("remind", self.reminder_manager.set_reminder))
        self.application.add_handler(CommandHandler("language", self.language_manager.set_language))
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))
        self.application.add_error_handler(self.message_handlers._error_handler)
        self.application.run_webhook = self.run_webhook

    async def setup_webhook(self):
        """Set up webhook with proper update processing."""
        webhook_path = f"/webhook/{self.token}"
        webhook_url = f"{os.getenv('WEBHOOK_URL')}{webhook_path}"

        # First, delete existing webhook and get pending updates
        await self.application.bot.delete_webhook(drop_pending_updates=True)

        webhook_config = {
            "url": webhook_url,
            "allowed_updates": ["message", "edited_message", "callback_query", "inline_query"],
            "max_connections": 1000
        }

        self.logger.info(f"Setting webhook to: {webhook_url}")

        if not self.application.running:
            await self.application.initialize()

        # Set up webhook with new configuration
        await self.application.bot.set_webhook(**webhook_config)

        # Log webhook info for verification
        webhook_info = await self.application.bot.get_webhook_info()
        self.logger.info(f"Webhook status: {webhook_info}")

        # Only start the application if it's not already running
        if not self.application.running:
            await self.application.start()
        else:
            self.logger.info("Application is already running. Skipping start.")

    async def process_update(self, update_data: dict):
        """Process updates received from webhook."""
        try:
            update = Update.de_json(update_data, self.application.bot)
            self.logger.debug(f"Received update: {update}")
            await self.application.process_update(update)
            self.logger.debug("Processed update successfully.")
        except Exception as e:
            self.logger.error(f"Error in process_update: {e}")
            self.logger.error(traceback.format_exc())

    def run_webhook(self, loop):
        @app.post(f"/webhook/{self.token}")     
        async def webhook_handler(request: Request):
            try:
                update_data = await request.json()
                await self.process_update(update_data)
                return JSONResponse(content={"status": "ok", "method": "webhook"}, status_code=200)
            except Exception as e:
                self.logger.error(f"Update processing error: {e}")
                self.logger.error(traceback.format_exc())
                return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

async def start_bot(bot: TelegramBot):
    try:
        await bot.application.initialize()
        await bot.application.start()
        logger.info("Bot started successfully.")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def create_app(bot: TelegramBot, loop):
    bot.run_webhook(loop)
    return app

if __name__ == '__main__':
    main_bot = TelegramBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = create_app(main_bot, loop)

    if os.environ.get('DEV_SERVER') == 'uvicorn':
        port = int(os.environ.get("PORT", 8000))  # Add this line
        uvicorn.run(app, host="0.0.0.0", port=port)  # Modified line
    else:
        try:
            webhook_url = os.getenv('WEBHOOK_URL')
            if not webhook_url:
                logger.error("WEBHOOK_URL not set in .env")
                sys.exit(1)

            loop.create_task(main_bot.setup_webhook())
            loop.create_task(start_bot(main_bot))

            def run_fastapi():
                port = int(os.environ.get("PORT", 8000))
                uvicorn.run(app, host="0.0.0.0", port=port)

            fastapi_thread = Thread(target=run_fastapi)
            fastapi_thread.start()
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            main_bot.logger.error(f"Unhandled exception: {str(e)}")
        finally:
            close_database_connection(main_bot.client)
            loop.stop()
            loop.close()