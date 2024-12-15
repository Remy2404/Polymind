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
from handlers.message_handlers import MessageHandlers
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
""" global main_bot
main_bot = None

@app.on_event("startup")
async def startup_event():
    global main_bot
    main_bot = TelegramBot()
    await start_bot(main_bot)
    logger.info("Telegram bot started.")

@app.on_event("shutdown")
async def shutdown_event():
    global main_bot
    if main_bot:
        await main_bot.application.stop()
        await flux_lora_image_generator.close()
        main_bot.shutdown()
        logger.info("Telegram bot stopped and database connection closed.")
        main_bot = None

@app.post("/webhook/{token}")
async def telegram_update(token: str, request: Request):
    if token != os.getenv('TELEGRAM_BOT_TOKEN'):
        return JSONResponse(status_code=403, content={"error": "Invalid token"})
    try:
        update_data = await request.json()
        update = Update.de_json(update_data, main_bot.application.bot)
        await main_bot.application.process_update(update)
        return JSONResponse(status_code=200, content={"status": "ok"})
    except Exception as e:
        logger.error(f"An error occurred while processing a Telegram update: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)}) """

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

    def _setup_webhook_route(self):
        @self.router.post(f"/webhook/{self.token}")
        async def webhook_handler(request: Request):
            try:
                update_data = await request.json()
                update = Update.de_json(update_data, self.application.bot)
                await self.application.process_update(update)
                return JSONResponse(content={"status": "ok"}, status_code=200)
            except Exception as e:
                logger.error(f"Error processing update: {e}")
                return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

    async def setup_webhook(self):
        try:
            webhook_path = f"/webhook/{self.token}"
            webhook_url = f"{os.getenv('WEBHOOK_URL').rstrip('/')}{webhook_path}"

            await self.application.bot.delete_webhook(drop_pending_updates=True)

            webhook_config = {
                "url": webhook_url,
                "allowed_updates": ["message", "edited_message", "callback_query", "inline_query"],
                "max_connections": 100
            }

            self.logger.info(f"Setting webhook to: {webhook_url}")
            await self.application.bot.set_webhook(**webhook_config)

            webhook_info = await self.application.bot.get_webhook_info()
            self.logger.info(f"Webhook status: {webhook_info}")

        except Exception as e:
            self.logger.error(f"Webhook setup failed: {e}")
            raise

    async def process_update(self, update_data: dict):
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

    try:
        if not os.getenv('WEBHOOK_URL'):
            logger.error("WEBHOOK_URL not set in .env")
            sys.exit(1)

        port = int(os.environ.get("PORT", 8000))
        
        loop.run_until_complete(main_bot.application.initialize())
        loop.run_until_complete(main_bot.setup_webhook())
        
        if os.environ.get('DYNO'):
            uvicorn.run(
                "main:app",
                host="0.0.0.0",
                port=port,
                log_level="info",
                reload=True
            )
        else:
            loop.run_forever()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        loop.run_until_complete(main_bot.application.shutdown())
        loop.run_until_complete(flux_lora_image_generator.close())
        close_database_connection(main_bot.client)
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()