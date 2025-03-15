import os
import sys
import logging
import asyncio
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from telegram import Update
import traceback
from telegram.ext import (
    Application, 
    CommandHandler, 
    PicklePersistence,
)
from cachetools import TTLCache
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import get_database, close_database_connection
from src.services.user_data_manager import UserDataManager
from src.services.gemini_api import GeminiAPI
from src.handlers.command_handlers import CommandHandlers
from src.handlers.text_handlers import TextHandler
from src.handlers.message_handlers import MessageHandlers  # Ensure this is your custom handler
from src.utils.telegramlog import TelegramLogger, telegram_logger
from threading import Thread
from src.services.reminder_manager import ReminderManager
from src.utils.language_manager import LanguageManager 
from src.services.rate_limiter import RateLimiter
import google.generativeai as genai
from src.services.flux_lora_img import flux_lora_image_generator
import uvicorn
from src.services.document_processing import DocumentProcessor
from src.database.connection import get_database
from src.services.user_data_manager import user_data_manager
from src.services.text_to_video import text_to_video_generator

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get('/health')
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

class TelegramBot:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_cache = TTLCache(maxsize=1000, ttl=3600)  # Increased cache size for better performance
        self.user_response_cache = {}
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
            
        # Initialize database connection
        self._init_db_connection()
        
        # Create application with optimized settings
        self.application = (
            Application.builder()
            .token(self.token)
            .persistence(PicklePersistence(filepath='conversation_states.pickle'))
            .http_version('1.1')
            .get_updates_http_version('1.1')
            .connection_pool_size(32)  # Increased for better concurrency
            .build()
        )
        
        self._init_services()
        self._setup_handlers()
        
    def _init_db_connection(self):
        max_retries = 5
        retry_delay = 0.2  # Start with 200ms
        
        for attempt in range(max_retries):
            try:
                self.db, self.client = get_database()
                if self.db is None:
                    raise ConnectionError("Failed to connect to the database")
                self.logger.info("Connected to MongoDB successfully")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Database connection attempt {attempt+1} failed, retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"All database connection attempts failed: {e}")
                    raise
    
    def _init_services(self):
        try:
            # Use faster model for better response times
            model_name = "gemini-2.0-flash"
            vision_model = genai.GenerativeModel(model_name)
            rate_limiter = RateLimiter(requests_per_minute=30)  # Increased for faster throughput
            self.gemini_api = GeminiAPI(vision_model=vision_model, rate_limiter=rate_limiter)
            
            self.user_data_manager = user_data_manager(self.db)
            self.telegram_logger = telegram_logger
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            raise

        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api, 
            user_data_manager=self.user_data_manager,
            telegram_logger=self.telegram_logger,
            flux_lora_image_generator=flux_lora_image_generator,
        )
        
        self.document_processor = DocumentProcessor()
        self.message_handlers = MessageHandlers(
            self.gemini_api,
            self.user_data_manager,
            self.telegram_logger,
            self.document_processor,
            self.text_handler
        )
        self.reminder_manager = ReminderManager(self.application.bot)
        self.language_manager = LanguageManager()

    async def shutdown(self):
        """Properly shut down all services"""
        try:
            close_database_connection(self.client)
            
            if hasattr(self, 'reminder_manager') and hasattr(self.reminder_manager, 'reminder_check_task'):
                await self.reminder_manager.stop()
                
            if hasattr(self, 'application') and self.application.running:
                await self.application.stop()
                
            logger.info("Bot shutdown complete")
        except Exception as e:
            logger.error(f"Error during bot shutdown: {e}")

    def _setup_handlers(self):
        self.command_handler.register_handlers(self.application, cache=self.response_cache)
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)
        self.message_handlers.register_handlers(self.application)
        self.application.add_handler(CommandHandler("remind", self.reminder_manager.set_reminder))
        self.application.add_handler(CommandHandler("language", self.language_manager.set_language))
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))
        self.application.add_handler(CommandHandler("documents", self.command_handler.show_document_history))
        # Remove existing error handlers before adding new one
        self.application.error_handlers.clear()
        self.application.add_error_handler(self.message_handlers._error_handler)
        self.application.run_webhook = self.run_webhook
    async def setup_webhook(self):
        webhook_path = f"/webhook/{self.token}"
        webhook_url = f"{os.getenv('WEBHOOK_URL')}{webhook_path}"

        await self.application.bot.delete_webhook(drop_pending_updates=True)

        webhook_config = {
            "url": webhook_url,
            "allowed_updates": ["message", "edited_message", "callback_query", "inline_query"],
            "max_connections": 200  # Increased for better parallelism
        }

        self.logger.info(f"Setting webhook to: {webhook_url}")

        if not self.application.running:
            await self.application.initialize()
            await self.application.start()

        await self.application.bot.set_webhook(**webhook_config)

        webhook_info = await self.application.bot.get_webhook_info()
        self.logger.info(f"Webhook status: {webhook_info}")

        if not self.application.running:
            await self.application.start()

    async def process_update(self, update_data: dict):
        """Process updates received from webhook"""
        try:
            if not self.application.running:
                self.logger.info("Application not initialized yet, initializing now...")
                await self.application.initialize()
                await self.application.start()
                
            update = Update.de_json(update_data, self.application.bot)
            asyncio.create_task(self.application.process_update(update))
            
        except Exception as e:
            if os.getenv('ENVIRONMENT') == 'production':
                self.logger.error(f"Error in process_update: {str(e)}")
            else:
                self.logger.error(f"Error in process_update: {e}")
                self.logger.error(traceback.format_exc())
                
    def run_webhook(self, loop):
        @app.post(f"/webhook/{self.token}")
        async def webhook_handler(request: Request):
            try:
                update_data = await request.json()
                asyncio.create_task(self.process_update(update_data))
                return JSONResponse(content={"status": "ok"}, status_code=200)
            except Exception as e:
                self.logger.error(f"Webhook error: {str(e)}")
                return JSONResponse(content={"status": "error"}, status_code=500)

async def start_bot(webhook: TelegramBot):
    try:
        await webhook.application.initialize()
        await webhook.application.start()
        logger.info("Bot started successfully.")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def create_app(webhook: TelegramBot, loop):
    webhook.run_webhook(loop)
    return app

if __name__ == '__main__':
    main_bot = TelegramBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    if os.environ.get('DEV_SERVER') == 'uvicorn':
        loop.run_until_complete(main_bot.application.initialize())
        loop.run_until_complete(main_bot.application.start())
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            if not os.getenv('WEBHOOK_URL'):
                logger.error("WEBHOOK_URL not set in .env")
                sys.exit(1)

            loop.run_until_complete(main_bot.application.initialize())
            loop.run_until_complete(main_bot.application.start())
            logger.info("Bot application initialized and started")
            
            loop.run_until_complete(main_bot.setup_webhook())
            
            app = create_app(main_bot, loop)
            
            def run_fastapi():
                port = int(os.environ.get("PORT", 8000))
                uvicorn.run(app, host="0.0.0.0", port=port)

            fastapi_thread = Thread(target=run_fastapi)
            fastapi_thread.start()
            loop.run_forever()
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            main_bot.shutdown()
            sys.exit(1)

def get_application():
    """Create and configure the application for uvicorn"""
    os.environ["DEV_SERVER"] = "uvicorn"
    
    app = FastAPI()
    
    @app.get('/health')
    async def health_check():
        return JSONResponse(content={"status": "ok"}, status_code=200)
    
    @app.get("/")
    async def read_root():
        return {"message": "Hello, World!"}
    
    bot = TelegramBot()
    existing_loop = asyncio.get_event_loop()
    
    @app.post(f"/webhook/{bot.token}")
    async def webhook_handler(request: Request):
        try:
            update_data = await request.json()
            asyncio.create_task(bot.process_update(update_data))
            return JSONResponse(content={"status": "ok"}, status_code=200)
        except Exception as e:
            bot.logger.error(f"Webhook error: {str(e)}")
            return JSONResponse(content={"status": "error"}, status_code=500)
    
    @app.on_event("startup")
    async def startup_event():
        await bot.application.initialize()
        await bot.application.start()
        
        if os.getenv('WEBHOOK_URL'):
            await bot.setup_webhook()
        
        if hasattr(bot, 'reminder_manager'):
            await bot.reminder_manager.start()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Application shutdown initiated")
        
        try:
            if 'text_to_video_generator' in globals() and hasattr(text_to_video_generator, 'session'):
                await text_to_video_generator.close()
        except Exception as e:
            logger.error(f"Error closing text_to_video session: {e}")
        
        try:
            if 'flux_lora_image_generator' in globals() and hasattr(flux_lora_image_generator, 'session'):
                await flux_lora_image_generator.close()
        except Exception as e:
            logger.error(f"Error closing flux_lora session: {e}")
        
        try:
            await bot.shutdown()
        except Exception as e:
            logger.error(f"Error during bot shutdown: {e}")
            
    return app

app = get_application()
