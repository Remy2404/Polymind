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
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import get_database, close_database_connection
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from handlers.command_handlers import CommandHandlers
from handlers.text_handlers import TextHandler
from handlers.message_handlers import MessageHandlers  # Ensure this is your custom handler
from utils.telegramlog import TelegramLogger, telegram_logger
from threading import Thread
from services.reminder_manager import ReminderManager
from utils.language_manager import LanguageManager 
from services.rate_limiter import RateLimiter
import google.generativeai as genai
from services.flux_lora_img import flux_lora_image_generator
import uvicorn
from services.document_processing import DocumentProcessor
from database.connection import get_database
from services.user_data_manager import user_data_manager
from services.text_to_video import text_to_video_generator

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
@app.get('/')
async def root():
    return {"message": "Hello World"}

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

        vision_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
        rate_limiter = RateLimiter(requests_per_minute=10)
        self.gemini_api = GeminiAPI(vision_model=vision_model, rate_limiter=rate_limiter)

        self.db, self.client = get_database()
        self.user_data_manager = user_data_manager(self.db)
        self.telegram_logger = telegram_logger
        self.text_handler = TextHandler(self.gemini_api, self.user_data_manager)
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api, 
            user_data_manager=self.user_data_manager,
            telegram_logger=self.telegram_logger,
            flux_lora_image_generator=flux_lora_image_generator,
        )
        # Initialize DocumentProcessor
        self.document_processor = DocumentProcessor()
        # Update MessageHandlers initialization with document_processor
        self.message_handlers = MessageHandlers(
            self.gemini_api,
            self.user_data_manager,
            self.telegram_logger,
            self.document_processor,
            self.text_handler
        )
        self.reminder_manager = ReminderManager(self.application.bot)
        self.language_manager = LanguageManager()
        self._setup_handlers()

    async def shutdown(self):
        """Properly clean up resources on shutdown"""
        try:
            # Cancel any pending tasks first
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if tasks:
                self.logger.info(f"Cancelling {len(tasks)} pending tasks...")
                # Give tasks a chance to complete gracefully
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Perform specific cleanup operations
            if hasattr(self, 'application') and self.application.running:
                self.logger.info("Stopping PTB application...")
                await self.application.stop()
                
            if hasattr(self, 'reminder_manager') and hasattr(self.reminder_manager, 'reminder_check_task'):
                self.logger.info("Stopping reminder manager...")
                await self.reminder_manager.stop()
                
            # Close external service connections if they exist
            services_to_close = [
                ('text_to_video_generator', 'session'),
                ('flux_lora_image_generator', 'session'),
                ('gemini_api', 'session')
            ]
            
            for service_name, attr_name in services_to_close:
                if hasattr(self, service_name) and hasattr(getattr(self, service_name), attr_name):
                    service = getattr(self, service_name)
                    if hasattr(service, 'close') and callable(service.close):
                        self.logger.info(f"Closing {service_name}...")
                        await service.close()
                    else:
                        # Just set to None to help garbage collection
                        setattr(service, attr_name, None)
            
            # Close database connection last
            if hasattr(self, 'client') and self.client:
                self.logger.info("Closing database connection...")
                close_database_connection(self.client)
                self.client = None
                self.db = None
                
            self.logger.info("Shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.error(traceback.format_exc())

    def _setup_handlers(self):
        self.command_handler.register_handlers(self.application)
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)
        self.message_handlers.register_handlers(self.application)
        self.application.add_handler(CommandHandler("remind", self.reminder_manager.set_reminder))
        self.application.add_handler(CommandHandler("language", self.language_manager.set_language))
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))
        self.application.add_handler(CommandHandler("documents", self.command_handler.show_document_history))
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
    app = create_app(main_bot, loop)

    if os.environ.get('DEV_SERVER') == 'uvicorn':
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            if not os.getenv('WEBHOOK_URL'):
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
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            loop.run_until_complete(main_bot.application.shutdown())
            loop.run_until_complete(flux_lora_image_generator.close())
            # Add this line to properly close text_to_video_generator
           
            loop.run_until_complete(text_to_video_generator.close())
            close_database_connection(main_bot.client)
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()