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
        # Initialize only essential services at startup
        self.logger = logging.getLogger(__name__)
        self.response_cache = TTLCache(maxsize=100, ttl=3600)
        self.user_response_cache = {}
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
            
        # Initialize database connection with a timeout and retry mechanism
        self._init_db_connection()
        
        # Create application with reasonable timeout settings
        self.application = (
            Application.builder()
            .token(self.token)
            .persistence(PicklePersistence(filepath='conversation_states.pickle'))
            .http_version('1.1')
            .get_updates_http_version('1.1')
            .read_timeout(20)     # Reduced from 30s
            .write_timeout(20)    # Reduced from 30s
            .connect_timeout(15)  # Reduced from 30s
            .pool_timeout(15)     # Reduced from 30s
            .connection_pool_size(16)  # Add connection pool setting
            .build()
        )
        
        # Initialize other services as needed
        self._init_services()
        self._setup_handlers()
        
    def _init_db_connection(self):
        # More efficient retry with exponential backoff
        max_retries = 3
        retry_delay = 0.5  # Start with 500ms
        
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
        # Initialize services with proper error handling
        try:
            # Use a more efficient model if available
            model_name = "gemini-2.0-flash"
            vision_model = genai.GenerativeModel(model_name)
            rate_limiter = RateLimiter(requests_per_minute=20)  # Increased rate limit
            self.gemini_api = GeminiAPI(vision_model=vision_model, rate_limiter=rate_limiter)
            
            # Other initializations
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

    def   shutdown(self):
        close_database_connection(self.client)
        logger.info("Shutdown complete. Database connection closed.")

    def _setup_handlers(self):
        # Create a response cache
        self.response_cache = TTLCache(maxsize=1000, ttl=60)     
        # Register handlers with cache awareness
        self.command_handler.register_handlers(self.application, cache=self.response_cache)
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)
        self.message_handlers.register_handlers(self.application)
        self.application.add_handler(CommandHandler("remind", self.reminder_manager.set_reminder))
        self.application.add_handler(CommandHandler("language", self.language_manager.set_language))
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))
        self.application.add_handler(CommandHandler("documents", self.command_handler.show_document_history))
        
        # Remove any existing error handlers before adding new one
        self.application.error_handlers.clear()
        # Add error handler last
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
            "max_connections": 100
        }

        self.logger.info(f"Setting webhook to: {webhook_url}")

        if not self.application.running:
            await self.application.initialize()
            await self.application.start()

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
        """Process updates received from webhook with improved error handling."""
        try:
            # Ensure application is initialized before processing updates
            if not self.application.running:
                self.logger.info("Application not initialized yet, initializing now...")
                await self.application.initialize()
                await self.application.start()
                
            update = Update.de_json(update_data, self.application.bot)
            
            try:
                # Process update normally after ensuring initialization
                # Increased timeout to 300 seconds for model processing
                await asyncio.wait_for(
                    self.application.process_update(update),
                    timeout=300.0
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Update processing timed out: {update.update_id}")
                if hasattr(update, 'message') and update.message:
                    asyncio.create_task(update.message.reply_text(
                        "I'm still thinking about your request... This may take a bit longer as I'm carefully considering my response."
                    ))
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
                # Extract data first
                update_data = await request.json()
                
                # Start processing in background task
                asyncio.create_task(self.process_update(update_data))
                
                # Return response immediately
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
    
    # Ensure the application is initialized before accepting webhook requests
    if os.environ.get('DEV_SERVER') == 'uvicorn':
        # For development server, initialize the application first
        loop.run_until_complete(main_bot.application.initialize())
        loop.run_until_complete(main_bot.application.start())
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            if not os.getenv('WEBHOOK_URL'):
                logger.error("WEBHOOK_URL not set in .env")
                sys.exit(1)

            # Initialize and start the application before setting up webhooks
            loop.run_until_complete(main_bot.application.initialize())
            loop.run_until_complete(main_bot.application.start())
            logger.info("Bot application initialized and started")
            
            # Now set up the webhook
            loop.run_until_complete(main_bot.setup_webhook())
            
            # Register the webhook handler
            app = create_app(main_bot, loop)
            
            # Run FastAPI
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
    """Create and configure the application for uvicorn without creating a new event loop"""
    # Set the environment variable so our code knows we're using uvicorn
    os.environ["DEV_SERVER"] = "uvicorn"
    
    # Initialize the bot
    bot = TelegramBot()
    
    # Setup webhook handling without creating a new loop
    existing_loop = asyncio.get_event_loop()
    bot.run_webhook(existing_loop)
    
    # Create a startup event to initialize the application when uvicorn starts
    @app.on_event("startup")
    async def startup_event():
        # Initialize and start the application using uvicorn's event loop
        await bot.application.initialize()
        await bot.application.start()
        
        # Set up the webhook if WEBHOOK_URL is provided
        if os.getenv('WEBHOOK_URL'):
            await bot.setup_webhook()
    
    return app

# For uvicorn to import
application = get_application()