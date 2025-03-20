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
import signal
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
from src.handlers.message_handlers import MessageHandlers
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

# Load environment variables at the start
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Telegram-Gemini-Bot API", version="1.0.0")

@app.get('/health')
async def health_check():
    """Health check endpoint"""
    return JSONResponse(content={"status": "ok"}, status_code=200)

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {"message": "Telegram Gemini Bot is running!"}

class TelegramBot:
    def __init__(self):
        """Initialize the Telegram Bot with necessary services"""
        self.logger = logging.getLogger(__name__)
        self.response_cache = TTLCache(maxsize=1000, ttl=3600)
        self.user_response_cache = {}
        
        # Check for required environment variables
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            if os.getenv('DEV_MODE', 'false').lower() == 'true':
                self.logger.warning("TELEGRAM_BOT_TOKEN not found in .env file, using placeholder for development")
                self.token = "dev_placeholder_token"
            else:
                raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
            
        # Initialize database connection with fallback
        self._init_db_connection()
        
        # Create application with optimized settings
        self.application = (
            Application.builder()
            .token(self.token)
            .persistence(PicklePersistence(filepath='conversation_states.pickle'))
            .http_version('1.1')
            .get_updates_http_version('1.1')
            .connection_pool_size(32)
            .build()
        )
        
        self._init_services()
        self._setup_handlers()
        
    def _init_db_connection(self):
        """Initialize MongoDB connection with retry and fallback mechanism"""
        max_retries = 5
        retry_delay = 0.2
        
        for attempt in range(max_retries):
            try:
                self.db, self.client = get_database()
                
                # If DEV_MODE and IGNORE_DB_ERROR are enabled, the get_database function
                # will return a mock database that won't cause errors
                if self.db is None and os.getenv('DEV_MODE', 'false').lower() != 'true' and os.getenv('IGNORE_DB_ERROR', 'false').lower() != 'true':
                    raise ConnectionError("Failed to connect to the database")
                
                self.logger.info("Database connection established or mock DB initialized")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Database connection attempt {attempt+1} failed: {str(e)}. Retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.logger.error(f"All database connection attempts failed: {e}")
                    if os.getenv('DEV_MODE', 'false').lower() == 'true' or os.getenv('IGNORE_DB_ERROR', 'false').lower() == 'true':
                        self.logger.warning("Running in development mode without database")
                        self.db = None
                        self.client = None
                    else:
                        raise
    
    def _init_services(self):
        """Initialize services with proper error handling"""
        try:
            # Initialize Gemini API
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                if os.getenv('DEV_MODE', 'false').lower() == 'true':
                    self.logger.warning("GEMINI_API_KEY not found, using placeholder for development")
                    api_key = "fake_api_key_for_dev"
                    genai.configure(api_key=api_key)
                else:
                    raise ValueError("GEMINI_API_KEY not found in environment")
            else:
                genai.configure(api_key=api_key)
            
            # Use model based on availability
            model_name = "gemini-2.0-flash"
            self.logger.info(f"Using model: {model_name}")
            
            try:
                vision_model = genai.GenerativeModel(model_name)
            except Exception as model_error:
                self.logger.error(f"Error loading model {model_name}: {model_error}")
                vision_model = None
                
            rate_limiter = RateLimiter(requests_per_minute=30)
            self.gemini_api = GeminiAPI(vision_model=vision_model, rate_limiter=rate_limiter)
            
            # Initialize other required services
            self.user_data_manager = user_data_manager(self.db)
            self.telegram_logger = telegram_logger
        except Exception as e:
            self.logger.error(f"Error initializing core services: {e}")
            self.logger.error(traceback.format_exc())
            if os.getenv('DEV_MODE', 'false').lower() != 'true':
                raise

        # Initialize handlers and processors
        try:
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
        except Exception as handler_error:
            self.logger.error(f"Error initializing handlers: {handler_error}")
            self.logger.error(traceback.format_exc())
            if os.getenv('DEV_MODE', 'false').lower() != 'true':
                raise

    async def shutdown(self):
        """Properly shut down all services"""
        try:
            # First cancel any pending tasks
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
            # Close database connection
            if self.client:
                close_database_connection(self.client)
                self.client = None
                self.db = None
            
            # Stop reminder manager if running
            if hasattr(self, 'reminder_manager') and hasattr(self.reminder_manager, 'reminder_check_task'):
                await self.reminder_manager.stop()
                
            # Stop bot application if running
            if hasattr(self, 'application') and self.application.running:
                await self.application.stop()
                
            logger.info("Bot shutdown complete")
        except Exception as e:
            logger.error(f"Error during bot shutdown: {e}")
            logger.error(traceback.format_exc())
            
    def _setup_handlers(self):
        """Set up all message and command handlers"""
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
        """Set up webhook for Telegram updates"""
        webhook_path = f"/webhook/{self.token}"
        webhook_url = os.getenv('WEBHOOK_URL')
        
        if not webhook_url:
            self.logger.warning("WEBHOOK_URL not set in environment. Using localhost for development.")
            webhook_url = "https://localhost:8000"
            
        webhook_url = f"{webhook_url}{webhook_path}"
        
        # Delete existing webhook and clear pending updates
        await self.application.bot.delete_webhook(drop_pending_updates=True)

        # Webhook configuration
        webhook_config = {
            "url": webhook_url,
            "allowed_updates": ["message", "edited_message", "callback_query", "inline_query"],
            "max_connections": 100
        }

        self.logger.info(f"Setting webhook to: {webhook_url}")

        # Initialize application if needed
        if not self.application.running:
            await self.application.initialize()
            await self.application.start()

        # Set webhook
        await self.application.bot.set_webhook(**webhook_config)

        # Verify webhook status
        webhook_info = await self.application.bot.get_webhook_info()
        self.logger.info(f"Webhook status: {webhook_info}")

    async def process_update(self, update_data: dict):
        """Process updates received from webhook"""
        try:
            # Initialize application if not running
            if not self.application.running:
                self.logger.info("Application not initialized yet, initializing now...")
                await self.application.initialize()
                await self.application.start()
                
            update = Update.de_json(update_data, self.application.bot)
            
            # Process update as background task for better concurrency
            asyncio.create_task(self.application.process_update(update))
            
        except Exception as e:
            self.logger.error(f"Error in process_update: {str(e)}")
            self.logger.error(traceback.format_exc())
                
    def run_webhook(self, loop):
        """Configure webhook handler for FastAPI"""
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
    """Start the bot application"""
    try:
        await webhook.application.initialize()
        await webhook.application.start()
        logger.info("Bot started successfully.")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def create_app(webhook: TelegramBot, loop):
    """Create and configure FastAPI application"""
    webhook.run_webhook(loop)
    return app

def get_application():
    """Create and configure the application for uvicorn"""
    os.environ["DEV_SERVER"] = "uvicorn"
    
    # Load environment variables again to ensure they're available
    load_dotenv()
    
    # Create FastAPI app
    app = FastAPI(title="Telegram-Gemini-Bot API", version="1.0.0")
    
    @app.get('/health')
    async def health_check():
        return JSONResponse(content={"status": "ok"}, status_code=200)
    
    @app.get("/")
    async def read_root():
        return {"message": "Telegram Gemini Bot is running! (Server Mode)"}
    
    # Create bot instance
    bot = TelegramBot()
    
    # Get the event loop
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
        
        webhook_url = os.getenv('WEBHOOK_URL')
        if webhook_url:
            await bot.setup_webhook()
        
        if hasattr(bot, 'reminder_manager'):
            await bot.reminder_manager.start()
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Application shutdown initiated")
        await bot.shutdown()
            
    return app

# Check if running directly or imported
if __name__ == '__main__':
    # Create bot instance
    main_bot = TelegramBot()
    
    # Create event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Check run mode
    if os.environ.get('DEV_SERVER') == 'uvicorn':
        # For development with uvicorn, set up the app for hot reload
        loop.run_until_complete(main_bot.application.initialize())
        loop.run_until_complete(main_bot.application.start())
        app = create_app(main_bot, loop)
        # Let uvicorn handle the rest
    else:
        # For production mode with integrated server
        try:
            # Setup webhook and create app
            loop.run_until_complete(main_bot.application.initialize())
            loop.run_until_complete(main_bot.application.start())
            
            if os.getenv('WEBHOOK_URL'):
                loop.run_until_complete(main_bot.setup_webhook())
            else:
                logger.warning("WEBHOOK_URL not set. Running in development mode without webhook.")
                
            app = create_app(main_bot, loop)
            
            # Run FastAPI in a thread
            def run_fastapi():
                port = int(os.environ.get("PORT", 8000))
                uvicorn.run(
                    app, 
                    host="0.0.0.0", 
                    port=port,
                    log_level="info" if os.getenv('DEV_MODE', 'false').lower() == 'true' else "warning"
                )
            
            # Start FastAPI thread
            fastapi_thread = Thread(target=run_fastapi, daemon=True)
            fastapi_thread.start()
            
            # Set up signal handlers for graceful shutdown
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(
                    sig, 
                    lambda: asyncio.create_task(main_bot.shutdown())
                )
            
            # Keep main thread running
            logger.info("Bot is now running. Press Ctrl+C to stop.")
            loop.run_forever()
            
        except Exception as e:
            logger.error(f"Startup error: {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)
            
        finally:
            # Ensure cleanup on exit
            loop.run_until_complete(main_bot.shutdown())
            loop.close()

# For use with uvicorn directly
app = get_application()
