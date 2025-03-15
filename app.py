import os
import sys
import logging
import asyncio
import time
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from dotenv import load_dotenv
from telegram import Update
import traceback
from telegram.ext import (
    Application, 
    CommandHandler, 
    PicklePersistence,
)
from contextlib import asynccontextmanager
from cachetools import TTLCache, LRUCache
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

# Load environment variables
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

# Create thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

class TelegramBot:
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Caching strategy
        self.response_cache = TTLCache(maxsize=500, ttl=3600)
        self.user_response_cache = LRUCache(maxsize=100)
        
        # Get token from environment variables
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
            
        # Initialize database connection
        self._init_db_connection()
        
        # Create application
        self.application = (
            Application.builder()
            .token(self.token)
            .persistence(PicklePersistence(filepath='conversation_states.pickle'))
            .http_version('1.1')
            .get_updates_http_version('1.1')
            .connection_pool_size(128)
            .build()
        )
        
        # Initialize services and handlers
        self._init_services() 
        self._setup_handlers()

    async def create_session(self):
        """Create an aiohttp session for HTTP requests."""
        if not hasattr(self, 'session') or self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)  
            )
            return self.session

    async def close_db_connection(self):
        """Close the database connection."""
        if hasattr(self, 'client') and self.client:
            close_database_connection(self.client)
            self.logger.info("Database connection closed.")
            self.client = None
            self.db = None
    
    def _init_db_connection(self):
        """Initialize database connection with retry logic."""
        max_retries = 3
        retry_delay = 0.5
        
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
                    retry_delay *= 2
                else:
                    self.logger.error(f"All database connection attempts failed: {e}")
                    raise
    
    def _init_services(self):
        """Initialize all required services."""
        try:
            # Initialize Gemini API
            model_name = "google/gemma-3-27b-it"
            vision_model = genai.GenerativeModel(model_name)
            rate_limiter = RateLimiter(requests_per_minute=30)
            self.gemini_api = GeminiAPI(vision_model=vision_model, rate_limiter=rate_limiter)
            
            # Initialize user data manager and logger
            self.user_data_manager = user_data_manager(self.db)
            self.telegram_logger = telegram_logger
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            raise

        # Initialize handlers
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
        """Clean up resources on shutdown."""
        # Close aiohttp session
        if hasattr(self, 'session') and self.session and not self.session.closed:
            await self.session.close()
        
        # Close database connection
        await self.close_db_connection()
        logger.info("Shutdown complete. All resources closed.")

    def _setup_handlers(self):
        """Set up all message handlers."""
        # Configure response cache
        self.response_cache = TTLCache(maxsize=1000, ttl=300)
        
        # Register command handlers
        self.command_handler.register_handlers(self.application, cache=self.response_cache)
        
        # Register text handlers
        for handler in self.text_handler.get_handlers():
            self.application.add_handler(handler)
            
        # Register message handlers
        self.message_handlers.register_handlers(self.application)
        
        # Add special command handlers
        self.application.add_handler(CommandHandler("remind", self.reminder_manager.set_reminder))
        self.application.add_handler(CommandHandler("language", self.language_manager.set_language))
        self.application.add_handler(CommandHandler("history", self.text_handler.show_history))
        self.application.add_handler(CommandHandler("documents", self.command_handler.show_document_history))
        
        # Set error handler
        self.application.error_handlers.clear()
        self.application.add_error_handler(self.message_handlers._error_handler)
        
    async def setup_webhook(self):
        """Set up webhook with proper update processing."""
        webhook_path = f"/webhook/{self.token}"
        webhook_url = f"{os.getenv('WEBHOOK_URL')}{webhook_path}"

        # Delete existing webhook
        await self.application.bot.delete_webhook(drop_pending_updates=True)

        # Configure webhook
        webhook_config = {
            "url": webhook_url,
            "allowed_updates": ["message", "edited_message", "callback_query", "inline_query"],
            "max_connections": 150
        }

        self.logger.info(f"Setting webhook to: {webhook_url}")

        # Ensure application is running
        if not self.application.running:
            await self.application.initialize()
            await self.application.start()

        # Set webhook
        await self.application.bot.set_webhook(**webhook_config)
        
        # Verify webhook is set
        webhook_info = await self.application.bot.get_webhook_info()
        self.logger.info(f"Webhook status: {webhook_info}")
        
        # Double-check if application is running
        if not self.application.running:
            await self.application.start()
            self.logger.info("Application started.")
        else:
            self.logger.info("Application is already running.")

    async def process_update(self, update_data: dict):
        """Process updates received from webhook."""
        try:
            # Ensure application is initialized
            if not self.application.running:
                await self.application.initialize()
                await self.application.start()
                
            # Parse update
            update = Update.de_json(update_data, self.application.bot)
            
            # Process update asynchronously
            await self.application.process_update(update)
                
        except Exception as e:
            self.logger.error(f"Error in process_update: {str(e)}")
            self.logger.error(traceback.format_exc())

# Create FastAPI application with lifespan management
def create_application():
    """Create the FastAPI application with proper configuration."""
    
    # Initialize FastAPI app
    app = FastAPI(title="Telegram Bot API")
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Create bot instance
    bot = TelegramBot()
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        # Startup
        app.state.bot = bot
        await bot.create_session()
        await bot.application.initialize()
        await bot.application.start()
        
        if os.getenv('WEBHOOK_URL'):
            await bot.setup_webhook()
        
        logger.info("Application startup complete.")
        yield
        
        # Shutdown
        logger.info("Application shutting down...")
        if hasattr(bot, 'session') and not bot.session.closed:
            await bot.session.close()
        await bot.application.stop()
        await bot.application.shutdown()
        await bot.close_db_connection()
        logger.info("Application shutdown complete.")
    
    # Register lifespan
    app.lifespan = lifespan
    
    # Register routes
    @app.get("/health")
    async def health_check():
        """Simple health check endpoint."""
        return {"status": "ok"}
    
    @app.post("/")
    async def root_post():
        """Handle POST requests to root."""
        return JSONResponse(content={"message": "Post request received"}, status_code=200)
    
    @app.post(f"/webhook/{bot.token}")
    async def webhook_handler(request: Request):
        """Handle webhook requests from Telegram."""
        try:
            update_data = await request.json()
            await bot.process_update(update_data)
            return JSONResponse(content={"status": "ok"}, status_code=200)
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                content={"status": "error", "detail": str(e)},
                status_code=500
            )
    
    # Add fallback handler for unknown tokens
    @app.post("/webhook/{token:path}")
    async def fallback_webhook(token: str, request: Request):
        """Handle webhook requests with unknown tokens."""
        logger.warning(f"Received webhook for unknown token: {token}")
        return JSONResponse(content={"status": "ok"}, status_code=200)
    
    return app

# Create application for uvicorn to import
application = create_application()

# Entry point for direct execution
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    # Configure uvicorn
    config = uvicorn.Config(
        "app:application",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        workers=int(os.environ.get("WORKERS", 1)),
        timeout_keep_alive=70,
        timeout_graceful_shutdown=30,
        loop="asyncio"
    )
    
    # Run server
    server = uvicorn.Server(config)
    server.run()