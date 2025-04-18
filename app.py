from src.handlers.message_context_handler import MessageContextHandler
from src.handlers.response_formatter import ResponseFormatter
from src.handlers.media_context_extractor import MediaContextExtractor
from src.services.media.image_processor import ImageProcessor
from src.services.media.voice_processor import VoiceProcessor
from src.services.model_handlers.prompt_formatter import PromptFormatter
from src.services.user_preferences_manager import UserPreferencesManager
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
from telegram import Update
import traceback
from telegram.ext import (
    Application,
    CommandHandler,
    PicklePersistence,
)
from cachetools import TTLCache, LRUCache
import platform
import psutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database.connection import get_database, close_database_connection
from src.services.user_data_manager import UserDataManager
from src.services.gemini_api import GeminiAPI
from src.services.openrouter_api import OpenRouterAPI
from src.handlers.command_handlers import CommandHandlers
from src.handlers.text_handlers import TextHandler
from src.services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM
from src.handlers.message_handlers import (
    MessageHandlers,
)  # Ensure this is your custom handler
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
from dotenv import load_dotenv

load_dotenv()

# Configure logging efficiently
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Initialize FastAPI with performance optimizations
app = FastAPI()
app.add_middleware(
    GZipMiddleware, minimum_size=1000
)  # Compress responses to reduce network latency

# Create thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)


@app.get("/")
async def root_get():
    """Root endpoint for health checks."""
    return JSONResponse(
        content={"status": "ok", "message": "Telegram Bot API is running"},
        status_code=200,
    )


@app.post("/")
async def root_post():
    """Root endpoint for POST requests."""
    return JSONResponse(
        content={"status": "ok", "message": "Telegram Bot API is running"},
        status_code=200,
    )


@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint optimized for Koyeb deployments.
    Returns 200 OK with detailed health information.
    """

    try:
        # Gather system metrics
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "uptime": time.time() - psutil.boot_time(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

        # Return detailed health information
        return JSONResponse(
            content={
                "status": "ok",
                "message": "Service is healthy",
                "timestamp": time.time(),
                "system_info": system_info,
            },
            status_code=200,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
            },
        )
    except Exception as e:
        logger.error(f"Error collecting health metrics: {str(e)}")
        return JSONResponse(
            content={"status": "ok", "message": "Service is operational"},
            status_code=200,
        )


@app.post("/test-webhook")
async def test_webhook(request: Request):
    """Test endpoint for webhook validation with Postman."""
    try:
        # Parse the incoming JSON
        data = await request.json()

        # Log the received data
        logger.info(f"Received test webhook data: {data}")

        # Return a successful response with the echoed data
        return JSONResponse(
            content={
                "status": "success",
                "message": "Webhook test successful",
                "received_data": data,
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error in test webhook: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": f"Webhook test failed: {str(e)}"},
            status_code=400,
        )


class TelegramBot:
    def __init__(self):
        # Initialize only essential services at startup
        self.logger = logging.getLogger(__name__)

        # More efficient caching strategy
        self.response_cache = TTLCache(maxsize=500, ttl=3600)  # Increased cache size
        self.user_response_cache = LRUCache(
            maxsize=100
        )  # Use LRU cache for user responses

        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

        # Initialize database connection with a timeout and retry mechanism
        self._init_db_connection()

        # Create application with optimized timeout settings
        self.application = (
            Application.builder()
            .token(self.token)
            .persistence(PicklePersistence(filepath="conversation_states.pickle"))
            .http_version("1.1")
            .get_updates_http_version("1.1")
            .read_timeout(None)
            .write_timeout(None)
            .connect_timeout(None)
            .pool_timeout(None)
            .connection_pool_size(128)
            .build()
        )

        # Initialize other services as needed
        self._init_services()
        self._setup_handlers()

        # Create client session for HTTP requests
        self.session = None

    async def create_session(self):
        """Create an aiohttp session for HTTP requests."""
        try:
            if self.session is None or self.session.closed:
                # Create session with optimized parameters
                tcp_connector = aiohttp.TCPConnector(
                    limit=150,  # Increased connection pool size
                    limit_per_host=30,
                    force_close=False,
                    enable_cleanup_closed=True,
                )
                self.session = aiohttp.ClientSession(
                    connector=tcp_connector,
                    timeout=aiohttp.ClientTimeout(total=300, connect=30),
                )
                self.logger.info("Created new aiohttp session for bot")
            return self.session
        except Exception as e:
            self.logger.error(f"Failed to create aiohttp session: {str(e)}")
            raise

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
                    self.logger.warning(
                        f"Database connection attempt {attempt+1} failed, retrying..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.error(f"All database connection attempts failed: {e}")
                    raise

    def _init_services(self):
        # Initialize services with proper error handling
        try:
            # Use a more efficient model if available
            model_name = "gemini-2.5-pro-exp-03-25"
            vision_model = genai.GenerativeModel(model_name)
            rate_limiter = RateLimiter(
                requests_per_minute=30
            )  # Increased rate limit for better throughput
            self.gemini_api = GeminiAPI(
                vision_model=vision_model, rate_limiter=rate_limiter
            )

            # Initialize OpenRouter API for Quasar Alpha
            openrouter_rate_limiter = RateLimiter(requests_per_minute=20)
            self.openrouter_api = OpenRouterAPI(rate_limiter=openrouter_rate_limiter)

            # Initialize DeepSeekLLM for DeepSeek model
            self.deepseek_api = DeepSeekLLM()

            # Store API instances in application context instead of directly on bot
            # The telegram library doesn't allow adding attributes to the bot object
            if not hasattr(self.application, "bot_data"):
                self.application.bot_data = {}
            self.application.bot_data["gemini_api"] = self.gemini_api
            self.application.bot_data["openrouter_api"] = self.openrouter_api
            self.application.bot_data["deepseek_api"] = self.deepseek_api

            # Other initializations
            self.user_data_manager = UserDataManager(self.db)
            self.telegram_logger = telegram_logger

            # Create instances of utility classes
            self.context_handler = MessageContextHandler()
            self.response_formatter = ResponseFormatter()
            self.media_context_extractor = MediaContextExtractor()
            self.image_processor = ImageProcessor(self.gemini_api)
            self.voice_processor = VoiceProcessor()
            self.prompt_formatter = PromptFormatter()
            self.preferences_manager = UserPreferencesManager(self.user_data_manager)

            # Store utility instances in application context for access from handlers
            self.application.bot_data["context_handler"] = self.context_handler
            self.application.bot_data["response_formatter"] = self.response_formatter
            self.application.bot_data["media_context_extractor"] = (
                self.media_context_extractor
            )
            self.application.bot_data["image_processor"] = self.image_processor
            self.application.bot_data["voice_processor"] = self.voice_processor
            self.application.bot_data["prompt_formatter"] = self.prompt_formatter
            self.application.bot_data["preferences_manager"] = self.preferences_manager

        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            raise

        # Initialize TextHandler with utility classes and ALL API instances
        self.text_handler = TextHandler(
            gemini_api=self.gemini_api,
            user_data_manager=self.user_data_manager,
            openrouter_api=self.openrouter_api,  # Ensure this is passed
            deepseek_api=self.deepseek_api,  # Ensure this is passed
        )

        # Once text_handler is initialized, create ConversationManager
        from src.services.conversation_manager import ConversationManager

        self.conversation_manager = ConversationManager(
            self.text_handler.memory_manager, self.text_handler.model_history_manager
        )
        self.application.bot_data["conversation_manager"] = self.conversation_manager

        # Initialize other handlers - pass ALL API instances needed by any model
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api,
            user_data_manager=self.user_data_manager,
            telegram_logger=self.telegram_logger,
            flux_lora_image_generator=flux_lora_image_generator,
            deepseek_api=self.deepseek_api,  # Ensure this is passed
            openrouter_api=self.openrouter_api,  # Ensure this is passed for DeepCoder and other models
        )

        # Initialize DocumentProcessor with the bot parameter
        self.document_processor = DocumentProcessor(bot=self.application.bot)

        # Update MessageHandlers initialization with all dependencies including utility classes
        self.message_handlers = MessageHandlers(
            self.gemini_api,
            self.user_data_manager,
            self.telegram_logger,
            self.document_processor,
            self.text_handler,
            deepseek_api=self.deepseek_api,  # Pass the deepseek_api instance
            openrouter_api=self.openrouter_api,  # Pass the openrouter_api instance
        )

        # Share utility classes with message_handlers
        self.message_handlers.context_handler = self.context_handler
        self.message_handlers.response_formatter = self.response_formatter
        self.message_handlers.media_context_extractor = self.media_context_extractor
        self.message_handlers.image_processor = self.image_processor
        self.message_handlers.voice_processor = self.voice_processor
        self.message_handlers.prompt_formatter = self.prompt_formatter
        self.message_handlers.preferences_manager = self.preferences_manager
        self.message_handlers._conversation_manager = self.conversation_manager

        # Initialize other services
        self.reminder_manager = ReminderManager(self.application.bot)
        self.language_manager = LanguageManager()

    async def shutdown(self):
        """Properly clean up resources on shutdown"""
        # Close aiohttp session if it exists
        if self.session and not self.session.closed:
            await self.session.close()

        # Close database connection
        close_database_connection(self.client)
        logger.info("Shutdown complete. Database connection closed.")

    def _setup_handlers(self):
        # Create a response cache - improved sizing
        self.response_cache = TTLCache(
            maxsize=1000, ttl=300
        )  # Reduced TTL for fresher responses

        # Register command handlers via CommandHandlers
        self.command_handler.register_handlers(
            self.application, cache=self.response_cache
        )
        # Register message handlers
        self.message_handlers.register_handlers(self.application)
        # Register reminder and language commands separately
        self.application.add_handler(
            CommandHandler("remind", self.reminder_manager.set_reminder)
        )
        self.application.add_handler(
            CommandHandler("language", self.language_manager.set_language)
        )
        # Note: Other commands (start, help, reset, settings, stats, export, etc.)
        # are registered inside CommandHandlers.register_handlers

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

        # Optimized webhook configuration
        webhook_config = {
            "url": webhook_url,
            "allowed_updates": [
                "message",
                "edited_message",
                "callback_query",
                "inline_query",
            ],
            "max_connections": 150,  # Increased for better parallel processing
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
        """Process updates received from webhook without timeout."""
        try:
            if not self.application.running:
                await self.application.initialize()
                await self.application.start()

            update = Update.de_json(update_data, self.application.bot)

            # Process update in task to avoid blocking
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self.application.process_update(update))

        except Exception as e:
            self.logger.error(f"Error in process_update: {str(e)}")
            if hasattr(update, "message") and update.message:
                try:
                    await update.message.reply_text("Processing your request...")
                except Exception as reply_error:
                    self.logger.error(f"Failed to send error message: {reply_error}")

    # Update the webhook handler in run_webhook method
    def run_webhook(self, loop):
        @app.post(f"/webhook/{self.token}")
        async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
            try:
                # Extract data with no timeout
                update_data = await request.json()

                # Log incoming update for debugging
                self.logger.info(
                    f"Received webhook update: {update_data.get('update_id', 'unknown')}"
                )

                # Return immediate response before processing to prevent webhook timeout
                background_tasks.add_task(self.process_update, update_data)

                return JSONResponse(
                    content={"status": "ok"},
                    status_code=200,
                    headers={"Connection": "keep-alive"},
                )
            except Exception as e:
                self.logger.error(f"Webhook error: {str(e)}")
                return JSONResponse(
                    content={"status": "error", "detail": str(e)}, status_code=500
                )

    async def start_keep_alive(self):
        """Start the keep-alive task to maintain persistent connections."""
        self.logger.info("Starting keep-alive task")
        asyncio.create_task(self._keep_alive_task())

    async def _keep_alive_task(self):
        """
        Advanced keep-alive mechanism specifically optimized for Koyeb deployments.
        This prevents the instance from being stopped due to inactivity.
        """
        self.logger.info("Starting enhanced keep-alive task for Koyeb")
        ping_interval = 60  # Ping every minute (more frequent for Koyeb)
        reconnect_attempts = 0
        max_reconnect_attempts = 5

        while True:
            try:
                # 1. Check Telegram API connection
                me = await self.application.bot.get_me()
                self.logger.debug(f"Keep-alive: Connected to Telegram as {me.username}")
                reconnect_attempts = 0  # Reset counter on successful connection

                # 2. Ping our own health endpoint with additional headers
                if os.getenv("WEBHOOK_URL"):
                    # Extract base URL from webhook URL (remove /webhook/TOKEN part)
                    base_url = os.getenv("WEBHOOK_URL").split("/webhook")[0]
                    async with aiohttp.ClientSession() as session:
                        health_url = f"{base_url}/health"
                        headers = {
                            "User-Agent": "TelegramBot/1.0 KeepAlive",
                            "Connection": "keep-alive",
                            "X-Keep-Alive": "true",
                        }
                        async with session.get(
                            health_url, headers=headers, timeout=10
                        ) as response:
                            if response.status == 200:
                                self.logger.debug(
                                    f"Health check passed: {await response.text()}"
                                )
                            else:
                                self.logger.warning(
                                    f"Health check returned status {response.status}"
                                )
                else:
                    self.logger.warning(
                        "No WEBHOOK_URL set, skipping self health check"
                    )

            except asyncio.CancelledError:
                self.logger.info("Keep-alive task cancelled")
                break

            except Exception as e:
                self.logger.warning(f"Keep-alive check failed: {e}")
                reconnect_attempts += 1

                # If connection to Telegram API was lost, try to reconnect with exponential backoff
                if reconnect_attempts <= max_reconnect_attempts:
                    try:
                        if not self.application.running:
                            await self.application.initialize()
                            await self.application.start()
                            await self.setup_webhook()
                            self.logger.info("Successfully reconnected to Telegram API")
                    except Exception as reconnect_error:
                        self.logger.error(f"Failed to reconnect: {reconnect_error}")

                    # Exponential backoff for reconnection attempts
                    backoff_time = min(30, 2**reconnect_attempts)
                    self.logger.info(
                        f"Backing off for {backoff_time}s before next reconnection attempt"
                    )
                    await asyncio.sleep(backoff_time)
                    continue

            # Use a shorter interval to prevent Koyeb from stopping the instance
            await asyncio.sleep(ping_interval)


async def start_bot(webhook: TelegramBot):
    try:
        # Create HTTP session
        await webhook.create_session()

        # Initialize and start application
        await webhook.application.initialize()
        await webhook.application.start()
        logger.info("Bot started successfully.")
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# Add this method to TelegramBot class
async def keep_alive(self):
    """Keep the connection alive."""
    while True:
        try:
            await self.application.bot.get_me()
        except Exception as e:
            self.logger.warning(f"Keep-alive check failed: {e}")
        finally:
            await asyncio.sleep(60)  # Check every minute


def create_app(webhook: TelegramBot, loop):
    webhook.run_webhook(loop)
    return app


if __name__ == "__main__":
    main_bot = TelegramBot()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Ensure the application is initialized before accepting webhook requests
    if os.environ.get("DEV_SERVER") == "uvicorn":
        # For development server, initialize the application first
        loop.run_until_complete(main_bot.create_session())
        loop.run_until_complete(main_bot.application.initialize())
        loop.run_until_complete(main_bot.application.start())
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        try:
            if not os.getenv("WEBHOOK_URL"):
                logger.error("WEBHOOK_URL not set in .env")
                sys.exit(1)

            # Initialize and start the application before setting up webhooks
            loop.run_until_complete(main_bot.create_session())
            loop.run_until_complete(main_bot.application.initialize())
            loop.run_until_complete(main_bot.application.start())
            logger.info("Bot application initialized and started")

            # Now set up the webhook
            loop.run_until_complete(main_bot.setup_webhook())

            # Register the webhook handler
            app = create_app(main_bot, loop)

            def run_fastapi():
                port = int(os.environ.get("PORT", 8000))
                config = uvicorn.Config(
                    app,
                    host="0.0.0.0",
                    port=port,
                    loop="asyncio",
                    timeout_keep_alive=None,
                    timeout_graceful_shutdown=None,
                    limit_concurrency=None,  #
                    backlog=4096,
                    workers=4,
                )
                server = uvicorn.Server(config)
                server.run()

            fastapi_thread = Thread(target=run_fastapi)
            fastapi_thread.start()
            loop.run_forever()
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            loop.run_until_complete(main_bot.shutdown())  # Proper async shutdown
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
        try:
            await bot.create_session()
            await bot.application.initialize()
            await bot.application.start()
            logger.info("Bot application initialized and started")

            # Start the keep-alive mechanism
            await bot.start_keep_alive()
            logger.info("Keep-alive mechanism started")

            if os.getenv("WEBHOOK_URL"):
                await bot.setup_webhook()
                logger.info(f"Webhook set up at {os.getenv('WEBHOOK_URL')}")

            # Log successful startup
            logger.info("Application startup complete and all systems operational")
        except Exception as e:
            logger.error(f"Startup error: {str(e)}")
            logger.error(traceback.format_exc())

    # Add shutdown handler
    @app.on_event("shutdown")
    async def shutdown_event():
        await bot.application.stop()
        await bot.application.shutdown()

    return app


app = get_application()
