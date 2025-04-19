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
from dotenv import load_dotenv
from telegram import Update
import traceback
from telegram.ext import (
    Application,
    CommandHandler,
    PicklePersistence,
)
from cachetools import TTLCache, LRUCache
import threading
import requests
from contextlib import asynccontextmanager
import uuid
import json
import ipaddress
import psutil

# Import the message filter
from src.utils.ignore_message import message_filter

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
@app.head("/")
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
@app.head("/health")  # Add HEAD request handler for health endpoint
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={"status": "ok", "message": "Service is healthy"}, status_code=200
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

        # Track active update processing tasks
        self._update_tasks = set()

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

    async def setup_webhook(self):
        """Set up webhook with proper update processing."""
        webhook_path = f"/webhook/{self.token}"
        webhook_url = f"{os.getenv('WEBHOOK_URL')}{webhook_path}"

        # First, delete existing webhook without dropping pending updates to preserve queued messages
        await self.application.bot.delete_webhook(drop_pending_updates=False)

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
        """Process updates with better task management."""
        try:
            # Check if this update should be ignored using the external message filter
            bot_username = getattr(
                self.application.bot, "username", "Gemini_AIAssistBot"
            )
            if message_filter.should_ignore_update(update_data, bot_username):
                self.logger.info(
                    f"Ignoring update {update_data.get('update_id', 'unknown')}"
                )
                return  # Skip processing this update

            if not self.application.running:
                await self.application.initialize()
                await self.application.start()

            # Process the update as usual
            update = Update.de_json(update_data, self.application.bot)

            # Process update in a dedicated task
            task = asyncio.create_task(self.application.process_update(update))
            # Add task to a set for tracking
            self._update_tasks.add(task)

            # Wait for the task to complete or until it's canceled
            await task

        except Exception as e:
            self.logger.error(f"Error in process_update: {str(e)}")
            # Ensure the task is properly cleaned up
            if task in self._update_tasks:
                self._update_tasks.remove(task)
                if not task.done():
                    task.cancel()
        finally:
            pass


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


def get_application():
    """Create and configure the application for uvicorn without creating a new event loop"""
    # Set the environment variable so our code knows we're using uvicorn
    os.environ["DEV_SERVER"] = "uvicorn"

    # Initialize the bot
    bot = TelegramBot()

    # Create a FastAPI app with exception handlers
    app = FastAPI()

    # Add middleware for request ID tracking and timing
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timer for performance monitoring
        start_time = time.time()

        # Add request ID to logger context
        logger_with_context = logging.LoggerAdapter(
            bot.logger, {"request_id": request_id}
        )

        # Log incoming request
        logger_with_context.info(
            f"Request started: {request.method} {request.url.path}"
        )

        try:
            # Process the request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Add custom headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id

            # Log successful response with timing
            logger_with_context.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {process_time:.3f}s"
            )

            # Record metrics for monitoring (if you have a metrics system)
            # This could be expanded to use Prometheus, StatsD, etc.
            if process_time > 1.0:
                logger_with_context.warning(
                    f"Slow request detected: {process_time:.3f}s"
                )

            return response

        except Exception as e:
            # Log exception with full details
            process_time = time.time() - start_time
            logger_with_context.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Time: {process_time:.3f}s",
                exc_info=True,
            )

            # Return proper error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,  # Include request ID for troubleshooting
                },
            )

    # Define custom exception handler for webhook errors
    class WebhookException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(self.detail)

    @app.exception_handler(WebhookException)
    async def webhook_exception_handler(request: Request, exc: WebhookException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    # Setup webhook handling without creating a new loop
    existing_loop = asyncio.get_event_loop()

    # Extract the token value for use in both URL-encoded and raw forms
    raw_token = bot.token
    # URL-encoded token is what Telegram actually sends in the webhook URL
    url_encoded_token = raw_token.replace(":", "%3A")

    # Register the webhook endpoint with BOTH the raw token and URL-encoded token paths
    # This ensures we catch the request regardless of encoding
    @app.post(f"/webhook/{raw_token}")
    @app.post(
        f"/webhook/{url_encoded_token}"
    )  # Add this path to match Telegram's URL-encoded format
    async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
        # Get request ID from state (added by middleware)
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        logger_with_context = logging.LoggerAdapter(
            bot.logger, {"request_id": request_id}
        )

        # Log the actual path for debugging
        logger_with_context.info(f"Webhook received at path: {request.url.path}")

        # Track timing for monitoring
        start_time = time.time()

        try:
            # Validate request content type
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                raise WebhookException(
                    status_code=415,
                    detail="Unsupported Media Type: Content-Type must be application/json",
                )

            # Apply rate limiting (optional)
            # This is a simple implementation - could be replaced with Redis-based solution
            client_ip = request.client.host if request.client else "unknown"
            current_rate = getattr(get_application, f"rate_{client_ip}", 0)
            if current_rate > 30:  # 30 requests per minute
                logger_with_context.warning(f"Rate limit exceeded for IP: {client_ip}")
                raise WebhookException(status_code=429, detail="Too Many Requests")

            setattr(get_application, f"rate_{client_ip}", current_rate + 1)
            background_tasks.add_task(
                lambda: setattr(get_application, f"rate_{client_ip}", current_rate)
            )

            # Extract data with timeout handling
            try:
                # Set a reasonable timeout for JSON parsing
                update_data = await asyncio.wait_for(request.json(), timeout=2.0)
            except asyncio.TimeoutError:
                raise WebhookException(
                    status_code=408,
                    detail="Request Timeout: JSON parsing took too long",
                )
            except json.JSONDecodeError:
                raise WebhookException(
                    status_code=400, detail="Bad Request: Invalid JSON format"
                )

            # Basic validation of Telegram update structure
            if not isinstance(update_data, dict):
                raise WebhookException(
                    status_code=400, detail="Bad Request: Update data must be an object"
                )

            if "update_id" not in update_data:
                raise WebhookException(
                    status_code=400, detail="Bad Request: Missing update_id field"
                )

            # Log incoming update with useful context
            update_id = update_data.get("update_id", "unknown")
            logger_with_context.info(
                f"Received webhook update {update_id} - "
                f"Type: {_get_update_type(update_data)} - "
                f"Size: {len(json.dumps(update_data))} bytes"
            )

            # Process update with retry mechanism for transient errors
            # Add to background tasks for async processing
            background_tasks.add_task(
                _process_update_with_retry, bot, update_data, logger_with_context
            )

            # Track processing time
            process_time = time.time() - start_time

            # Return immediate response with useful headers
            return JSONResponse(
                content={"status": "ok", "received_at": time.time()},
                status_code=200,
                headers={
                    "X-Process-Time": str(process_time),
                    "X-Request-ID": request_id,
                    "Connection": "keep-alive",
                },
            )

        except WebhookException as e:
            # Already formatted exceptions just get passed through
            raise

        except Exception as e:
            # Log unexpected errors
            logger_with_context.error(
                f"Webhook unexpected error: {str(e)}", exc_info=True
            )

            # Return a proper error response
            return JSONResponse(
                content={"status": "error", "detail": str(e), "request_id": request_id},
                status_code=500,
            )

    # Helper functions for webhook processing
    def _get_update_type(update_data):
        """Determine the type of Telegram update for better logging"""
        if "message" in update_data:
            if "text" in update_data["message"]:
                return "text_message"
            elif "photo" in update_data["message"]:
                return "photo_message"
            elif "voice" in update_data["message"]:
                return "voice_message"
            elif "document" in update_data["message"]:
                return "document_message"
            return "other_message"
        elif "edited_message" in update_data:
            return "edited_message"
        elif "callback_query" in update_data:
            return "callback_query"
        elif "inline_query" in update_data:
            return "inline_query"
        return "unknown"

    async def _process_update_with_retry(bot, update_data, logger):
        """Process update with retry mechanism for transient errors"""
        max_retries = 3
        base_delay = 0.5  # Start with 500ms delay

        # Special handling for updates that can be safely ignored
        # Use the external message_filter instead of the deprecated _should_ignore_update
        bot_username = getattr(bot.application.bot, "username", "Gemini_AIAssistBot")
        if message_filter.should_ignore_update(update_data, bot_username):
            logger.info(
                f"Filtering out content in update {update_data.get('update_id')}"
            )
            return

        for attempt in range(max_retries):
            try:
                # Process the update
                await bot.process_update(update_data)
                if attempt > 0:
                    # Log successful retry
                    logger.info(
                        f"Successfully processed update {update_data.get('update_id')} on attempt {attempt+1}"
                    )
                return

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                # These are transient errors, retry with backoff
                retry_delay = base_delay * (2**attempt)  # Exponential backoff

                if attempt < max_retries - 1:
                    logger.warning(
                        f"Transient error processing update {update_data.get('update_id')}, "
                        f"retrying in {retry_delay}s: {str(e)}"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to process update {update_data.get('update_id')} "
                        f"after {max_retries} attempts: {str(e)}"
                    )

            except Exception as e:
                # Non-transient errors, don't retry
                logger.error(
                    f"Error processing update {update_data.get('update_id')}: {str(e)}",
                    exc_info=True,
                )
                return

    # Create a health check endpoint with detailed status
    @app.get("/health")
    @app.head("/health")
    async def health_check():
        """Enhanced health check endpoint with detailed status information."""
        health_data = {
            "status": "ok",
            "timestamp": time.time(),
            "version": os.getenv("APP_VERSION", "1.0.0"),
            "components": {},
        }

        # Check Telegram API connection
        try:
            me = await bot.application.bot.get_me()
            health_data["components"]["telegram_api"] = {
                "status": "ok",
                "bot_username": me.username,
            }
        except Exception as e:
            health_data["status"] = "degraded"
            health_data["components"]["telegram_api"] = {
                "status": "error",
                "error": str(e),
            }

        # Check database connection
        try:
            # Just a simple ping to check connection
            ping_result = bot.db.command("ping")
            health_data["components"]["database"] = {
                "status": "ok" if ping_result.get("ok") == 1 else "error"
            }
        except Exception as e:
            health_data["status"] = "degraded"
            health_data["components"]["database"] = {"status": "error", "error": str(e)}

        # Add system metrics
        health_data["system"] = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        }

        # Set appropriate status code based on overall health
        status_code = 200 if health_data["status"] == "ok" else 503

        return JSONResponse(content=health_data, status_code=status_code)

    # Create a lifespan context manager to handle startup and shutdown events
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup logic - runs before application starts taking requests
        logger.info("Starting application with enhanced monitoring...")

        try:
            await bot.application.initialize()
            await bot.application.start()

            # If WEBHOOK_URL is set, configure webhook; otherwise, start polling fallback
            if os.getenv("WEBHOOK_URL"):
                await bot.setup_webhook()
                bot.logger.info(
                    f"Webhook endpoints registered at /webhook/{raw_token} and /webhook/{url_encoded_token}"
                )
            else:
                # Start polling in a background thread for development/local usage
                from threading import Thread

                def _polling():
                    bot.application.run_polling()

                Thread(target=_polling, daemon=True).start()
                bot.logger.info(
                    "Polling fallback started; bot will process updates via polling."
                )

            # Log successful startup
            logger.info("Application started successfully")

            yield  # Application runs and handles requests here

        except Exception as e:
            logger.error(f"Error during application startup: {e}", exc_info=True)
            raise
        finally:
            # Shutdown logic - runs after application finishes handling requests
            logger.info("Shutting down application...")

            try:
                await bot.application.stop()
                await bot.application.shutdown()
                logger.info("Application shutdown completed successfully")
            except Exception as e:
                logger.error(f"Error during application shutdown: {e}", exc_info=True)

    # Set the lifespan for the FastAPI app
    app.router.lifespan_context = lifespan

    return app


# Override module-level app with the TelegramBot-configured application
app = get_application()
