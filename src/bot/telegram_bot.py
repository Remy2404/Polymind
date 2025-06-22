import os
import logging
import asyncio
import time
import aiohttp
import traceback
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
)
from cachetools import TTLCache, LRUCache

from src.database.connection import get_database, close_database_connection
from src.services.user_data_manager import UserDataManager
from src.services.gemini_api import GeminiAPI
from src.services.openrouter_api import OpenRouterAPI
from src.handlers.command_handlers import CommandHandlers
from src.handlers.text_handlers import TextHandler
from src.services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM
from src.handlers.message_handlers import MessageHandlers
from src.utils.log.telegramlog import telegram_logger
from src.services.reminder_manager import ReminderManager
from src.utils.lang.language_manager import LanguageManager
from src.services.rate_limiter import RateLimiter
from src.services.flux_lora_img import flux_lora_image_generator
from src.utils.docgen.document_processor import DocumentProcessor
from src.utils.ignore_message import message_filter
from src.services.group_chat.integration import GroupChatIntegration

logger = logging.getLogger(__name__)


# Use the proper GroupChatIntegration implementation from integration.py


class TelegramBot:
    """
    TelegramBot class that handles interactions with the Telegram API.
    Manages bot initialization, message handling, and service management.
    """

    def __init__(self):
        # Initialize essential services at startup
        self.logger = logging.getLogger(__name__)

        # Track active update processing tasks
        self._update_tasks = set()

        # Efficient caching strategy
        self.response_cache = TTLCache(maxsize=500, ttl=3600)
        self.user_response_cache = LRUCache(maxsize=100)

        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

        # Initialize database connection with retry mechanism
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

        # Initialize services and handlers
        self._init_services()
        self._setup_handlers()

        # Create client session for HTTP requests
        self.session = None

    async def create_session(self):
        """Create an aiohttp session for HTTP requests."""
        try:
            if self.session is None or self.session.closed:
                tcp_connector = aiohttp.TCPConnector(
                    limit=150,
                    limit_per_host=500,
                    force_close=False,
                    enable_cleanup_closed=True,
                )
                self.session = aiohttp.ClientSession(
                    connector=tcp_connector,
                    timeout=aiohttp.ClientTimeout(total=300, connect=500),
                )
                self.logger.info("Created new aiohttp session for bot")
            return self.session
        except Exception as e:
            self.logger.error(f"Failed to create aiohttp session: {str(e)}")
            raise

    def _init_db_connection(self):
        """Initialize database connection with retry mechanism."""
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
        """Initialize bot services and API clients."""
        try:
            # Initialize model APIs
            self._init_model_apis()

            # Initialize utility classes
            self._init_utility_classes()

            # Initialize handlers
            self._init_handlers()

        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            raise

    def _init_model_apis(self):
        """Initialize AI model APIs."""
        # Gemini API
        rate_limiter = RateLimiter(requests_per_minute=30)
        self.gemini_api = GeminiAPI(rate_limiter=rate_limiter)

        # OpenRouter API
        openrouter_rate_limiter = RateLimiter(requests_per_minute=20)
        self.openrouter_api = OpenRouterAPI(rate_limiter=openrouter_rate_limiter)

        # DeepSeek API
        self.deepseek_api = DeepSeekLLM()

        # Store API instances in application context
        if not hasattr(self.application, "bot_data"):
            self.application.bot_data = {}
        self.application.bot_data["gemini_api"] = self.gemini_api
        self.application.bot_data["openrouter_api"] = self.openrouter_api
        self.application.bot_data["deepseek_api"] = self.deepseek_api

    def _init_utility_classes(self):
        """Initialize utility classes for message handling."""
        from src.handlers.message_context_handler import MessageContextHandler
        from src.handlers.response_formatter import ResponseFormatter
        from src.handlers.media_context_extractor import MediaContextExtractor
        from src.services.media.image_processor import ImageProcessor
        from src.services.media.voice_processor import VoiceProcessor
        from src.services.model_handlers.prompt_formatter import PromptFormatter
        from src.services.user_preferences_manager import UserPreferencesManager

        # User data management
        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger

        # Create utility instances
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.media_context_extractor = MediaContextExtractor()
        self.image_processor = ImageProcessor(self.gemini_api)
        self.voice_processor = VoiceProcessor()
        self.prompt_formatter = PromptFormatter()
        self.preferences_manager = UserPreferencesManager(self.user_data_manager)

        # Store in application context
        self.application.bot_data["context_handler"] = self.context_handler
        self.application.bot_data["response_formatter"] = self.response_formatter
        self.application.bot_data["media_context_extractor"] = (
            self.media_context_extractor
        )
        self.application.bot_data["image_processor"] = self.image_processor
        self.application.bot_data["voice_processor"] = self.voice_processor
        self.application.bot_data["prompt_formatter"] = self.prompt_formatter
        self.application.bot_data["preferences_manager"] = self.preferences_manager

    def _init_handlers(self):
        """Initialize message and command handlers."""
        # Initialize TextHandler
        self.text_handler = TextHandler(
            gemini_api=self.gemini_api,
            user_data_manager=self.user_data_manager,
            openrouter_api=self.openrouter_api,
            deepseek_api=self.deepseek_api,
        )

        # Initialize ConversationManager
        from src.services.memory_context.conversation_manager import ConversationManager

        self.conversation_manager = ConversationManager(
            self.text_handler.memory_manager, self.text_handler.model_history_manager
        )
        self.application.bot_data["conversation_manager"] = self.conversation_manager

        # Initialize CommandHandler
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api,
            user_data_manager=self.user_data_manager,
            telegram_logger=self.telegram_logger,
            flux_lora_image_generator=flux_lora_image_generator,
            deepseek_api=self.deepseek_api,
            openrouter_api=self.openrouter_api,
        )

        # Initialize DocumentProcessor and MessageHandlers
        self.document_processor = DocumentProcessor(gemini_api=self.gemini_api)
        self.message_handlers = MessageHandlers(
            self.gemini_api,
            self.user_data_manager,
            self.telegram_logger,
            self.text_handler,
            deepseek_api=self.deepseek_api,
            openrouter_api=self.openrouter_api,
            command_handlers=self.command_handler,
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
        self.message_handlers.document_processor = self.document_processor

        # Initialize other services
        self.reminder_manager = ReminderManager(self.application.bot)
        self.language_manager = LanguageManager()

        # Initialize GroupChatIntegration

        self.group_chat_integration = GroupChatIntegration(
            self.user_data_manager, self.conversation_manager
        )
        self.application.bot_data["group_chat_integration"] = (
            self.group_chat_integration
        )

        # Share group chat integration with message handlers
        self.message_handlers._group_chat_integration = self.group_chat_integration

    async def shutdown(self):
        """Properly clean up resources on shutdown."""
        # Close aiohttp session if it exists
        if self.session and not self.session.closed:
            await self.session.close()

        # Close database connection
        close_database_connection(self.client)
        logger.info("Shutdown complete. Database connection closed.")

    def _setup_handlers(self):
        """Register handlers with the application."""
        # Create a response cache with optimized settings
        self.response_cache = TTLCache(maxsize=1000, ttl=300)

        # Register command handlers
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

        # Set up error handler
        self.application.error_handlers.clear()
        self.application.add_error_handler(self.message_handlers._error_handler)

    async def setup_webhook(self):
        """Set up webhook with proper update processing."""
        webhook_path = f"/webhook/{self.token}"
        webhook_url = f"{os.getenv('WEBHOOK_URL')}{webhook_path}"

        # Delete existing webhook without dropping pending updates
        await self.application.bot.delete_webhook(drop_pending_updates=False)

        # Webhook configuration
        webhook_config = {
            "url": webhook_url,
            "allowed_updates": [
                "message",
                "edited_message",
                "callback_query",
                "inline_query",
            ],
            # Increase max_connections for better throughput
            "max_connections": 500,
        }

        self.logger.info(f"Setting webhook to: {webhook_url}")

        if not self.application.running:
            await self.application.initialize()
            await self.application.start()

        # Set up webhook with new configuration
        await self.application.bot.set_webhook(**webhook_config)

        # Log webhook info
        webhook_info = await self.application.bot.get_webhook_info()
        self.logger.info(f"Webhook status: {webhook_info}")

        # Start application if not running
        if not self.application.running:
            await self.application.start()
        else:
            self.logger.info("Application is already running. Skipping start.")

    async def process_update(self, update_data):
        """Process updates with task management."""
        task = None
        try:
            # Expect update_data as dict
            if isinstance(update_data, dict):
                update_data_dict = update_data
            else:
                self.logger.error(f"Unsupported update_data type: {type(update_data)}")
                return

            # Convert dict to Update object
            update = Update.de_json(update_data_dict, self.application.bot)

            if update is None:
                self.logger.warning(f"Failed to parse update: {update_data}")
                return

            # Use the application's update processor
            await self.application.process_update(update)

            self.logger.debug(f"Successfully processed update {update.update_id}")

        except Exception as e:
            # Safe error logging that doesn't rely on .get() method
            update_id = "unknown"
            try:
                if hasattr(update_data, "update_id"):
                    update_id = update_data.update_id
                elif isinstance(update_data, dict):
                    update_id = update_data.get('update_id', 'unknown')
            except:
                pass
                
            self.logger.error(
                f"Error processing update {update_id}: {str(e)}",
                exc_info=True,
            )
            # Log full traceback for debugging
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
