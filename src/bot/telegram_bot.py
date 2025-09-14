import os
import logging
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
from src.services.group_chat.integration import GroupChatIntegration
from src.services.mcp_bot_integration import initialize_mcp_for_bot
logger = logging.getLogger(__name__)
class TelegramBot:
    """
    TelegramBot class that handles interactions with the Telegram API.
    Manages bot initialization, message handling, and service management.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._update_tasks = set()
        self.response_cache = TTLCache(maxsize=500, ttl=3600)
        self.user_response_cache = LRUCache(maxsize=100)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")
        self.user_rate_limits = TTLCache(maxsize=10000, ttl=60)
        self.user_blocklist = set()
        self._init_db_connection()
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
        self._init_services()
        self._setup_handlers()
        self.session = None
    def is_user_blocked(self, user_id):
        """Check if user is blocked."""
        return user_id in self.user_blocklist
    def block_user(self, user_id):
        """Block a user from accessing the bot."""
        self.user_blocklist.add(user_id)
        self.logger.warning(f"Blocked user {user_id} due to suspicious activity.")
    def check_user_rate_limit(self, user_id):
        """Check and update per-user rate limit. Returns True if allowed, False if rate limited."""
        count = self.user_rate_limits.get(user_id, 0)
        if count > 30:
            self.logger.warning(
                f"User {user_id} exceeded rate limit: {count} requests/minute."
            )
            return False
        self.user_rate_limits[user_id] = count + 1
        return True
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
        """Initialize database connection with enhanced retry mechanism and fallback."""
        max_retries = int(os.getenv("DB_MAX_RETRIES", "5"))
        retry_delay = float(os.getenv("DB_RETRY_DELAY", "2.0"))
        self.logger.info(
            f"Initializing database connection with {max_retries} max retries..."
        )
        for attempt in range(max_retries):
            try:
                self.db, self.client = get_database(max_retries=2, retry_interval=3.0)
                if self.db is None:
                    raise ConnectionError("Database connection returned None")
                self.logger.info("✅ Connected to MongoDB successfully")
                try:
                    collections = self.db.list_collection_names()
                    self.logger.info(
                        f"Database collections accessible: {len(collections)} found"
                    )
                except Exception as test_error:
                    self.logger.warning(f"Database test operation failed: {test_error}")
                return
            except Exception as e:
                error_msg = str(e).lower()
                is_timeout = any(
                    keyword in error_msg for keyword in ["timeout", "timed out"]
                )
                if attempt < max_retries - 1:
                    if is_timeout:
                        self.logger.warning(
                            f"⏰ Database connection timeout on attempt {attempt + 1}/{max_retries}. "
                            f"This is common with MongoDB Atlas. Retrying in {retry_delay:.1f}s..."
                        )
                    else:
                        self.logger.warning(
                            f"🔄 Database connection attempt {attempt + 1}/{max_retries} failed: {str(e)[:150]}... "
                            f"Retrying in {retry_delay:.1f}s..."
                        )
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 10.0)
                else:
                    if is_timeout:
                        self.logger.error(
                            f"❌ Database connection failed after {max_retries} attempts due to persistent timeouts. "
                            f"Bot will continue with limited functionality. Consider checking: "
                            f"1) Internet connection, 2) MongoDB Atlas whitelist, 3) Connection string validity"
                        )
                    else:
                        self.logger.error(
                            f"❌ All database connection attempts failed: {e}"
                        )
                    self.db = None
                    self.client = None
                    self.logger.warning(
                        "🚨 Bot starting in degraded mode without database persistence"
                    )
    def _init_services(self):
        """Initialize bot services and API clients."""
        try:
            self.logger.info("MCP integration will be initialized when bot starts...")
            self._init_model_apis()
            self._init_utility_classes()
            self._init_handlers()
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            raise
    def _init_model_apis(self):
        """Initialize AI model APIs."""
        rate_limiter = RateLimiter(requests_per_minute=30)
        self.gemini_api = GeminiAPI(rate_limiter=rate_limiter)
        openrouter_rate_limiter = RateLimiter(requests_per_minute=20)
        self.openrouter_api = OpenRouterAPI(rate_limiter=openrouter_rate_limiter)
        self.deepseek_api = DeepSeekLLM()
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
        self.user_data_manager = UserDataManager(self.db)
        self.telegram_logger = telegram_logger
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.media_context_extractor = MediaContextExtractor()
        self.image_processor = ImageProcessor(self.gemini_api)
        self.voice_processor = VoiceProcessor()
        self.prompt_formatter = PromptFormatter()
        self.preferences_manager = UserPreferencesManager(self.user_data_manager)
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
        self.text_handler = TextHandler(
            gemini_api=self.gemini_api,
            user_data_manager=self.user_data_manager,
            openrouter_api=self.openrouter_api,
            deepseek_api=self.deepseek_api,
        )
        from src.services.memory_context.conversation_manager import ConversationManager
        self.conversation_manager = ConversationManager(
            self.text_handler.memory_manager, self.text_handler.model_history_manager
        )
        self.application.bot_data["conversation_manager"] = self.conversation_manager
        self.command_handler = CommandHandlers(
            gemini_api=self.gemini_api,
            user_data_manager=self.user_data_manager,
            telegram_logger=self.telegram_logger,
            flux_lora_image_generator=flux_lora_image_generator,
            deepseek_api=self.deepseek_api,
            openrouter_api=self.openrouter_api,
        )
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
        self.message_handlers.context_handler = self.context_handler
        self.message_handlers.response_formatter = self.response_formatter
        self.message_handlers.voice_processor = self.voice_processor
        self.message_handlers.preferences_manager = self.preferences_manager
        self.message_handlers._conversation_manager = self.conversation_manager
        self.message_handlers.document_processor = self.document_processor
        self.reminder_manager = ReminderManager(self.application.bot)
        self.language_manager = LanguageManager()
        self.group_chat_integration = GroupChatIntegration(
            self.user_data_manager, self.conversation_manager
        )
        self.application.bot_data["group_chat_integration"] = (
            self.group_chat_integration
        )
        self.message_handlers._group_chat_integration = self.group_chat_integration
    def get_message_handlers(self):
        """Get the message handlers instance for Web App API access."""
        return getattr(self, "message_handlers", None)
    async def shutdown(self):
        """Properly clean up resources on shutdown."""
        if self.session and not self.session.closed:
            await self.session.close()
        close_database_connection(self.client)
        logger.info("Shutdown complete. Database connection closed.")
    def _setup_handlers(self):
        """Register handlers with the application."""
        self.response_cache = TTLCache(maxsize=1000, ttl=300)
        self.command_handler.register_handlers(
            self.application, cache=self.response_cache
        )
        self.message_handlers.register_handlers(self.application)
        async def remind_handler(update: Update, context):
            user = update.effective_user
            user_id = user.id if user is not None else None
            args = context.args if hasattr(context, "args") else []
            message_obj = update.effective_message
            if user_id is None:
                if message_obj:
                    await message_obj.reply_text(
                        "User ID not found. Cannot set reminder."
                    )
                return
            if len(args) < 2:
                if message_obj:
                    await message_obj.reply_text("Usage: /remind <time> <message>")
                return
            from datetime import datetime
            time_str = args[0]
            message = " ".join(args[1:])
            try:
                remind_time = datetime.fromisoformat(time_str)
            except Exception:
                if message_obj:
                    await message_obj.reply_text(
                        "Invalid time format. Use YYYY-MM-DDTHH:MM"
                    )
                return
            await self.reminder_manager.set_reminder(user_id, remind_time, message)
            if message_obj:
                await message_obj.reply_text(
                    f"Reminder set for {remind_time}: {message}"
                )
        self.application.add_handler(CommandHandler("remind", remind_handler))
        self.application.add_handler(
            CommandHandler("language", self.language_manager.set_language)
        )
        self.application.error_handlers.clear()
        async def error_handler(update_or_obj, context):
            if hasattr(update_or_obj, "message") or hasattr(update_or_obj, "update_id"):
                await self.message_handlers._error_handler(update_or_obj, context)
            else:
                self.logger.error(
                    f"Error handler received non-Update object: {update_or_obj}"
                )
        self.application.add_error_handler(error_handler)
    async def setup_webhook(self):
        """Set up webhook with proper update processing."""
        await initialize_mcp_for_bot()
        webhook_path = f"/webhook/{self.token}"
        webhook_url = f"{os.getenv('WEBHOOK_URL')}{webhook_path}"
        await self.application.bot.delete_webhook(drop_pending_updates=False)
        webhook_config = {
            "url": webhook_url,
            "allowed_updates": [
                "message",
                "edited_message",
                "callback_query",
                "inline_query",
            ],
            "max_connections": 500,
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
        else:
            self.logger.info("Application is already running. Skipping start.")
    async def process_update(self, update_data):
        """Process updates with task management."""
        try:
            if isinstance(update_data, dict):
                update_data_dict = update_data
            else:
                self.logger.error(f"Unsupported update_data type: {type(update_data)}")
                return
            update = Update.de_json(update_data_dict, self.application.bot)
            if update is None:
                self.logger.warning(f"Failed to parse update: {update_data}")
                return
            await self.application.process_update(update)
            self.logger.debug(f"Successfully processed update {update.update_id}")
        except Exception as e:
            update_id = "unknown"
            try:
                if isinstance(update_data, dict):
                    update_id = update_data.get("update_id", "unknown")
                elif hasattr(update_data, "update_id"):
                    update_id = update_data.update_id
            except (AttributeError, KeyError, TypeError):
                pass
            self.logger.error(
                f"Error processing update {update_id}: {str(e)}",
                exc_info=True,
            )
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
