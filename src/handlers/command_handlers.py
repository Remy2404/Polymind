import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telegram import (
    Update,
)
from telegram.ext import (
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
    Application,
)
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from services.model_handlers.simple_api_manager import SuperSimpleAPIManager
from utils.log.telegramlog import TelegramLogger as telegram_logger
import logging
from services.flux_lora_img import (
    FluxLoraImageGenerator as flux_lora_image_generator,
)
import time, asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cachetools import TTLCache
from typing import Optional
from telegram.constants import ChatAction
from PIL import Image
import io

# Import all command modules
from .commands import (
    BasicCommands,
    SettingsCommands,
    ImageCommands,
    ModelCommands,
    DocumentCommands,
    ExportCommands,
    CallbackHandlers,
    OpenWebAppCommands,
)
from src.services.group_chat.integration import GroupChatIntegration


@dataclass
class ImageRequest:
    prompt: str
    width: int
    height: int
    steps: int
    timestamp: float = field(default_factory=time.time)


class ImageGenerationHandler:
    def __init__(self):
        self.request_cache = TTLCache(maxsize=100, ttl=3600)
        self.request_limiter = {}
        self.processing_queue = asyncio.Queue()
        self.rate_limit_time = 30

    def is_rate_limited(self, user_id: int) -> bool:
        if user_id in self.request_limiter:
            last_request = self.request_limiter[user_id]
            if datetime.now() - last_request < timedelta(seconds=self.rate_limit_time):
                return True
        return False

    def update_rate_limit(self, user_id: int) -> None:
        self.request_limiter[user_id] = datetime.now()

    def get_cached_image(
        self, prompt: str, width: int, height: int, steps: int
    ) -> Optional[Image.Image]:
        cache_key = f"{prompt}_{width}_{height}_{steps}"
        return self.request_cache.get(cache_key)

    def cache_image(
        self, prompt: str, width: int, height: int, steps: int, image: Image.Image
    ) -> None:
        cache_key = f"{prompt}_{width}_{height}_{steps}"
        self.request_cache[cache_key] = image


class CommandHandlers:
    def __init__(
        self,
        gemini_api: GeminiAPI,
        user_data_manager: UserDataManager,
        telegram_logger: telegram_logger,
        flux_lora_image_generator: flux_lora_image_generator,
        deepseek_api=None,
        openrouter_api=None,
    ):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.flux_lora_image_generator = flux_lora_image_generator
        self.logger = logging.getLogger(__name__)
        self.telegram_logger = telegram_logger
        self.image_handler = ImageGenerationHandler()
        self.deepseek_api = deepseek_api
        self.openrouter_api = openrouter_api

        # Create SuperSimpleAPIManager for model handling
        self.api_manager = SuperSimpleAPIManager(
            gemini_api, deepseek_api, openrouter_api
        )

        # Initialize command modules
        self.basic_commands = BasicCommands(user_data_manager, telegram_logger)
        self.settings_commands = SettingsCommands(user_data_manager, telegram_logger)
        self.image_commands = ImageCommands(
            flux_lora_image_generator,
            user_data_manager,
            telegram_logger,
            self.image_handler,
        )
        self.model_commands = ModelCommands(self.api_manager, user_data_manager)
        self.document_commands = DocumentCommands(
            gemini_api, user_data_manager, telegram_logger, self.api_manager
        )
        self.export_commands = ExportCommands(
            gemini_api, user_data_manager, telegram_logger
        )
        self.open_web_app_commands = OpenWebAppCommands(
            user_data_manager, telegram_logger
        )

        # Initialize callback handlers
        self.callback_handlers = CallbackHandlers(
            self.document_commands, self.model_commands, self.export_commands
        )

    # Delegate basic commands
    async def start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.basic_commands.start_command(update, context)

    async def help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.basic_commands.help_command(update, context)

    async def reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.basic_commands.reset_command(update, context)

    # Delegate settings commands
    async def settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.settings_commands.settings(update, context)

    async def handle_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.settings_commands.handle_stats(update, context)

    async def handle_preferences(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.settings_commands.handle_preferences(update, context)

    # Delegate image commands
    async def generate_image_advanced(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.image_commands.generate_image_advanced(update, context)

    async def generate_together_image(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.image_commands.generate_together_image(update, context)

    # Delegate model commands
    async def switch_model_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.model_commands.switchmodel_command(update, context)

    async def list_models_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.model_commands.list_models_command(update, context)

    async def current_model_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.model_commands.current_model_command(update, context)

    # Delegate document commands
    async def generate_ai_document_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.document_commands.generate_ai_document_command(
            update, context
        )

    # Delegate export commands
    async def export_to_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.export_commands.export_to_document(update, context)

    async def handle_export(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.export_commands.handle_export(update, context)

    async def handle_export_conversation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.export_commands.handle_export_conversation(
            update, context
        )  # Main callback query handler - delegate to central callback handler

    async def handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        data = query.data

        # Handle basic callbacks that don't need special routing
        if data == "help":
            await self.help_command(update, context)
        elif data == "preferences":
            await self.handle_preferences(update, context)
        elif data == "settings":
            await self.settings(update, context)
        elif data in ["toggle_markdown", "toggle_code_suggestions"]:
            await self.settings_commands.handle_toggle_settings(update, context, data)
        elif data.startswith("img_"):
            await self.image_commands.handle_image_settings(update, context, data)
        elif data.startswith("pref_"):
            await self.settings_commands.handle_user_preferences(update, context, data)
        # Route hierarchical model selection callbacks to CallbackHandlers
        elif data.startswith(("category_", "model_")) or data in (
            "back_to_categories",
            "current_model",
        ):
            await self.callback_handlers.handle_callback_query(update, context)
        elif data.startswith(("aidoc_type_", "aidoc_format_", "aidoc_model_")):
            await self.document_commands.handle_ai_document_callback(
                update, context, data
            )
        elif data.startswith("export_format_"):
            document_format = data.replace("export_format_", "")
            await self.export_commands.generate_document(
                update, context, document_format
            )
        elif data == "export_conversation":
            await self.export_commands.handle_export_conversation(update, context)
        elif data == "export_custom":
            await query.edit_message_text(
                "Please send the text you want to convert to a document. You can include markdown formatting."
            )
            context.user_data["awaiting_doc_text"] = True
        elif data == "export_cancel":
            await query.edit_message_text("Document export cancelled.")
        else:
            # Route to central callback handler for more complex routing
            await self.callback_handlers.handle_callback_query(update, context)

    async def group_stats_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /groupstats command for group analytics."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get analytics
                analytics = await group_integration.group_manager.get_group_analytics(
                    chat.id
                )
                if not analytics:
                    await update.message.reply_text(
                        "No analytics available for this group yet. Try interacting more!"
                    )
                    return

                # Format stats without markdown to avoid escaping issues
                formatted_stats = (
                    await group_integration.ui_manager.format_group_analytics(analytics)
                )

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(formatted_stats)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_stats_command: {e}")
                await update.message.reply_text(
                    "❌ Group statistics feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_stats_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving group statistics. Please try again."
            )

    async def group_settings_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /groupsettings command for group configuration."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get settings menu
                settings_menu = await group_integration.ui_manager.create_settings_menu(
                    chat.id
                )

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(settings_menu)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_settings_command: {e}")
                await update.message.reply_text(
                    "❌ Settings feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_settings_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving group settings. Please try again."
            )

    async def group_context_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /groupcontext command to show shared memory."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get group context
                group_context = (
                    await group_integration.group_manager._get_or_create_group_context(
                        chat, update.effective_user
                    )
                )

                # Format shared memory
                if group_context.shared_memory:
                    context_text = "🧠 Group Shared Memory:\n\n"
                    for key, value in group_context.shared_memory.items():
                        context_text += f"• {key}: {value}\n"
                else:
                    context_text = "🧠 Group Shared Memory is empty\n\nAs the conversation continues, important information will be automatically stored here for future reference."

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(context_text)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_context_command: {e}")
                await update.message.reply_text(
                    "❌ Memory feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_context_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving group context. Please try again."
            )

    async def group_threads_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /groupthreads command to list active conversation threads."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get group context
                group_context = (
                    await group_integration.group_manager._get_or_create_group_context(
                        chat, update.effective_user
                    )
                )

                # Format threads
                if group_context.threads:
                    formatted_threads = (
                        await group_integration.ui_manager.format_thread_list(
                            group_context.threads
                        )
                    )
                else:
                    formatted_threads = "🧵 No active conversation threads\n\nThreads are created automatically when users reply to messages. Start a discussion by replying to a message!"

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(formatted_threads)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_threads_command: {e}")
                await update.message.reply_text(
                    "❌ Thread feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_threads_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving conversation threads. Please try again."
            )

    async def clean_threads_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /cleanthreads command to clean up inactive threads."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Check if user is admin
            user_member = await context.bot.get_chat_member(
                chat.id, update.effective_user.id
            )
            if user_member.status not in ["administrator", "creator"]:
                await update.message.reply_text(
                    "❌ Only group administrators can clean conversation threads."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Clean threads
                cleaned_count = (
                    await group_integration.group_manager.cleanup_inactive_threads(
                        chat.id
                    )
                )

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(
                    f"🧹 Thread Cleanup Complete\n\nRemoved {cleaned_count} inactive conversation threads."
                )

            except AttributeError as e:
                self.logger.error(f"Missing attribute in clean_threads_command: {e}")
                await update.message.reply_text(
                    "❌ Thread cleanup feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in clean_threads_command: {e}")
            await update.message.reply_text(
                "❌ Error cleaning conversation threads. Please try again."
            )

    # Delegate web app commands
    async def open_web_app_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.open_web_app_commands.open_web_app_command(update, context)

    # Delegate export commands
    async def export_to_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.export_commands.export_to_document(update, context)

    async def handle_export(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.export_commands.handle_export(update, context)

    async def handle_export_conversation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        return await self.export_commands.handle_export_conversation(
            update, context
        )  # Main callback query handler - delegate to central callback handler

    async def handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        data = query.data

        # Handle basic callbacks that don't need special routing
        if data == "help":
            await self.help_command(update, context)
        elif data == "preferences":
            await self.handle_preferences(update, context)
        elif data == "settings":
            await self.settings(update, context)
        elif data in ["toggle_markdown", "toggle_code_suggestions"]:
            await self.settings_commands.handle_toggle_settings(update, context, data)
        elif data.startswith("img_"):
            await self.image_commands.handle_image_settings(update, context, data)
        elif data.startswith("pref_"):
            await self.settings_commands.handle_user_preferences(update, context, data)
        # Route hierarchical model selection callbacks to CallbackHandlers
        elif data.startswith(("category_", "model_")) or data in (
            "back_to_categories",
            "current_model",
        ):
            await self.callback_handlers.handle_callback_query(update, context)
        elif data.startswith(("aidoc_type_", "aidoc_format_", "aidoc_model_")):
            await self.document_commands.handle_ai_document_callback(
                update, context, data
            )
        elif data.startswith("export_format_"):
            document_format = data.replace("export_format_", "")
            await self.export_commands.generate_document(
                update, context, document_format
            )
        elif data == "export_conversation":
            await self.export_commands.handle_export_conversation(update, context)
        elif data == "export_custom":
            await query.edit_message_text(
                "Please send the text you want to convert to a document. You can include markdown formatting."
            )
            context.user_data["awaiting_doc_text"] = True
        elif data == "export_cancel":
            await query.edit_message_text("Document export cancelled.")
        else:
            # Route to central callback handler for more complex routing
            await self.callback_handlers.handle_callback_query(update, context)

    async def group_stats_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /groupstats command for group analytics."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get analytics
                analytics = await group_integration.group_manager.get_group_analytics(
                    chat.id
                )
                if not analytics:
                    await update.message.reply_text(
                        "No analytics available for this group yet. Try interacting more!"
                    )
                    return

                # Format stats without markdown to avoid escaping issues
                formatted_stats = (
                    await group_integration.ui_manager.format_group_analytics(analytics)
                )

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(formatted_stats)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_stats_command: {e}")
                await update.message.reply_text(
                    "❌ Group statistics feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_stats_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving group statistics. Please try again."
            )

    async def group_settings_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /groupsettings command for group configuration."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get settings menu
                settings_menu = await group_integration.ui_manager.create_settings_menu(
                    chat.id
                )

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(settings_menu)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_settings_command: {e}")
                await update.message.reply_text(
                    "❌ Settings feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_settings_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving group settings. Please try again."
            )

    async def group_context_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /groupcontext command to show shared memory."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get group context
                group_context = (
                    await group_integration.group_manager._get_or_create_group_context(
                        chat, update.effective_user
                    )
                )

                # Format shared memory
                if group_context.shared_memory:
                    context_text = "🧠 Group Shared Memory:\n\n"
                    for key, value in group_context.shared_memory.items():
                        context_text += f"• {key}: {value}\n"
                else:
                    context_text = "🧠 Group Shared Memory is empty\n\nAs the conversation continues, important information will be automatically stored here for future reference."

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(context_text)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_context_command: {e}")
                await update.message.reply_text(
                    "❌ Memory feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_context_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving group context. Please try again."
            )

    async def group_threads_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /groupthreads command to list active conversation threads."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get group context
                group_context = (
                    await group_integration.group_manager._get_or_create_group_context(
                        chat, update.effective_user
                    )
                )

                # Format threads
                if group_context.threads:
                    formatted_threads = (
                        await group_integration.ui_manager.format_thread_list(
                            group_context.threads
                        )
                    )
                else:
                    formatted_threads = "🧵 No active conversation threads\n\nThreads are created automatically when users reply to messages. Start a discussion by replying to a message!"

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(formatted_threads)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_threads_command: {e}")
                await update.message.reply_text(
                    "❌ Thread feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_threads_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving conversation threads. Please try again."
            )

    async def clean_threads_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /cleanthreads command to clean up inactive threads."""
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Check if user is admin
            user_member = await context.bot.get_chat_member(
                chat.id, update.effective_user.id
            )
            if user_member.status not in ["administrator", "creator"]:
                await update.message.reply_text(
                    "❌ Only group administrators can clean conversation threads."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Clean threads
                cleaned_count = (
                    await group_integration.group_manager.cleanup_inactive_threads(
                        chat.id
                    )
                )

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(
                    f"🧹 Thread Cleanup Complete\n\nRemoved {cleaned_count} inactive conversation threads."
                )

            except AttributeError as e:
                self.logger.error(f"Missing attribute in clean_threads_command: {e}")
                await update.message.reply_text(
                    "❌ Thread cleanup feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in clean_threads_command: {e}")
            await update.message.reply_text(
                "❌ Error cleaning conversation threads. Please try again."
            )

    def register_handlers(self, application: Application, cache=None) -> None:
        try:
            # Command handlers
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("reset", self.reset_command))
            application.add_handler(CommandHandler("settings", self.settings))
            application.add_handler(CommandHandler("stats", self.handle_stats))
            application.add_handler(CommandHandler("export", self.handle_export))
            application.add_handler(
                CommandHandler("preferences", self.handle_preferences)
            )
            application.add_handler(
                CommandHandler("imagen3", self.generate_image_advanced)
            )
            application.add_handler(
                CommandHandler("genimg", self.generate_together_image)
            )
            application.add_handler(
                CommandHandler("switchmodel", self.switch_model_command)
            )
            application.add_handler(
                CommandHandler("listmodels", self.list_models_command)
            )
            application.add_handler(
                CommandHandler("currentmodel", self.current_model_command)
            )
            application.add_handler(
                CommandHandler("exportdoc", self.export_to_document)
            )
            application.add_handler(
                CommandHandler("gendoc", self.generate_ai_document_command)
            )

            # Web App commands
            application.add_handler(
                CommandHandler("webapp", self.open_web_app_command)
            )

            # Group chat commands
            application.add_handler(
                CommandHandler("groupstats", self.group_stats_command)
            )
            application.add_handler(
                CommandHandler("groupsettings", self.group_settings_command)
            )
            application.add_handler(
                CommandHandler("groupcontext", self.group_context_command)
            )
            application.add_handler(
                CommandHandler("groupthreads", self.group_threads_command)
            )
            application.add_handler(
                CommandHandler("cleanthreads", self.clean_threads_command)
            )

            # Specific callback handlers if needed
            self.response_cache = cache

            # General callback handler LAST (handles all callbacks including model selection)
            application.add_handler(CallbackQueryHandler(self.handle_callback_query))

            self.logger.info("Modular command handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register command handlers: {e}")
            raise
