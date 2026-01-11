import logging
import asyncio
from telegram import Update
from telegram.ext import ContextTypes
from src.services.gemini_api import GeminiAPI
from src.services.user_data_manager import UserDataManager
from src.utils.message_utils import extract_reply_context
from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager
from src.services.memory_context.conversation_manager import ConversationManager
from src.handlers.response_formatter import ResponseFormatter
from src.services.model_handlers.prompt_formatter import PromptFormatter
from src.handlers.text_processing.media_analyzer import MediaAnalyzer
from src.utils.bot_username_helper import BotUsernameHelper
from .media_helpers import MediaHelpers
from .conversation_logic import ConversationLogic

class TextHandler:
    def __init__(
        self,
        gemini_api: GeminiAPI,
        user_data_manager: UserDataManager,
        openrouter_api=None,
        deepseek_api=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.openrouter_api = openrouter_api
        self.deepseek_api = deepseek_api
        
        # Core Managers
        self.memory_manager = MemoryManager(
            db=user_data_manager.db if hasattr(user_data_manager, "db") else None,
        )
        self.memory_manager.short_term_limit = 15
        self.memory_manager.token_limit = 64000
        self.model_history_manager = ModelHistoryManager(self.memory_manager)
        
        self.response_formatter = ResponseFormatter()
        self.prompt_formatter = PromptFormatter()
        self.conversation_manager = ConversationManager(
            self.memory_manager, self.model_history_manager
        )
        self.media_analyzer = MediaAnalyzer(gemini_api, openrouter_api)
        
        # Initialize Logic Component
        self.conversation_logic = ConversationLogic(
            self.logger,
            self.gemini_api,
            self.user_data_manager,
            self.response_formatter,
            self.prompt_formatter,
            self.conversation_manager,
            self.memory_manager,
            self.media_analyzer,
            self.openrouter_api,
            self.deepseek_api
        )

        self._group_chat_integration = None

    async def handle_text_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Main handler for text messages.
        Delegates processing to ConversationLogic and MediaHelpers.
        """
        if not update.message and not update.edited_message:
            return
            
        user_id = update.effective_user.id
        message = update.message or update.edited_message
        message_text = message.text
        chat = update.effective_chat
        
        # Group Chat Logic (simplified integration)
        if chat and chat.type in ["group", "supergroup"]:
            if hasattr(self, "_group_chat_integration") and self._group_chat_integration:
                enhanced_message = await self._group_chat_integration.process_message(
                    update, context
                )
                if enhanced_message and enhanced_message.get("enhanced_text"):
                    message_text = enhanced_message["enhanced_text"]

        quoted_text, quoted_message_id = extract_reply_context(message)
        
        try:
            # Handle Edited Messages
            if update.edited_message and "bot_messages" in context.user_data:
                await self._handle_edited_message(update, context)

            # Bot Mention Check for Groups
            if chat.type in ["group", "supergroup"]:
                if not BotUsernameHelper.is_bot_mentioned(message_text, context, message.entities):
                    return
                message_text = BotUsernameHelper.remove_bot_mention(message_text, context)

            # Media Extraction
            has_attached_media, media_files, media_type = await MediaHelpers.extract_media_files(update, context)
            if media_type == "media_group_start":
                # Logic handled in extract_media_files via async task creation usually, 
                # but here we rely on the specific implementation in media_helpers
                # or we trigger the processor if needed.
                asyncio.create_task(
                    self.conversation_logic.process_complete_media_group(
                        update.message.media_group_id,
                        update.effective_chat.id,
                        update.effective_user.id,
                        update.message.caption or "",
                        context
                    )
                )
                return

            thinking_message = await message.reply_text("Processing your request...🧠")
            await MediaHelpers.send_appropriate_chat_action(
                update, context, has_attached_media, media_type
            )

            # User Prep
            preferred_model = await self._get_user_preferred_model(user_id)
            await self.memory_manager.extract_and_save_user_info(user_id, message_text)
            
            history_context = await self.conversation_manager.get_conversation_history(
                user_id, max_messages=15, model=preferred_model
            )

            if has_attached_media:
                await self.conversation_logic.handle_media_analysis(
                    update, context, thinking_message, media_files, media_type,
                    message_text, user_id, preferred_model
                )
            else:
                await self.conversation_logic.handle_text_conversation(
                    update, context, thinking_message, message_text,
                    quoted_text, quoted_message_id, history_context,
                    user_id, preferred_model
                )

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            if 'thinking_message' in locals() and thinking_message:
                await thinking_message.delete()
            await self.response_formatter.safe_send_message(
                update.message, "Sorry, I encountered an error. Please try again later."
            )

    async def _handle_edited_message(self, update, context):
        """Handle when a user edits their previous message"""
        # Kept simple here as it's specific to this handler's state usage
        original_message_id = update.edited_message.message_id
        if original_message_id in context.user_data.get("bot_messages", {}):
            for msg_id in context.user_data["bot_messages"][original_message_id]:
                if msg_id:
                    try:
                        await context.bot.delete_message(
                            chat_id=update.effective_chat.id, message_id=msg_id
                        )
                    except Exception:
                        pass
            if original_message_id in context.user_data["bot_messages"]:
                del context.user_data["bot_messages"][original_message_id]

    async def _get_user_preferred_model(self, user_id):
        # Quick helper proxy
        from src.services.user_preferences_manager import UserPreferencesManager
        preferences_manager = UserPreferencesManager(self.user_data_manager)
        return await preferences_manager.get_user_model_preference(user_id)
