"""
Text Handler - Main handler for text messages.

This is the facade class combining all text handling functionality.
"""
import logging
import asyncio
from telegram import Update
from telegram.ext import ContextTypes

from src.api.llm.gemini_api import GeminiAPI
from src.services.user_data_manager import UserDataManager
from src.handlers.message_context_handler import MessageContextHandler
from src.handlers.response_formatter import ResponseFormatter
from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager
from src.services.model_handlers.prompt_formatter import PromptFormatter
from src.services.memory_context.conversation_manager import ConversationManager
from src.handlers.text_processing.media_analyzer import MediaAnalyzer
from src.handlers.document_sender import DocumentSender
from src.services.memory_context.personalized_rag_system import PersonalizedRAGSystem
from src.services.memory_context.rag_integration import RAGIntegration
from src.utils.bot_username_helper import BotUsernameHelper

from src.handlers.text.media_extraction import MediaExtractionMixin
from src.handlers.text.conversation import TextConversationMixin
from src.handlers.text.response_utils import ResponseUtilitiesMixin


class TextHandler(MediaExtractionMixin, TextConversationMixin, ResponseUtilitiesMixin):
    """
    Unified text handler combining all text handling functionality.
    
    Uses composition via mixins for:
    - Media extraction (from MediaExtractionMixin)
    - Text conversation (from TextConversationMixin)
    - Response utilities (from ResponseUtilitiesMixin)
    """

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
        self.max_context_length = 9

        # Memory and conversation management
        self.memory_manager = MemoryManager(
            db=user_data_manager.db if hasattr(user_data_manager, "db") else None,
        )
        self.memory_manager.short_term_limit = 15
        self.memory_manager.token_limit = 64000
        self.model_history_manager = ModelHistoryManager(self.memory_manager)
        self.conversation_manager = ConversationManager(self.memory_manager, self.model_history_manager)

        # Context and formatting
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.prompt_formatter = PromptFormatter()

        # Media handling
        self.media_analyzer = MediaAnalyzer(gemini_api, openrouter_api)
        self.document_sender = DocumentSender()
        self.user_model_manager = None

    async def _initialize_rag_system(self):
        """Initialize Personalized RAG system for enhanced memory."""
        try:
            rag_system = PersonalizedRAGSystem(
                memory_manager=self.memory_manager,
                persistence_manager=self.memory_manager.persistence_manager,
                db=self.user_data_manager.db if hasattr(self.user_data_manager, "db") else None,
            )
            self.rag_integration = RAGIntegration(rag_system)
            self.logger.info("Personalized RAG system initialized successfully")
            return True
        except Exception as e:
            self.logger.warning(f"Could not initialize RAG system: {e}. Continuing without RAG.")
            return False

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Main handler for text messages."""
        if not update.message and not update.edited_message:
            return

        user_id = update.effective_user.id
        message = update.message or update.edited_message
        message_text = message.text
        chat = update.effective_chat
        is_group = chat and chat.type in ["group", "supergroup"]

        # Handle group chat enhancements
        if is_group and hasattr(self, "_group_chat_integration") and self._group_chat_integration:
            enhanced_message = await self._group_chat_integration.process_message(update, context)
            if enhanced_message and enhanced_message.get("enhanced_text"):
                message_text = enhanced_message["enhanced_text"]

        if "enhanced_message" in context.user_data:
            if update.effective_chat.type in ["group", "supergroup"]:
                message_text = context.user_data["enhanced_message"]

        quoted_text, quoted_message_id = self.context_handler.extract_reply_context(message)

        # Initialize RAG on first message
        if not hasattr(self, "rag_integration"):
            await self._initialize_rag_system()

        try:
            # Handle edited messages
            if update.edited_message and "bot_messages" in context.user_data:
                await self._handle_edited_message(update, context)

            # Check for bot mention in group chats
            if update.effective_chat.type in ["group", "supergroup"]:
                entities = []
                if message.entities:
                    entities = [
                        {"type": e.type, "offset": e.offset, "length": e.length}
                        for e in message.entities
                    ]
                if not BotUsernameHelper.is_bot_mentioned(message_text, context, entities=entities):
                    return
                message_text = BotUsernameHelper.remove_bot_mention(message_text, context)

            # Extract media files
            has_attached_media, media_files, media_type = await self._extract_media_files(update, context)

            # Send thinking message
            thinking_message = await message.reply_text("Processing your request...")
            await self._send_appropriate_chat_action(update, context, has_attached_media, media_type)

            # Get user preferences and history
            preferred_model = await self._get_user_preferred_model(user_id)
            await self.memory_manager.extract_and_save_user_info(user_id, message_text)
            history_context = await self.conversation_manager.get_conversation_history(
                user_id, max_messages=self.max_context_length, model=preferred_model
            )

            # Add user context
            user_context = await self._load_user_context(user_id, update)
            if user_context and history_context:
                history_context.insert(0, {"role": "system", "content": f"User information: {user_context}"})

            # Handle media or text
            if has_attached_media:
                await self._handle_media_analysis(
                    update, context, thinking_message, media_files, media_type,
                    message_text, user_id, preferred_model
                )
                return

            await self._handle_text_conversation(
                update, context, thinking_message, message_text, quoted_text,
                quoted_message_id, history_context, user_id, preferred_model
            )

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            if "thinking_message" in locals() and thinking_message:
                await thinking_message.delete()
            await self.response_formatter.safe_send_message(
                update.message, "Sorry, I encountered an error. Please try again later."
            )

    async def _handle_edited_message(self, update, context):
        """Handle when a user edits their previous message."""
        original_message_id = update.edited_message.message_id
        if original_message_id in context.user_data["bot_messages"]:
            for msg_id in context.user_data["bot_messages"][original_message_id]:
                if msg_id:
                    try:
                        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg_id)
                    except Exception:
                        pass
            del context.user_data["bot_messages"][original_message_id]

    async def _process_complete_media_group(self, media_group_id, chat_id, user_id, caption, context):
        """Process a complete media group after a delay."""
        await asyncio.sleep(1.5)
        if "media_groups" in context.bot_data and media_group_id in context.bot_data["media_groups"]:
            media_files = context.bot_data["media_groups"].pop(media_group_id)
            if media_files:
                thinking_message = await context.bot.send_message(chat_id=chat_id, text="Processing multiple files...")
                try:
                    from src.services.user_preferences_manager import UserPreferencesManager
                    from src.services.media.multi_file_processor import MultiFileProcessor

                    preferences_manager = UserPreferencesManager(self.user_data_manager)
                    preferred_model = await preferences_manager.get_user_model_preference(user_id)
                    multi_processor = MultiFileProcessor(self.gemini_api)
                    result = await multi_processor.process_multiple_files(media_files, caption or "Analyze these files")

                    if thinking_message:
                        await thinking_message.delete()

                    if "results" in result:
                        formatted_results = [
                            f"*{filename}*:\n{content}"
                            for filename, content in result["results"].items()
                            if isinstance(content, str)
                        ]
                        if formatted_results:
                            response = "\n\n".join(formatted_results)
                            chunks = await self.response_formatter.split_long_message(response)
                            for chunk in chunks:
                                mock_message = self.MockMessage(context.bot, chat_id)
                                await self.response_formatter.safe_send_message(mock_message, chunk)
                except Exception as e:
                    self.logger.error(f"Error processing media group: {e}")
                    if thinking_message:
                        await thinking_message.delete()
                    await context.bot.send_message(chat_id=chat_id, text="Sorry, there was an error processing your files.")

    async def _handle_media_analysis(
        self, update, context, thinking_message, media_files, media_type,
        message_text, user_id, preferred_model
    ):
        """Handle analysis of media files."""
        if len(media_files) > 1:
            from src.services.media.multi_file_processor import MultiFileProcessor
            multi_processor = MultiFileProcessor(self.gemini_api)
            result = await multi_processor.process_multiple_files(media_files, message_text or "Analyze these files")

            if thinking_message:
                try:
                    await thinking_message.delete()
                except Exception:
                    pass

            if "results" in result:
                formatted_results = [
                    f"*{filename}*:\n{content}"
                    for filename, content in result["results"].items()
                    if isinstance(content, str)
                ]
                if formatted_results:
                    response = "\n\n".join(formatted_results)
                    chunks = await self.response_formatter.split_long_message(response)
                    for chunk in chunks:
                        await self.response_formatter.safe_send_message(update.message, chunk)
                    await self.conversation_manager.save_message_pair(
                        user_id, message_text or "[Multiple files uploaded]", response, preferred_model
                    )
                    return
            await update.message.reply_text("Sorry, I couldn't analyze the content you provided.")
            return

        result = await self.media_analyzer.analyze_media(media_files, message_text, preferred_model)

        if thinking_message:
            try:
                await thinking_message.delete()
            except Exception:
                pass

        if result:
            model_indicator = "Gemini" if preferred_model == "gemini" else preferred_model.capitalize()
            text_to_send = self.response_formatter.format_with_model_indicator(result, model_indicator)
            await self.response_formatter.safe_send_message(update.message, text_to_send)
            await self.conversation_manager.save_message_pair(
                user_id, message_text or f"[{media_type.capitalize()} uploaded]", result, preferred_model
            )
            if self.user_data_manager:
                stat_param = "image" if media_type == "photo" else media_type
                await self.user_data_manager.update_stats(user_id, **{stat_param: True})
        else:
            await self.response_formatter.safe_send_message(
                update.message, "Sorry, I couldn't analyze the content you provided."
            )
