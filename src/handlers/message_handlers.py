"""
Message Handlers - Facade class combining all message handling functionality.

This is the main entry point for message handling, providing backwards
compatibility while using modular components internally.
"""
import logging
from telegram import Update
from telegram.ext import MessageHandler, filters, ContextTypes

from src.services.multimodal_processor import TelegramMultimodalProcessor
from src.utils.docgen.document_processor import DocumentProcessor
from src.handlers.message_context_handler import MessageContextHandler
from src.handlers.response_formatter import ResponseFormatter
from src.services.model_handlers.model_configs import ModelConfigurations, Provider

from src.handlers.message.model_helpers import ModelHelpersMixin
from src.handlers.message.text_handler import TextMessageHandlerMixin
from src.handlers.message.image_handler import ImageMessageHandlerMixin
from src.handlers.message.voice_handler import VoiceMessageHandlerMixin
from src.handlers.message.document_handler import DocumentMessageHandlerMixin


logger = logging.getLogger(__name__)


class MessageHandlers(
    ModelHelpersMixin,
    TextMessageHandlerMixin,
    ImageMessageHandlerMixin,
    VoiceMessageHandlerMixin,
    DocumentMessageHandlerMixin,
):
    """
    Unified message handlers combining all message handling functionality.
    
    This class provides a single interface for:
    - Model configuration and AI response generation (from ModelHelpersMixin)
    - Text message handling (from TextMessageHandlerMixin)
    - Image message handling (from ImageMessageHandlerMixin)
    - Voice message handling (from VoiceMessageHandlerMixin)
    - Document message handling (from DocumentMessageHandlerMixin)
    """

    def __init__(
        self,
        gemini_api,
        user_data_manager,
        telegram_logger,
        text_handler,
        deepseek_api=None,
        openrouter_api=None,
        command_handlers=None,
    ):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.text_handler = text_handler
        self.logger = logging.getLogger(__name__)
        self.multimodal_processor = TelegramMultimodalProcessor(gemini_api)
        self.deepseek_api = deepseek_api
        self.openrouter_api = openrouter_api
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.document_processor = DocumentProcessor(gemini_api)
        self.ai_command_router = None
        self._conversation_manager = None
        self._group_chat_integration = None
        self.all_models = ModelConfigurations.get_all_models()
        
        self.logger.info(f"Initialized with {len(self.all_models)} available models")
        for provider in Provider:
            provider_models = ModelConfigurations.get_models_by_provider(provider)
            if provider_models:
                self.logger.info(f"{provider.value.title()} models: {len(provider_models)} available")

    async def _error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors occurring in the dispatcher."""
        self.logger.error(f"Update {update} caused error: {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred while processing your request. Please try again later."
            )

    def register_handlers(self, application):
        """Register message handlers with the application."""
        try:
            application.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message)
            )
            application.add_handler(MessageHandler(filters.PHOTO, self._handle_image_message))
            application.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message))
            application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
            application.add_error_handler(self._error_handler)
            self.logger.info("Message handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register message handlers: {str(e)}")
            raise Exception("Failed to register message handlers") from e
