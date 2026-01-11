import logging
from telegram import Update
from telegram.ext import ContextTypes
from src.handlers.text_handlers import TextHandler

# New Modules
from .model_logic import ModelLogic
from .voice_logic import VoiceLogic
from .media_helpers import MediaHelpers

class MessageHandlers:
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
        self.deepseek_api = deepseek_api
        self.openrouter_api = openrouter_api
        
        self._conversation_manager = None

        # Logic Modules
        self.model_logic = ModelLogic(gemini_api, openrouter_api, deepseek_api)
        self.voice_logic = VoiceLogic(user_data_manager, telegram_logger, self.model_logic)

    # --- Model Accessors (Delegated to ModelLogic) ---
    def get_all_models(self): return self.model_logic.get_all_models()
    def get_models_by_provider(self, p): return self.model_logic.get_models_by_provider(p)
    def get_free_models(self): return self.model_logic.get_free_models()
    def get_model_stats(self): # Simplified proxy
        return {"total_models": len(self.model_logic.all_models)}

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        try:
            if not update.message:
                return

            # Use injected text_handler
            await self.text_handler.handle_text_message(update, context)
            
            # Update stats
            user_id = update.effective_user.id
            await self.user_data_manager.update_stats(user_id, {"text_messages": 1})

        except Exception as e:
            self.logger.error(f"Error in text handler router: {e}")

    async def _handle_image_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming image messages using MediaHelpers and TextHandler logic."""
        user_id = update.effective_user.id
        self.logger.info(f"Processing image from user {user_id}")
        
        if not update.message or not update.message.photo:
            return

        supports_images, _, error_msg = await self.model_logic.check_model_supports_media(
            self.user_data_manager, user_id, "image"
        )
        if not supports_images:
            await update.message.reply_text(error_msg, parse_mode="Markdown")
            return

        # Use MediaHelpers to extract
        has_media, media_files, media_type = await MediaHelpers.extract_media_files(update, context)
        
        if has_media:
            processing_msg = await update.message.reply_text("🖼️ Processing your image...")
            await MediaHelpers.send_appropriate_chat_action(update, context, True, "photo")
            
            preferred_model = await self.user_data_manager.get_user_preference(user_id, "preferred_model")

            await self.text_handler.conversation_logic.handle_media_analysis(
                update, context, processing_msg, media_files, media_type, 
                update.message.caption or "Analyze this image",
                user_id, preferred_model
            )

    async def _handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice messages via VoiceLogic."""
        # Ensure conversation manager access
        cm = getattr(self.text_handler, "conversation_manager", None)
             
        await self.voice_logic.handle_voice_message(update, context, cm)

    async def _handle_document_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document messages."""
        user_id = update.effective_user.id
        self.logger.info(f"Processing document for user: {user_id}")
        
        supports_docs, _, error_msg = await self.model_logic.check_model_supports_media(
             self.user_data_manager, user_id, "document"
        )
        
        if not supports_docs:
             await update.message.reply_text(error_msg, parse_mode="Markdown")
             return

        # Similar delegation flow as image
        has_media, media_files, media_type = await MediaHelpers.extract_media_files(update, context)
        if has_media:
             processing_msg = await update.message.reply_text("Processing document...")
             preferred_model = await self.user_data_manager.get_user_preference(user_id, "preferred_model")
             
             await self.text_handler.conversation_logic.handle_media_analysis(
                 update, context, processing_msg, media_files, media_type,
                 update.message.caption or "Analyze this document",
                 user_id, preferred_model
             )
