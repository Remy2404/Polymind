import logging
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from src.services.media.voice_processor import create_voice_processor, SpeechEngine, VoiceProcessor
from src.utils.message_utils import extract_reply_context, format_prompt_with_quote
from src.handlers.response_formatter import ResponseFormatter

class VoiceLogic:
    """Handles voice message transcription and processing."""

    def __init__(self, user_data_manager, telegram_logger, model_logic):
        self.logger = logging.getLogger(__name__)
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.model_logic = model_logic
        self.voice_processor = None
        self.response_formatter = ResponseFormatter()

    async def initialize_voice_processor(self):
        """Initialize the voice processor with enhanced engine if possible."""
        try:
            self.voice_processor = await create_voice_processor(
                engine=SpeechEngine.FASTER_WHISPER
            )
            self.logger.info("Enhanced voice processor initialized with Faster-Whisper")
        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced voice processor: {e}")
            self.voice_processor = VoiceProcessor()

    async def handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, conversation_manager):
        """Handle incoming voice messages."""
        if not self.voice_processor:
             await self.initialize_voice_processor()

        user_id = update.effective_user.id
        self.telegram_logger.log_message("Received voice message", user_id)
        
        quoted_text, quoted_message_id = extract_reply_context(update.message)
        
        status_message = await update.message.reply_text(
            "🎤 Processing your voice message with enhanced AI recognition..."
        )

        try:
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            ogg_file_path, wav_file_path = await self.voice_processor.download_and_convert(
                voice_file, str(user_id)
            )

            # Transcription logic
            lang = "en-US"
            if hasattr(self.voice_processor, "get_best_transcription"):
                text, recognition_language, metadata = await self.voice_processor.get_best_transcription(
                    wav_file_path, language=lang, confidence_threshold=0.6
                )
                engine_used = metadata.get("engine", "unknown")
                confidence = metadata.get("confidence", 0.0)
            else:
                text, recognition_language = await self.voice_processor.transcribe(
                    wav_file_path, lang
                )
                engine_used = "basic"
                confidence = 0.7

            if not text:
                await self._handle_transcription_error(status_message, update)
                return

            # Clean up status message
            try:
                await status_message.delete()
            except Exception:
                pass

            # Send transcription result
            await self._send_transcription_result(update, text, confidence, engine_used)

            # Initialize user and get conversation context
            await self.user_data_manager.initialize_user(user_id)
            
            await conversation_manager.save_media_interaction(
                user_id, "voice", text, f"I've transcribed your voice message which said: {text}"
            )
            
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            # Prepare prompt for AI
            prompt = text
            if quoted_text:
                prompt = format_prompt_with_quote(text, quoted_text)

            # Get model preference
            user_settings = await self.user_data_manager.get_user_settings(str(user_id))
            preferred_model = await self.user_data_manager.get_user_preference(
                user_id, "preferred_model", None
            )
            active_model = preferred_model or user_settings.get("active_model", "gemini")
            
            model_indicator, _ = self.model_logic.get_model_indicator_and_config(active_model)

            # Get conversation history
            conversation_history = await conversation_manager.get_conversation_history(
                 user_id, max_messages=10, model=active_model
            )

            # Generate AI response
            ai_response = await self.model_logic.generate_ai_response(
                prompt, active_model, user_id, conversation_history
            )
            
            # Format and send response
            await self._send_voice_response(
                update, ai_response, model_indicator, prompt, quoted_text, conversation_history, conversation_manager, user_id, active_model
            )

        except Exception as e:
            self.logger.error(f"Error processing voice message: {e}", exc_info=True)
            await update.message.reply_text("Sorry, there was an error processing your voice message.")

    async def _handle_transcription_error(self, status_message, update):
        error_text = (
            "❌ Sorry, I couldn't understand the audio.\n\n💡 **Tips:**\n"
            "• Speak clearly and avoid background noise\n"
            "• Try speaking in English for better accuracy\n"
            "• Send shorter voice messages (under 30 seconds)"
        )
        try:
            await status_message.edit_text(error_text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(error_text, parse_mode="Markdown")

    async def _send_transcription_result(self, update, text, confidence, engine_used):
        confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
        transcript_text = f"🎤 **Voice Message Transcribed** {confidence_emoji}\n\n{text}"
        if confidence > 0.7:
             transcript_text += f"\n\n_Engine: {engine_used.title()}, Confidence: {confidence:.1%}_"
        
        try:
             await self.response_formatter.safe_send_message(update.message, transcript_text)
        except Exception:
             await update.message.reply_text(f"🎤 Transcription: \n{text}")

    async def _send_voice_response(
        self, update, ai_response, model_indicator, prompt, quoted_text, history, conversation_manager, user_id, active_model
    ):
        voice_intro = "🎤 **Voice Response:**"
        context_hint = ""
        if history and len(history) > 0:
             context_hint = "_Continuing our conversation..._\n\n"
        
        voice_formatted_response = f"{voice_intro}\n\n{context_hint}{ai_response}"
        formatted_response = self.response_formatter.format_with_model_indicator(
            voice_formatted_response, model_indicator, quoted_text is not None
        )
        
        await self.response_formatter.safe_send_message(update.message, formatted_response)
        
        voice_enhanced_prompt = f"[Voice Message Transcribed]: {prompt}"
        await conversation_manager.save_message_pair(
            user_id, voice_enhanced_prompt, ai_response, active_model
        )
