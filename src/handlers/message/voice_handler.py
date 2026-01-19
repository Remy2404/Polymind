"""
Voice message handling.
"""
import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from src.handlers.text_handlers import TextHandler
from src.services.media.voice_processor import VoiceProcessor, SpeechEngine, create_voice_processor
from src.services.memory_context.conversation_manager import ConversationManager


class VoiceMessageHandlerMixin:
    """Mixin for voice message handling."""

    async def _handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming voice messages with enhanced multi-engine support."""
        if not update.message or not update.message.voice:
            self.logger.error("Received update with no voice message")
            return

        user_id = update.effective_user.id
        self.telegram_logger.log_message("Received voice message", user_id)

        quoted_text, quoted_message_id = self.context_handler.extract_reply_context(update.message)

        try:
            # Initialize voice processor
            if not hasattr(self, "voice_processor") or self.voice_processor is None:
                try:
                    self.voice_processor = await create_voice_processor(engine=SpeechEngine.FASTER_WHISPER)
                except Exception as e:
                    self.logger.error(f"Failed to initialize enhanced voice processor: {e}")
                    self.voice_processor = VoiceProcessor()

            # Initialize preferences manager
            if not hasattr(self, "preferences_manager") or self.preferences_manager is None:
                from src.services.user_preferences_manager import UserPreferencesManager
                self.preferences_manager = UserPreferencesManager(self.user_data_manager)

            lang = "en-US"
            status_message = await update.message.reply_text(
                "Processing your voice message with enhanced AI recognition..."
            )

            # Download and convert voice file
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            ogg_file_path, wav_file_path = await self.voice_processor.download_and_convert(voice_file, str(user_id))

            # Transcribe
            if hasattr(self.voice_processor, "get_best_transcription"):
                text, recognition_language, metadata = await self.voice_processor.get_best_transcription(
                    wav_file_path, language=lang, confidence_threshold=0.6
                )
                engine_used = metadata.get("engine", "unknown")
                confidence = metadata.get("confidence", 0.0)
            else:
                text, recognition_language = await self.voice_processor.transcribe(wav_file_path, lang)
                metadata = {"engine": "basic", "confidence": 0.7}
                engine_used = "basic"
                confidence = 0.7

            if not text:
                error_text = (
                    "Sorry, I couldn't understand the audio.\n\n"
                    "Tips:\n"
                    "Speak clearly and avoid background noise\n"
                    "Try speaking in English for better accuracy\n"
                    "Send shorter voice messages (under 30 seconds)"
                )
                try:
                    await status_message.edit_text(error_text)
                except Exception:
                    await update.message.reply_text(error_text)
                return

            try:
                await status_message.delete()
            except Exception:
                pass

            # Send transcription result
            confidence_emoji = "" if confidence > 0.8 else "" if confidence > 0.6 else ""
            transcript_text = f"**Voice Message Transcribed** {confidence_emoji}\n\n{text}"

            try:
                await self.response_formatter.safe_send_message(update.message, transcript_text)
            except Exception:
                await update.message.reply_text(f"Transcription:\n{text}")

            self.telegram_logger.log_message(f"Transcribed {recognition_language} text: {text}", user_id)
            await self.user_data_manager.initialize_user(user_id)

            # Get conversation manager
            if hasattr(self.text_handler, "conversation_manager"):
                conversation_manager = self.text_handler.conversation_manager
            else:
                if not hasattr(self, "_conversation_manager") or not self._conversation_manager:
                    text_handler = TextHandler(
                        self.gemini_api,
                        self.user_data_manager,
                        self.openrouter_api if hasattr(self, "openrouter_api") else None,
                        self.deepseek_api if hasattr(self, "deepseek_api") else None,
                    )
                    self._conversation_manager = ConversationManager(
                        text_handler.memory_manager, text_handler.model_history_manager
                    )
                conversation_manager = self._conversation_manager

            # Save to conversation history
            await conversation_manager.save_media_interaction(
                user_id, "voice", text, f"I've transcribed your voice message which said: {text}"
            )

            # Generate AI response
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

            prompt = text
            if quoted_text:
                prompt = self.context_handler.format_prompt_with_quote(text, quoted_text)

            user_settings = await self.user_data_manager.get_user_settings(str(user_id))
            preferred_model = await self.user_data_manager.get_user_preference(user_id, "preferred_model", None)
            active_model = preferred_model or user_settings.get("active_model", "gemini")

            model_indicator, model_config = self.get_model_indicator_and_config(active_model)

            try:
                conversation_history = await conversation_manager.get_conversation_history(
                    user_id, max_messages=10, model=active_model
                )
            except Exception:
                conversation_history = None

            ai_response = await self.generate_ai_response(prompt, active_model, user_id, conversation_history)

            if not ai_response:
                ai_response = "I'm sorry, I couldn't generate a response at this time."

            # Format and send response
            voice_formatted_response = f"**Voice Response:**\n\n{ai_response}"
            formatted_response = self.response_formatter.format_with_model_indicator(
                voice_formatted_response, model_indicator, quoted_text is not None
            )

            await self.response_formatter.safe_send_message(update.message, formatted_response)

            # Save to conversation
            await conversation_manager.save_message_pair(
                user_id, f"[Voice Message Transcribed]: {prompt}", ai_response, active_model
            )

        except Exception as e:
            self.logger.error(f"Error processing voice message: {str(e)}", exc_info=True)
            error_message = "Sorry, there was an error processing your voice message. Please try again later."
            try:
                if "status_message" in locals() and status_message:
                    await status_message.edit_text(error_message)
                else:
                    await update.message.reply_text(error_message)
            except Exception:
                pass
