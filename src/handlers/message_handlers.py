import os
import io
import logging
import traceback
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction
from telegram.ext import MessageHandler, filters, ContextTypes
from src.services.multimodal_processor import TelegramMultimodalProcessor
from src.handlers.text_handlers import TextHandler
from src.services.ai_command_router import AICommandRouter
from src.utils.docgen.document_processor import DocumentProcessor
from src.handlers.message_context_handler import MessageContextHandler
from src.handlers.response_formatter import ResponseFormatter
from src.services.media.voice_processor import (
    VoiceProcessor,
    SpeechEngine,
    create_voice_processor,
)
from src.services.memory_context.conversation_manager import ConversationManager
from src.services.group_chat.integration import GroupChatIntegration
from src.services.model_handlers.model_configs import (
    ModelConfigurations,
    Provider,
    ModelConfig,
)
from src.utils.bot_username_helper import BotUsernameHelper
logger = logging.getLogger(__name__)
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
        self.multimodal_processor = TelegramMultimodalProcessor(gemini_api)
        self.deepseek_api = deepseek_api
        self.openrouter_api = openrouter_api
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.document_processor = DocumentProcessor(gemini_api)
        self.ai_command_router = None
        if command_handlers:
            self.ai_command_router = AICommandRouter(command_handlers, gemini_api)
        self._conversation_manager = None
        self._group_chat_integration = None
        self.all_models = ModelConfigurations.get_all_models()
        self.logger.info(f"Initialized with {len(self.all_models)} available models")
        for provider in Provider:
            provider_models = ModelConfigurations.get_models_by_provider(provider)
            if provider_models:
                self.logger.info(
                    f"{provider.value.title()} models: {len(provider_models)} available"
                )
                model_names = list(provider_models.keys())[:5]
                self.logger.debug(f"  Examples: {model_names}")
        free_models = ModelConfigurations.get_free_models()
        self.logger.info(f"Free OpenRouter models available: {len(free_models)}")
        common_models = ["llama-3.3-8b", "gemini", "deepseek"]
        for model_id in common_models:
            self.log_model_verification(model_id)
    def get_all_models(self) -> dict:
        """Get all available models - useful for external access."""
        return self.all_models
    def get_models_by_provider(self, provider: Provider) -> dict:
        """Get models by specific provider."""
        return ModelConfigurations.get_models_by_provider(provider)
    def get_free_models(self) -> dict:
        """Get all free models."""
        return ModelConfigurations.get_free_models()
    def log_model_verification(self, model_id: str) -> bool:
        """Log verification for a specific model and return if it exists."""
        model_config = self.get_model_config(model_id)
        if model_config:
            self.logger.info(f"✅ Model '{model_id}' verified:")
            self.logger.info(f"  - Display name: {model_config.display_name}")
            self.logger.info(f"  - Provider: {model_config.provider.value}")
            self.logger.info(
                f"  - OpenRouter key: {model_config.openrouter_model_key or 'N/A'}"
            )
            self.logger.info(f"  - Emoji: {model_config.indicator_emoji}")
            return True
        else:
            self.logger.warning(f"❌ Model '{model_id}' not found in configurations")
            return False
    def get_model_stats(self) -> dict:
        """Get statistics about available models."""
        total_models = len(self.all_models)
        free_models = len(self.get_free_models())
        provider_counts = {}
        for provider in Provider:
            provider_models = self.get_models_by_provider(provider)
            provider_counts[provider.value] = len(provider_models)
        return {
            "total_models": total_models,
            "free_models": free_models,
            "provider_counts": provider_counts,
            "model_ids": list(self.all_models.keys())[:10],
        }
    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get model configuration by ID."""
        return self.all_models.get(model_id)
    def get_model_indicator_and_config(self, model_id: str) -> tuple[str, ModelConfig]:
        """Get model indicator emoji and configuration for a model."""
        model_config = self.get_model_config(model_id)
        if model_config:
            return (
                f"{model_config.indicator_emoji} {model_config.display_name}",
                model_config,
            )
        else:
            self.logger.warning(f"Unknown model ID: {model_id}, using default")
            return "🤖 Unknown Model", None

    async def check_model_supports_media(self, user_id: int, media_type: str) -> tuple[bool, str, str]:
        """
        Check if the user's preferred model supports a specific media type.

        Args:
            user_id: The user's ID
            media_type: Type of media ('image', 'document', 'audio', 'video')

        Returns:
            tuple: (supports_media, model_id, error_message)
        """
        try:
            # Get user's preferred model
            preferred_model = await self.user_data_manager.get_user_preference(
                user_id, "preferred_model", default="gemini"
            )

            model_config = self.get_model_config(preferred_model)
            if not model_config:
                return False, preferred_model, f"Model '{preferred_model}' not found."

            # Check capabilities based on media type
            if media_type == "image":
                supports_media = getattr(model_config, 'supports_images', False)
                capability_name = "images"
            elif media_type == "document":
                supports_media = getattr(model_config, 'supports_documents', False)
                capability_name = "documents"
            elif media_type in ["audio", "voice"]:
                supports_media = getattr(model_config, 'supports_audio', False)
                capability_name = "audio"
            elif media_type == "video":
                supports_media = getattr(model_config, 'supports_video', False)
                capability_name = "video"
            else:
                return False, preferred_model, f"Unsupported media type: {media_type}"

            if not supports_media:
                error_message = (
                    f"❌ **{model_config.display_name}** doesn't support {capability_name}.\n\n"
                    f"💡 **To process {capability_name}, please switch to a vision-capable model:**\n"
                    f"• Use `/switchmodel` command\n"
                    f"• Look for models with 👁️ or ✨ emoji\n"
                    f"• Recommended: **Gemini 2.5 Flash** (supports images & documents)"
                )
                return False, preferred_model, error_message

            return True, preferred_model, ""

        except Exception as e:
            self.logger.error(f"Error checking model media support: {str(e)}")
            return False, "unknown", f"Error checking model capabilities: {str(e)}"
    async def generate_ai_response(
        self,
        prompt: str,
        model_id: str,
        user_id: int,
        conversation_context: list = None,
    ) -> str:
        """Generate AI response using the specified model with conversation context."""
        model_config = self.get_model_config(model_id)
        if not model_config:
            self.logger.error(f"Model configuration not found for: {model_id}")
            return f"Sorry, the model '{model_id}' is not available."
        if conversation_context:
            self.logger.info(f"🧠 AI Context Debug - Model: {model_id}")
            self.logger.info(f"   └─ Context messages: {len(conversation_context)}")
            for i, msg in enumerate(conversation_context[-5:]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                preview = content[:100] + ("..." if len(content) > 100 else "")
                self.logger.info(f"   └─ Message {i + 1} [{role.upper()}]: {preview}")
                if "name" in content.lower():
                    self.logger.info("   └─ ⭐ Contains name/identity information!")
            self.logger.info(f"   └─ Current prompt: {prompt[:100]}...")
        else:
            self.logger.info(
                f"⚠ No conversation context provided for model: {model_id}"
            )
        try:
            self.logger.info(f"Generating response with model: {model_id}")
            self.logger.info(
                f"Attempting to use {model_config.provider.value.title()} API with {model_id} model"
            )
            if model_config.provider == Provider.GEMINI:
                self.logger.info("Successfully used Gemini API")
                return await self.gemini_api.generate_response(prompt)
            elif (
                model_config.provider == Provider.DEEPSEEK
                and hasattr(self, "deepseek_api")
                and self.deepseek_api
            ):
                self.logger.info("Successfully used DeepSeek API")
                return await self.deepseek_api.generate_response(prompt)
            elif (
                model_config.provider == Provider.OPENROUTER
                and hasattr(self, "openrouter_api")
                and self.openrouter_api
            ):
                if model_config.openrouter_model_key:
                    self.logger.info("Successfully used OpenRouter API with model key")
                    response = (
                        await self.openrouter_api.generate_response_with_model_key(
                            prompt=prompt,
                            openrouter_model_key=model_config.openrouter_model_key,
                            context=conversation_context,
                        )
                    )
                else:
                    self.logger.info("Successfully used OpenRouter API with model ID")
                    response = await self.openrouter_api.generate_response(
                        prompt=prompt, context=conversation_context, model=model_id
                    )
                if response and conversation_context:
                    context_keywords = ["name", "your name", "my name"]
                    if any(keyword in response.lower() for keyword in context_keywords):
                        self.logger.info(
                            "✅ AI response appears to use conversation context"
                        )
                    else:
                        self.logger.warning(
                            "⚠ AI response may not be using conversation context effectively"
                        )
                return response
            else:
                self.logger.warning(
                    f"No API available for provider: {model_config.provider.value}"
                )
                return f"Sorry, the {model_config.display_name} model is currently unavailable."
        except Exception as e:
            self.logger.error(f"Error generating response with {model_id}: {str(e)}")
            return f"Sorry, there was an error with the {model_config.display_name} model: {str(e)}"
    async def _handle_text_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages."""
        try:
            if update.message is None and update.callback_query is None:
                self.logger.error("Received update with no message or callback query")
                return
            if update.callback_query:
                user_id = update.callback_query.from_user.id
                message_text = update.callback_query.data
                await update.callback_query.answer()
            else:
                user_id = update.effective_user.id
                message_text = update.message.text
            if update.message and await self.handle_awaiting_doc_text(update, context):
                return
            if update.message and context.user_data.get("awaiting_aidoc_topic"):
                context.user_data["awaiting_aidoc_topic"] = False
                context.user_data["aidoc_prompt"] = update.message.text
                if hasattr(self, "command_handlers") and self.command_handlers:
                    await self.command_handlers.document_commands._show_ai_document_format_selection(
                        update, context
                    )
                else:
                    from src.handlers.commands.document_commands import DocumentCommands
                    doc_commands = DocumentCommands(
                        self.gemini_api, self.user_data_manager, self.telegram_logger
                    )
                    await doc_commands._show_ai_document_format_selection(
                        update, context
                    )
                return
            self.logger.info(
                f"Received text message from user {user_id}: {message_text}"
            )
            await self.user_data_manager.initialize_user(user_id)
            if self._group_chat_integration is None and self._conversation_manager:
                self._group_chat_integration = GroupChatIntegration(
                    self.user_data_manager, self._conversation_manager
                )
            enhanced_message_text = message_text
            group_metadata = {}
            chat = update.effective_chat
            if (
                self._group_chat_integration
                and chat
                and chat.type in ["group", "supergroup"]
            ):
                try:
                    (
                        enhanced_message_text,
                        group_metadata,
                    ) = await self._group_chat_integration.process_group_message(
                        update, context, message_text
                    )
                    self.logger.info(f"Enhanced group message for chat {chat.id}")
                except Exception as e:
                    self.logger.error(f"Error processing group message: {e}")
            message_entities = []
            if update.message and update.message.entities:
                message_entities = [
                    {
                        "type": entity.type,
                        "offset": entity.offset,
                        "length": entity.length,
                        "url": getattr(entity, "url", None),
                        "user": getattr(entity, "user", None),
                    }
                    for entity in update.message.entities
                ]
            if BotUsernameHelper.is_bot_mentioned(
                enhanced_message_text, context, entities=message_entities
            ):
                bot_username = BotUsernameHelper.get_bot_username(context, with_at=True)
                self.logger.info(
                    f"Bot mentioned by user {user_id} using username: {bot_username}"
                )
            if self.ai_command_router:
                try:
                    is_group_chat = chat and chat.type in ["group", "supergroup"]
                    is_mentioned = BotUsernameHelper.is_bot_mentioned(
                        enhanced_message_text, context, entities=message_entities
                    )
                    if not is_group_chat or is_mentioned:
                        has_attached_media = bool(
                            update.message
                            and (
                                update.message.photo
                                or update.message.video
                                or update.message.document
                                or update.message.audio
                                or update.message.voice
                            )
                        )
                        should_route = (
                            await self.ai_command_router.should_route_message(
                                message_text, has_attached_media
                            )
                        )
                        if should_route:
                            (
                                intent,
                                confidence,
                            ) = await self.ai_command_router.detect_intent(
                                message_text, has_attached_media
                            )
                            self.logger.info(
                                f"Routing message to command: {intent.value} (confidence: {confidence:.2f})"
                            )
                            command_executed = (
                                await self.ai_command_router.route_command(
                                    update, context, intent, message_text
                                )
                            )
                            if command_executed:
                                await self.user_data_manager.update_stats(
                                    user_id,
                                    {
                                        "text_messages": 1,
                                        "total_messages": 1,
                                        "ai_commands": 1,
                                    },
                                )
                                return
                            else:
                                self.logger.info(
                                    "Command routing failed, falling back to normal text processing"
                                )
                except Exception as e:
                    self.logger.error(f"Error in AI command routing: {str(e)}")
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )
            if (
                enhanced_message_text != message_text
                and chat
                and chat.type in ["group", "supergroup"]
            ):
                context.user_data["group_context"] = group_metadata
                context.user_data["original_message"] = message_text
                context.user_data["enhanced_message"] = enhanced_message_text
            await text_handler.handle_text_message(update, context)
            await self.user_data_manager.update_stats(
                user_id, {"text_messages": 1, "total_messages": 1}
            )
        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await self._error_handler(update, context)
    async def _handle_image_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming image messages using unified conversation system."""
        try:
            user_id = update.effective_user.id
            self.logger.info(f"Processing image from user {user_id}")
            self.telegram_logger.log_message("Received image message", user_id)

            if (
                not update.message
                or not update.message.photo
                or len(update.message.photo) == 0
            ):
                await update.message.reply_text("Sorry, I couldn't process this image.")
                return

            # Check if user's model supports images
            supports_images, model_id, error_message = await self.check_model_supports_media(user_id, "image")
            if not supports_images:
                await update.message.reply_text(error_message, parse_mode="Markdown")
                return

            # Route through TextHandler for unified conversation processing
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )

            # Extract media files using TextHandler's method
            has_attached_media, media_files, media_type = await text_handler._extract_media_files(update, context)

            if has_attached_media and media_files:
                # Send processing message
                processing_message = await update.message.reply_text(
                    "🖼️ Processing your image. Please wait..."
                )
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id, action=ChatAction.TYPING
                )

                try:
                    # Use TextHandler's media analysis with conversation context
                    await text_handler._handle_media_analysis(
                        update,
                        context,
                        processing_message,
                        media_files,
                        media_type,
                        update.message.caption or "Analyze this image",  # Use caption as prompt
                        user_id,
                        model_id,
                    )

                    self.telegram_logger.log_message(
                        "Image processed successfully via unified system", user_id
                    )

                except Exception as e:
                    self.logger.error(f"Image processing error: {str(e)}")
                    try:
                        await processing_message.edit_text(
                            "❌ Sorry, there was an error processing your image. Please try again."
                        )
                    except Exception:
                        await update.message.reply_text(
                            "❌ Sorry, there was an error processing your image. Please try again."
                        )
            else:
                await update.message.reply_text("Sorry, I couldn't extract the image data.")

        except Exception as e:
            self.logger.error(f"Error in image message handler: {str(e)}")
            await self._error_handler(update, context)
    async def _handle_voice_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming voice messages with enhanced multi-engine support."""
        if not update.message or not update.message.voice:
            self.logger.error("Received update with no voice message")
            return
        user_id = update.effective_user.id
        conversation_id = f"user_{user_id}"
        self.telegram_logger.log_message("Received voice message", user_id)
        quoted_text, quoted_message_id = self.context_handler.extract_reply_context(
            update.message
        )
        try:
            if not hasattr(self, "voice_processor") or self.voice_processor is None:
                try:
                    self.voice_processor = await create_voice_processor(
                        engine=SpeechEngine.FASTER_WHISPER
                    )
                    self.logger.info(
                        "Enhanced voice processor initialized with Faster-Whisper"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to initialize enhanced voice processor: {e}"
                    )
                    self.voice_processor = VoiceProcessor()
            if (
                not hasattr(self, "preferences_manager")
                or self.preferences_manager is None
            ):
                from src.services.user_preferences_manager import UserPreferencesManager
                self.preferences_manager = UserPreferencesManager(
                    self.user_data_manager
                )
            lang = "en-US"
            self.logger.info(
                f"Voice recognition language set to: {lang} (English only)"
            )
            processing_text = (
                "🎤 Processing your voice message with enhanced AI recognition..."
            )
            status_message = await update.message.reply_text(processing_text)
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            (
                ogg_file_path,
                wav_file_path,
            ) = await self.voice_processor.download_and_convert(
                voice_file, str(user_id)
            )
            if hasattr(self.voice_processor, "get_best_transcription"):
                self.logger.info(
                    f"🎤 Enhanced voice transcription starting for language: {lang}"
                )
                (
                    text,
                    recognition_language,
                    metadata,
                ) = await self.voice_processor.get_best_transcription(
                    wav_file_path, language=lang, confidence_threshold=0.6
                )
                engine_used = metadata.get("engine", "unknown")
                confidence = metadata.get("confidence", 0.0)
                self.logger.info("🔍 VOICE TRANSCRIPTION RESULT:")
                self.logger.info(f"  → Engine: {engine_used}")
                self.logger.info(f"  → Confidence: {confidence:.3f}")
                self.logger.info(f"  → Text length: {len(text)} chars")
                self.logger.info(f"  → Text preview: {text[:100]}...")
                self.logger.info(
                    f"Enhanced transcription: engine={engine_used}, confidence={confidence:.2f}"
                )
            else:
                text, recognition_language = await self.voice_processor.transcribe(
                    wav_file_path, lang
                )
                metadata = {"engine": "basic", "confidence": 0.7}
                engine_used = "basic"
                confidence = 0.7
            if not text:
                error_text = (
                    "❌ Sorry, I couldn't understand the audio.\n\n💡 **Tips:**\n"
                    "• Speak clearly and avoid background noise\n"
                    "• Try speaking in English for better accuracy\n"
                    "• Send shorter voice messages (under 30 seconds)\n"
                    "• Make sure you're in a quiet environment"
                )
                try:
                    await status_message.edit_text(error_text, parse_mode="Markdown")
                except Exception:
                    await update.message.reply_text(error_text, parse_mode="Markdown")
                return
            try:
                await status_message.delete()
            except Exception as msg_error:
                self.logger.warning(
                    f"Could not delete status message: {str(msg_error)}"
                )
            confidence_emoji = (
                "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            )
            transcript_text = (
                f"🎤 **Voice Message Transcribed** {confidence_emoji}\n\n{text}"
            )
            if confidence > 0.7:
                transcript_text += (
                    f"\n\n_Engine: {engine_used.title()}, Confidence: {confidence:.1%}_"
                )
            try:
                await self.response_formatter.safe_send_message(
                    update.message, transcript_text
                )
            except Exception as reply_error:
                self.logger.error(
                    f"Error sending transcript message: {str(reply_error)}"
                )
                await update.message.reply_text(f"🎤 Transcription: \n{text}")
            self.telegram_logger.log_message(
                f"Transcribed {recognition_language} text: {text}", user_id
            )
            await self.user_data_manager.initialize_user(user_id)
            if hasattr(self.text_handler, "conversation_manager"):
                conversation_manager = self.text_handler.conversation_manager
            else:
                if (
                    not hasattr(self, "_conversation_manager")
                    or not self._conversation_manager
                ):
                    text_handler = TextHandler(
                        self.gemini_api,
                        self.user_data_manager,
                        (
                            self.openrouter_api
                            if hasattr(self, "openrouter_api")
                            else None
                        ),
                        self.deepseek_api if hasattr(self, "deepseek_api") else None,
                    )
                    self._conversation_manager = ConversationManager(
                        text_handler.memory_manager, text_handler.model_history_manager
                    )
                conversation_manager = self._conversation_manager
            await conversation_manager.save_media_interaction(
                user_id,
                "voice",
                text,
                f"I've transcribed your voice message which said: {text}",
            )
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            prompt = text
            if quoted_text:
                prompt = self.context_handler.format_prompt_with_quote(
                    text, quoted_text
                )
            user_settings = await self.user_data_manager.get_user_settings(str(user_id))
            preferred_model = await self.user_data_manager.get_user_preference(
                user_id, "preferred_model", None
            )
            active_model = preferred_model or user_settings.get(
                "active_model", "gemini"
            )
            self.logger.info(f"Voice message model selection - User: {user_id}")
            self.logger.info(f" - preferred_model from preferences: {preferred_model}")
            self.logger.info(
                f" - active_model from user settings: {user_settings.get('active_model', 'not set')}"
            )
            self.logger.info(
                f"SELECTED MODEL: '{active_model}' for voice response. Prompt: {prompt[:50]}..."
            )
            model_indicator, model_config = self.get_model_indicator_and_config(
                active_model
            )
            self.logger.info(
                f"Using model indicator: {model_indicator} for model: {active_model}"
            )
            try:
                conversation_history = (
                    await conversation_manager.get_conversation_history(
                        user_id, max_messages=10, model=active_model
                    )
                )
                self.logger.info(
                    f"Retrieved {len(conversation_history)} conversation history messages for voice response"
                )
                if conversation_history:
                    self.logger.info(
                        f"Voice context debug - First message: {conversation_history[0].get('content', 'N/A')[:100]}..."
                    )
                    self.logger.info(
                        f"Voice context debug - Last message: {conversation_history[-1].get('content', 'N/A')[:100]}..."
                    )
                else:
                    self.logger.warning(
                        "No conversation history found for voice message context"
                    )
            except Exception as e:
                self.logger.warning(
                    f"Failed to retrieve conversation history: {str(e)}"
                )
                conversation_history = None
            ai_response = await self.generate_ai_response(
                prompt, active_model, user_id, conversation_history
            )
            if not ai_response:
                self.logger.warning(
                    f"Empty AI response for user {user_id} with active model {active_model}"
                )
                ai_response = "I'm sorry, I couldn't generate a response at this time. Please try again later."
            self.logger.info(
                f"Generated AI response of length {len(ai_response)} for voice message"
            )
            voice_intro = "🎤 **Voice Response:**"
            context_hint = ""
            if conversation_history and "name" in prompt.lower():
                has_name_context = any(
                    "name" in msg.get("content", "").lower()
                    for msg in conversation_history
                )
                if has_name_context:
                    context_hint = "_Based on our conversation..._\n\n"
                else:
                    context_hint = (
                        "_I don't have your name in our conversation history..._\n\n"
                    )
            elif conversation_history and len(conversation_history) > 0:
                context_hint = "_Continuing our conversation..._\n\n"
            voice_formatted_response = f"{voice_intro}\n\n{context_hint}{ai_response}"
            formatted_response = self.response_formatter.format_with_model_indicator(
                voice_formatted_response, model_indicator, quoted_text is not None
            )
            telegram_formatted_response = (
                await self.response_formatter.format_telegram_markdown(
                    formatted_response
                )
            )
            await self.response_formatter.safe_send_message(
                update.message, formatted_response
            )
            voice_enhanced_prompt = f"[Voice Message Transcribed]: {prompt}"
            await conversation_manager.save_message_pair(
                user_id, voice_enhanced_prompt, ai_response, active_model
            )
        except Exception as e:
            self.logger.error(
                f"Error processing voice message: {str(e)}", exc_info=True
            )
            if "text" in locals() and not text:
                error_message = "Sorry, I couldn't transcribe your voice message. Please try speaking more clearly or in a quieter environment."
            elif "prompt" in locals() and "ai_response" not in locals():
                error_message = "I understood your voice message, but I'm having trouble generating a response right now. Please try again later."
            else:
                error_message = "Sorry, there was an error processing your voice message. Please try again later."
            try:
                if "status_message" in locals() and status_message:
                    try:
                        await status_message.edit_text(error_message)
                    except Exception:
                        await update.message.reply_text(error_message)
                else:
                    await update.message.reply_text(error_message)
            except Exception as reply_error:
                self.logger.error(f"Failed to send error message: {str(reply_error)}")
    async def _handle_document_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle incoming document messages using unified conversation system."""
        user_id = update.effective_user.id
        self.logger.info(f"Processing document for user: {user_id}")
        self.telegram_logger.log_message("Received document message", user_id)

        try:
            if not update.message or not update.message.document:
                await update.message.reply_text(
                    "Sorry, I couldn't process this document."
                )
                return

            document = update.message.document
            if document.file_size and document.file_size > 50 * 1024 * 1024:
                await update.message.reply_text(
                    "📄 Sorry, this document is too large (max 50MB). Please send a smaller file."
                )
                return

            # Check if user's model supports documents
            supports_documents, model_id, error_message = await self.check_model_supports_media(user_id, "document")
            if not supports_documents:
                await update.message.reply_text(error_message, parse_mode="Markdown")
                return

            # Route through TextHandler for unified conversation processing
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )

            # Extract media files using TextHandler's method
            has_attached_media, media_files, media_type = await text_handler._extract_media_files(update, context)

            if has_attached_media and media_files:
                # Send processing message
                processing_message = await update.message.reply_text(
                    f"📄 Processing your document: {document.file_name}. Please wait..."
                )
                await context.bot.send_chat_action(
                    chat_id=update.effective_chat.id, action=ChatAction.TYPING
                )

                try:
                    # Use TextHandler's media analysis with conversation context
                    prompt = update.message.caption or f"Please analyze this {document.file_name} file."
                    await text_handler._handle_media_analysis(
                        update,
                        context,
                        processing_message,
                        media_files,
                        media_type,
                        prompt,
                        user_id,
                        model_id,
                    )

                    self.telegram_logger.log_message(
                        f"Document processed successfully via unified system: {document.file_name}",
                        user_id,
                    )

                    try:
                        await self.user_data_manager.update_stats(
                            user_id, document=True
                        )
                    except Exception as stats_error:
                        self.logger.warning(
                            f"Failed to update stats: {str(stats_error)}"
                        )

                except Exception as e:
                    self.logger.error(f"Document processing error: {str(e)}")
                    try:
                        await processing_message.edit_text(
                            "❌ Sorry, there was an error processing your document. Please try again."
                        )
                    except Exception:
                        await update.message.reply_text(
                            "❌ Sorry, there was an error processing your document. Please try again."
                        )
            else:
                await update.message.reply_text("Sorry, I couldn't extract the document data.")

        except Exception as e:
            self.logger.error(f"Error in document message handler: {str(e)}")
            if "RATE_LIMIT_EXCEEDED" in str(e).upper():
                await update.message.reply_text(
                    "The service is currently experiencing high demand. Please try again later."
                )
            else:
                await self._error_handler(update, context)
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        conversation_id = f"user_{user_id}"
        self.telegram_logger.log_message("Processing document", user_id)
        try:
            if update.effective_chat.type in ["group", "supergroup"]:
                caption = update.message.caption or ""
                caption_entities = []
                if update.message.caption_entities:
                    caption_entities = [
                        {
                            "type": entity.type,
                            "offset": entity.offset,
                            "length": entity.length,
                            "url": getattr(entity, "url", None),
                            "user": getattr(entity, "user", None),
                        }
                        for entity in update.message.caption_entities
                    ]
                if not BotUsernameHelper.is_bot_mentioned(
                    caption, context, entities=caption_entities
                ):
                    return
                else:
                    caption = BotUsernameHelper.remove_bot_mention(caption, context)
            else:
                caption = update.message.caption or "Please analyze this document."
            document = update.message.document
            file_name = document.file_name
            file_id = document.file_id
            file_extension = (
                os.path.splitext(file_name)[1][1:] if "." in file_name else ""
            )
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            status_message = await update.message.reply_text(
                f"Processing your {file_extension.upper()} document... This might take a moment."
            )
            document_file = await context.bot.get_file(file_id)
            file_content = await document_file.download_as_bytearray()
            document_file_obj = io.BytesIO(file_content)
            self.logger.info(
                f"Downloaded document: {file_name}, size: {len(file_content)} bytes, extension: {file_extension}"
            )
            prompt = (
                caption
                or f"Please analyze this {file_extension.upper()} file and provide a detailed summary."
            )
            self.logger.info(f"Starting document processing for {file_extension} file")
            if file_extension.lower() == "pdf":
                response = await self.document_processor.process_document_enhanced(
                    file=document_file_obj, file_extension=file_extension, prompt=prompt
                )
            else:
                response = await self.document_processor.process_document_from_file(
                    file=document_file_obj, file_extension=file_extension, prompt=prompt
                )
            self.logger.info(
                f"Document processing completed. Response success: {response.get('success', False) if response else False}"
            )
            try:
                await status_message.delete()
            except Exception as delete_error:
                self.logger.warning(
                    f"Failed to delete document status message: {str(delete_error)}"
                )
            if response:
                response_text = response.get(
                    "result", "Document processed successfully."
                )
                document_id = response.get("document_id", "Unknown")
                formatted_response = (
                    f"**Document Analysis Completed**\n\n"
                    f"{response_text}\n\n"
                    f"**Document ID:** {document_id}"
                )
                await self.response_formatter.safe_send_message(
                    update.message,
                    formatted_response,
                    disable_web_page_preview=True,
                )
                try:
                    if (
                        hasattr(self.text_handler, "memory_manager")
                        and self.text_handler.memory_manager
                    ):
                        memory_manager = self.text_handler.memory_manager
                        document_prompt = f"[Document submitted: {file_name}] {prompt}"
                        await memory_manager.add_user_message(
                            conversation_id,
                            document_prompt,
                            str(user_id),
                            document_type=file_extension,
                            document_name=file_name,
                            is_document=True,
                        )
                        document_summary = (
                            response_text[:500] + "..."
                            if len(response_text) > 500
                            else response_text
                        )
                        document_response = (
                            f"[Document analysis of {file_name}]: {document_summary}"
                        )
                        await memory_manager.add_assistant_message(
                            conversation_id, document_response
                        )
                        self.logger.info(
                            f"Document interaction stored in memory manager for user {user_id}"
                        )
                except Exception as memory_error:
                    self.logger.error(
                        f"Error storing document in memory manager: {str(memory_error)}"
                    )
            else:
                await update.message.reply_text(
                    "Sorry, I couldn't analyze the document. Please try again."
                )
        except ValueError as ve:
            self.logger.error(f"ValueError processing document: {str(ve)}")
            await update.message.reply_text(f"Document processing error: {str(ve)}")
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            self.logger.error(
                f"Document processing traceback: {traceback.format_exc()}"
            )
            await update.message.reply_text(
                f"Sorry, I couldn't process your document. Error: {str(e)[:100]}..."
            )
    async def _error_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
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
                MessageHandler(
                    filters.TEXT & ~filters.COMMAND, self._handle_text_message
                )
            )
            application.add_handler(
                MessageHandler(filters.PHOTO, self._handle_image_message)
            )
            application.add_handler(
                MessageHandler(filters.VOICE, self._handle_voice_message)
            )
            application.add_handler(
                MessageHandler(filters.Document.ALL, self.handle_document)
            )
            application.add_error_handler(self._error_handler)
            self.logger.info("Message handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register message handlers: {str(e)}")
            raise Exception("Failed to register message handlers") from e
    async def handle_awaiting_doc_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Process text when awaiting document content"""
        if context.user_data.get("awaiting_doc_text"):
            context.user_data["awaiting_doc_text"] = False
            content = update.message.text
            self.logger.info(
                f"📝 Custom text received for export - User ID: {update.effective_user.id}"
            )
            self.logger.info(
                f"📝 Content length: {len(content) if content else 0} characters"
            )
            self.logger.info(
                f"📝 Content preview: {content[:200] if content else 'None'}..."
            )
            context.user_data["doc_export_text"] = content
            stored_content = context.user_data.get("doc_export_text", "")
            self.logger.info(
                f"📝 Verified storage - Length: {len(stored_content)} characters"
            )
            format_options = [
                [
                    InlineKeyboardButton(
                        "📄 PDF Format", callback_data="export_format_pdf"
                    ),
                    InlineKeyboardButton(
                        "📝 DOCX Format", callback_data="export_format_docx"
                    ),
                ]
            ]
            format_markup = InlineKeyboardMarkup(format_options)
            await update.message.reply_text(
                f"✅ Your text has been received ({len(content)} characters). Please select the document format you want to export to:",
                reply_markup=format_markup,
            )
            return True
        return False
    async def handle_awaiting_doc_image(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Process image when awaiting document content"""
        if context.user_data.get("awaiting_doc_image"):
            context.user_data["awaiting_doc_image"] = False
            image_file = update.message.photo[-1].get_file()
            context.user_data["doc_export_image"] = image_file
            format_options = [
                [
                    InlineKeyboardButton(
                        "📄 PDF Format", callback_data="export_format_pdf"
                    ),
                ]
            ]
            format_markup = InlineKeyboardMarkup(format_options)
            await update.message.reply_text(
                "Please select the document format you want to export to:",
                reply_markup=format_markup,
            )
            await update.message.reply_text(
                "Please select the document format you want to export to:",
                reply_markup=format_markup,
            )
