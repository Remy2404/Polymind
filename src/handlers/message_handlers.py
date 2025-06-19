# Standard library imports
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

# Import utility classes
from src.handlers.message_context_handler import MessageContextHandler
from src.handlers.response_formatter import ResponseFormatter
from src.services.media.voice_config import VoiceConfig, VoiceQuality
from src.services.user_preferences_manager import UserPreferencesManager
from src.services.media.voice_processor import create_voice_processor
from src.services.memory_context.conversation_manager import ConversationManager
from src.services.group_chat.integration import GroupChatIntegration
from src.services.model_handlers.model_configs import ModelConfigurations, Provider, ModelConfig

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

        # Initialize multimodal processor
        self.multimodal_processor = TelegramMultimodalProcessor(gemini_api)
        self.deepseek_api = deepseek_api
        self.openrouter_api = openrouter_api

        # Initialize essential utility classes only
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()

        # Initialize document processor
        self.document_processor = DocumentProcessor(gemini_api)

        # Initialize AI command router
        self.ai_command_router = None
        if command_handlers:
            self.ai_command_router = AICommandRouter(command_handlers, gemini_api)

        # Initialize conversation manager (will be lazy-loaded with proper dependencies)
        self._conversation_manager = None

        # Initialize group chat integration
        self._group_chat_integration = None

        # Initialize all available models from ModelConfigurations
        self.all_models = ModelConfigurations.get_all_models()
        self.logger.info(f"Initialized with {len(self.all_models)} available models")
        
        # Log available models by provider for debugging
        for provider in Provider:
            provider_models = ModelConfigurations.get_models_by_provider(provider)
            if provider_models:
                self.logger.info(f"{provider.value.title()} models: {len(provider_models)} available")
                # Log some model names for verification
                model_names = list(provider_models.keys())[:5]  # First 5 models
                self.logger.debug(f"  Examples: {model_names}")

        # Log total free models count
        free_models = ModelConfigurations.get_free_models()
        self.logger.info(f"Free OpenRouter models available: {len(free_models)}")

        # Verify some commonly used models are available
        common_models = ["llama-3.3-8b", "deepseek-r1-zero", "gemini", "deepseek"]
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
            self.logger.info(f"  - OpenRouter key: {model_config.openrouter_model_key or 'N/A'}")
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
            "model_ids": list(self.all_models.keys())[:10]  # First 10 model IDs for reference
        }

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get model configuration by ID."""
        return self.all_models.get(model_id)

    def get_model_indicator_and_config(self, model_id: str) -> tuple[str, ModelConfig]:
        """Get model indicator emoji and configuration for a model."""
        model_config = self.get_model_config(model_id)
        if model_config:
            return f"{model_config.indicator_emoji} {model_config.display_name}", model_config
        else:
            # Fallback for unknown models
            self.logger.warning(f"Unknown model ID: {model_id}, using default")
            return "🤖 Unknown Model", None

    async def generate_ai_response(self, prompt: str, model_id: str, user_id: int, conversation_context: list = None) -> str:
        """Generate AI response using the specified model with conversation context."""
        model_config = self.get_model_config(model_id)
        
        if not model_config:
            self.logger.error(f"Model configuration not found for: {model_id}")
            return f"Sorry, the model '{model_id}' is not available."

        # Enhanced conversation context debugging
        if conversation_context:
            self.logger.info(f"🧠 AI Context Debug - Model: {model_id}")
            self.logger.info(f"   └─ Context messages: {len(conversation_context)}")
            
            # Show context summary
            for i, msg in enumerate(conversation_context[-5:]):  # Last 5 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                preview = content[:100] + ("..." if len(content) > 100 else "")
                self.logger.info(f"   └─ Message {i+1} [{role.upper()}]: {preview}")
                
                # Highlight name mentions
                if 'name' in content.lower():
                    self.logger.info(f"   └─ ⭐ Contains name/identity information!")
                    
            # Log the current prompt for reference
            self.logger.info(f"   └─ Current prompt: {prompt[:100]}...")
        else:
            self.logger.info(f"⚠ No conversation context provided for model: {model_id}")

        try:
            self.logger.info(f"Generating response with model: {model_id}")
            self.logger.info(f"Attempting to use {model_config.provider.value.title()} API with {model_id} model")
            
            if model_config.provider == Provider.GEMINI:
                self.logger.info("Successfully used Gemini API")
                return await self.gemini_api.generate_response(prompt)
                
            elif model_config.provider == Provider.DEEPSEEK and hasattr(self, "deepseek_api") and self.deepseek_api:
                self.logger.info("Successfully used DeepSeek API")
                return await self.deepseek_api.generate_response(prompt)
                
            elif model_config.provider == Provider.OPENROUTER and hasattr(self, "openrouter_api") and self.openrouter_api:
                # Use the openrouter_model_key for proper API mapping
                if model_config.openrouter_model_key:
                    self.logger.info("Successfully used OpenRouter API with model key")
                    response = await self.openrouter_api.generate_response_with_model_key(
                        prompt=prompt,
                        openrouter_model_key=model_config.openrouter_model_key,
                        context=conversation_context
                    )
                else:
                    # Fix the parameter order - use named parameters with conversation context
                    self.logger.info("Successfully used OpenRouter API with model ID")
                    response = await self.openrouter_api.generate_response(
                        prompt=prompt, 
                        context=conversation_context, 
                        model=model_id
                    )
                
                # Debug the AI response for context usage
                if response and conversation_context:
                    context_keywords = ['name', 'your name', 'my name']
                    if any(keyword in response.lower() for keyword in context_keywords):
                        self.logger.info("✅ AI response appears to use conversation context")
                    else:
                        self.logger.warning("⚠ AI response may not be using conversation context effectively")
                
                return response
                    
            else:
                self.logger.warning(f"No API available for provider: {model_config.provider.value}")
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

            # Check if we're waiting for document content
            if update.message and await self.handle_awaiting_doc_text(update, context):
                # The message was handled as document text input, stop further processing
                return

            # Check if we're waiting for AI document topic
            if update.message and context.user_data.get("awaiting_aidoc_topic"):
                # Clear the flag
                context.user_data["awaiting_aidoc_topic"] = False
                # Store the topic
                context.user_data["aidoc_prompt"] = update.message.text
                # Show format selection
                from handlers.command_handlers import CommandHandlers

                command_handler = CommandHandlers(
                    self.gemini_api,
                    self.user_data_manager,
                    self.telegram_logger,
                    None,  # flux_lora_image_generator not needed here
                )
                await command_handler._show_ai_document_format_selection(
                    update, context
                )
                return

            self.logger.info(
                f"Received text message from user {user_id}: {message_text}"
            )

            # Initialize user data if not already initialized
            await self.user_data_manager.initialize_user(user_id)

            # Initialize group chat integration if needed
            if self._group_chat_integration is None and self._conversation_manager:
                self._group_chat_integration = GroupChatIntegration(
                    self.user_data_manager, self._conversation_manager
                )

            # Process group chat features if applicable
            enhanced_message_text = message_text
            group_metadata = {}

            chat = update.effective_chat
            if (
                self._group_chat_integration
                and chat
                and chat.type in ["group", "supergroup"]
            ):

                try:
                    enhanced_message_text, group_metadata = (
                        await self._group_chat_integration.process_group_message(
                            update, context, message_text
                        )
                    )
                    self.logger.info(f"Enhanced group message for chat {chat.id}")
                except Exception as e:
                    self.logger.error(f"Error processing group message: {e}")
                    # Fall back to original message if group processing fails

            # Check if the bot is mentioned but don't send an automatic reply
            # Just log it for tracking purposes
            bot_username = "@Gemini_AIAssistBot"
            if bot_username in enhanced_message_text:
                self.logger.info(f"Bot mentioned by user {user_id}")
                # Remove the automatic greeting that was causing duplicate responses
                # We'll let the text handler process the full message instead

            # Try AI command routing first (if available)
            if self.ai_command_router:
                try:
                    # Only route commands if private chat or bot is mentioned in group chat
                    is_group_chat = chat and chat.type in ["group", "supergroup"]
                    is_mentioned = "@Gemini_AIAssistBot" in enhanced_message_text
                    if not is_group_chat or is_mentioned:
                        # IMPORTANT: Use original message_text for intent detection, not enhanced_message_text
                        # This prevents group context from interfering with command detection
                        should_route = (
                            await self.ai_command_router.should_route_message(
                                message_text
                            )
                        )
                        if should_route:
                            intent, confidence = (
                                await self.ai_command_router.detect_intent(message_text)
                            )
                            self.logger.info(
                                f"Routing message to command: {intent.value} (confidence: {confidence:.2f})"
                            )

                            # Attempt to route the command using original message
                            command_executed = (
                                await self.ai_command_router.route_command(
                                    update, context, intent, message_text
                                )
                            )

                            if command_executed:
                                # Command was successfully executed, update stats and return
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
                                # Command routing failed, fall back to normal text processing
                                self.logger.info(
                                    "Command routing failed, falling back to normal text processing"
                                )
                except Exception as e:
                    self.logger.error(f"Error in AI command routing: {str(e)}")
                    # Fall back to normal text processing on any error

            # Create text handler instance with all required API instances
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                openrouter_api=self.openrouter_api,  # Pass the openrouter_api for models like llama4_maverick
                deepseek_api=self.deepseek_api,  # Pass the deepseek_api for deepseek model
            )

            # Process the message (use enhanced message for group chats)
            # Create a temporary update with enhanced message for group processing
            if (
                enhanced_message_text != message_text
                and chat
                and chat.type in ["group", "supergroup"]
            ):
                # Store original message and metadata in context for the text handler
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
        # We already called update_stats above inside the try block, no need to call it again here

    async def _handle_image_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming image messages using the new multimodal system."""
        try:
            user_id = update.effective_user.id
            self.logger.info(f"Processing image from user {user_id}")
            self.telegram_logger.log_message("Received image message", user_id)

            # Check if we have a valid image
            if (
                not update.message
                or not update.message.photo
                or len(update.message.photo) == 0
            ):
                await update.message.reply_text("Sorry, I couldn't process this image.")
                return

            # Show processing message
            processing_message = await update.message.reply_text(
                "🖼️ Processing your image. Please wait..."
            )

            # Send typing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            try:
                # Get conversation context for the user
                context_messages = []
                try:
                    # Get recent conversation history if available
                    if hasattr(self, "conversation_manager"):
                        context_messages = await self.conversation_manager.get_context(
                            user_id
                        )
                except Exception:
                    pass  # Continue without context if unavailable

                # Process the message with multimodal processor
                result = await self.multimodal_processor.process_telegram_message(
                    message=update.message, context=context_messages
                )

                if result.success and result.content:
                    # Format the response
                    formatted_response = await self.response_formatter.format_response(
                        result.content, user_id, model_name="gemini-2.0-flash"
                    )

                    # Delete processing message and send response
                    await processing_message.delete()
                    await update.message.reply_text(
                        formatted_response,
                        parse_mode=(
                            "Markdown"
                            if "```" in formatted_response or "*" in formatted_response
                            else None
                        ),
                    )

                    # Log successful processing
                    self.telegram_logger.log_message(
                        f"Image processed successfully", user_id
                    )

                else:
                    # Handle error
                    await processing_message.edit_text(
                        f"❌ Sorry, I couldn't process your image: {result.error or 'Unknown error'}"
                    )

            except Exception as e:
                self.logger.error(f"Image processing error: {str(e)}")
                await processing_message.edit_text(
                    "❌ Sorry, there was an error processing your image. Please try again."
                )

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

        # Extract quote context using MessageContextHandler
        quoted_text, quoted_message_id = self.context_handler.extract_reply_context(
            update.message
        )

        try:
            # Factory initialization moved here
            if not hasattr(self, "voice_processor") or self.voice_processor is None:
                self.voice_processor = await create_voice_processor()
                # Initialize preferences manager
                self.preferences_manager = UserPreferencesManager(self.user_data_manager)

            # Get user's language preference
            user_lang = await self.preferences_manager.get_user_language_preference(user_id) or "en"

            # Try to detect if user has previously sent Khmer messages
            user_history = await self.user_data_manager.get_user_data(user_id)
            has_khmer_history = False
            
            if user_history and 'message_history' in user_history:
                # Check last 15 messages for Khmer content (increased from 10)
                for msg in user_history.get('message_history', [])[-15:]:
                    if msg.get('content') and any(char in msg.get('content', '') for char in 'កខគឃងចឆជឈញ'):
                        has_khmer_history = True
                        self.logger.info(f"User {user_id} has Khmer message history, will prioritize Khmer detection")
                        break
            
            # Look for Khmer in recent voice transcription history
            if not has_khmer_history and user_history and 'voice_transcriptions' in user_history:
                # Check past voice transcriptions for Khmer language detection
                for transcription in user_history.get('voice_transcriptions', [])[-10:]:
                    if transcription.get('detected_language') == 'km':
                        has_khmer_history = True
                        self.logger.info(f"User {user_id} has previous Khmer voice transcriptions, will prioritize Khmer detection")
                        break

            # If not found in preferences, use Telegram's language code
            if user_lang == "en" and update.effective_user.language_code:
                user_lang = update.effective_user.language_code
                
            # Force Khmer detection if user has Khmer history
            if has_khmer_history and user_lang == "en":
                self.logger.info(f"Overriding default language (en) with Khmer (km) based on user history")
                user_lang = "km"

            # Enhanced language mapping with better Khmer support
            language_map = {
                "en": "en-US", "km": "km-KH", "kh": "km-KH", "ru": "ru-RU", 
                "fr": "fr-FR", "es": "es-ES", "de": "de-DE", "ja": "ja-JP",
                "zh": "zh-CN", "th": "th-TH", "vi": "vi-VN",
            }

            # Extract language prefix properly
            lang_prefix = user_lang.split("-")[0] if "-" in user_lang else user_lang
            lang = language_map.get(lang_prefix, "en-US")

            # Set flag for Khmer processing
            is_khmer = lang_prefix in ["km", "kh"]

            # Log the detected language for debugging
            self.logger.info(f"Voice recognition language set to: {lang}, is_khmer={is_khmer}")

            # Show processing message
            processing_text = (
                "កំពុងដំណើរការសារសំឡេងរបស់អ្នក... សូមរង់ចាំ...\n(Processing your voice message. Please wait...)"
                if is_khmer
                else "🎤 Processing your voice message with enhanced AI recognition..."
            )

            status_message = await update.message.reply_text(processing_text)

            # Use enhanced VoiceProcessor for downloading and converting voice file
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            ogg_file_path, wav_file_path = await self.voice_processor.download_and_convert(
                voice_file, str(user_id), is_khmer
            )

            # Get audio duration for short audio detection optimization
            audio_duration = 0
            try:
                import soundfile as sf
                with sf.SoundFile(wav_file_path) as sound_file:
                    audio_duration = float(len(sound_file)) / sound_file.samplerate
                self.logger.info(f"Audio duration: {audio_duration:.2f}s")
            except Exception as e:
                self.logger.warning(f"Could not determine audio duration: {e}")

            # Special handling for short audio - increase Khmer detection sensitivity
            short_audio_khmer_mode = False
            if audio_duration > 0 and audio_duration < 3.0 and not is_khmer:
                self.logger.info(f"Short audio detected ({audio_duration:.2f}s), increasing Khmer detection sensitivity")
                # For very short audio, we'll give extra weight to Khmer detection
                short_audio_khmer_mode = True
            
            # Enhanced transcription with audio duration awareness
            if hasattr(self.voice_processor, "get_best_transcription"):
                self.logger.info(f"🎤 Enhanced voice transcription starting for language: {lang}")
                
                # For short audio, first try with language detection auto mode
                if short_audio_khmer_mode:
                    self.logger.info("Short audio detected, first trying with auto language detection")
                    # Start with auto-detection to see if Khmer is naturally detected
                    text, recognition_language, metadata = await self.voice_processor.get_best_transcription(
                        wav_file_path, language="auto", confidence_threshold=0.5
                    )
                    # If detected as Khmer, set is_khmer flag and update lang
                    if recognition_language == "km":
                        self.logger.info("Auto-detection identified Khmer language!")
                        is_khmer = True
                        lang = "km-KH"
                        # Update status message
                        try:
                            await status_message.edit_text("🇰🇭 ភាសាខ្មែរត្រូវបានបង្ហាញ... កំពុងដំណើរការ\n(Khmer language detected... processing)")
                        except Exception:
                            pass
                
                # Do normal transcription with the right language setting
                text, recognition_language, metadata = await self.voice_processor.get_best_transcription(
                    wav_file_path, language=lang, confidence_threshold=0.6
                )
                engine_used = metadata.get("engine", "unknown")
                confidence = metadata.get("confidence", 0.0)
                
                # Detect false-positive English transcription for Khmer audio
                if not is_khmer:
                    from src.services.media.voice_config import VoiceConfig
                    
                    # More aggressively detect false positives for short audio or if user has Khmer history
                    confidence_threshold = 0.6
                    if short_audio_khmer_mode or has_khmer_history:
                        confidence_threshold = 0.5
                    
                    is_false_positive = VoiceConfig.is_likely_false_english_for_khmer(text, confidence)
                        
                    if is_false_positive or (
                        (short_audio_khmer_mode or has_khmer_history) and confidence < confidence_threshold
                    ):
                        self.logger.warning(f"Detected likely Khmer audio mis-transcribed as English (confidence: {confidence:.2f})")
                        is_khmer = True
                        lang = "km-KH"
                        try:
                            await status_message.edit_text("🇰🇭 អត្ថបទមើលទៅដូចជាភាសាខ្មែរ កំពុងបកប្រែឡើងវិញ...\n(Detected Khmer speech, retrying with better settings...)")
                        except Exception:
                            pass
                        # Rerun transcription using Khmer enhanced settings and lower threshold
                        text, recognition_language, metadata = await self.voice_processor.get_best_transcription(
                            wav_file_path,
                            language=lang,
                            confidence_threshold=VoiceConfig.get_confidence_threshold("faster_whisper_khmer")
                        )
                        engine_used = metadata.get("engine", "unknown")
                        confidence = metadata.get("confidence", 0.0)
                
                # Enhanced logging for Khmer debugging with new metadata
                requested_lang = metadata.get("requested_language", "unknown")
                detected_lang = metadata.get("detected_language", "unknown")
                strategy_used = metadata.get("strategy", "unknown")
                language_mismatch = metadata.get("language_mismatch", False)
                
                self.logger.info(f"🔍 VOICE TRANSCRIPTION RESULT:")
                self.logger.info(f"  → Requested language: {requested_lang}")
                self.logger.info(f"  → Detected language: {detected_lang}")
                self.logger.info(f"  → Recognition language: {recognition_language}")
                self.logger.info(f"  → Engine: {engine_used}")
                self.logger.info(f"  → Strategy: {strategy_used}")
                self.logger.info(f"  → Confidence: {confidence:.3f}")
                self.logger.info(f"  → Text length: {len(text)} chars")
                self.logger.info(f"  → Text preview: {text[:100]}...")
                self.logger.info(f"  → Language mismatch: {language_mismatch}")
                
                # Store transcription in user history for future reference
                try:
                    # Initialize voice_transcriptions array if needed
                    if 'voice_transcriptions' not in user_history:
                        user_history['voice_transcriptions'] = []
                    
                    # Add this transcription to history
                    user_history['voice_transcriptions'].append({
                        'timestamp': str(update.message.date),
                        'detected_language': detected_lang,
                        'requested_language': requested_lang,
                        'confidence': confidence,
                        'is_khmer': is_khmer,
                        'duration': audio_duration,
                        'strategy': strategy_used
                    })
                    
                    # Keep only last 20 transcriptions
                    if len(user_history['voice_transcriptions']) > 20:
                        user_history['voice_transcriptions'] = user_history['voice_transcriptions'][-20:]
                    
                    # Save updates
                    await self.user_data_manager.update_user_data(user_id, user_history)
                except Exception as e:
                    self.logger.warning(f"Could not update voice transcription history: {e}")
                
                # Special warning for Khmer cases with enhanced information
                if is_khmer and language_mismatch:
                    self.logger.warning(f"⚠️ KHMER LANGUAGE MISMATCH DETECTED:")
                    self.logger.warning(f"  → Expected: km (Khmer)")
                    self.logger.warning(f"  → Got: {detected_lang}")
                    self.logger.warning(f"  → Strategy used: {strategy_used}")
                    self.logger.warning(f"  → All attempts tried: {metadata.get('all_attempts', 0)}")
                    self.logger.warning(f"  → This suggests the audio might not be clear Khmer speech")
                    self.logger.warning(f"  → Or the model is auto-detecting a different language")
                    
                    # Add user notification for Khmer mismatch with strategy info
                    if detected_lang == "en":
                        # Check if this might be a false positive English transcription
                        from src.services.media.voice_config import VoiceConfig
                        is_false_positive = VoiceConfig.is_likely_false_english_for_khmer(text, confidence)
                        
                        if is_false_positive:
                            khmer_warning = (
                                f"🇰🇭 **ការជូនដំណឹង:** ការបំលែងនេះអាចមិនត្រឹមត្រូវទេ។\n"
                                f"📢 **Notice:** This transcription may be incorrect (likely false positive).\n\n"
                                f"🔍 **បញ្ហា / Issue:** អក្សរដែលបានបំលែងនេះទំនងមានភាពមិនត្រឹមត្រូវ\n"
                                f"🔍 **Issue:** The transcribed text appears to be a false positive\n\n"
                                f"💡 **ការណែនាំ / Recommendations:**\n"
                                f"• និយាយឱ្យច្បាស់និងយឺត / Speak clearly and slowly\n"
                                f"• ប្រើពាក្យខ្មែរសុទ្ធ / Use pure Khmer words\n"
                                f"• ជៀសវាងសំឡេងរំខាន / Avoid background noise\n"
                                f"• សាកល្បងផ្ញើម្តងទៀត / Try sending again\n\n"
                                f"📊 **Technical:** Strategy: `{strategy_used}`, Confidence: {confidence:.1%}"
                            )
                        else:
                            khmer_warning = (
                                f"🇰🇭 **ការជូនដំណឹង:** សំឡេងរបស់អ្នកត្រូវបានសម្គាល់ជាភាសាអង់គ្លេសជំនួសឱ្យភាសាខ្មែរ។\n"
                                f"📢 **Notice:** Your voice was detected as English instead of Khmer.\n\n"
                                f"🔧 **ព័ត៌មានបច្ចេកទេស / Technical Info:**\n"
                                f"• វិធីសាស្រ្ត / Strategy: `{strategy_used}`\n"
                                f"• ការព្យាយាម / Attempts: {metadata.get('all_attempts', 0)}\n"
                                f"• កម្រិតទំនុកចិត្ត / Confidence: {confidence:.1%}\n\n"
                                f"💡 **ការណែនាំ / Tips:**\n"
                                f"• និយាយឱ្យច្បាស់និងយឺត / Speak clearly and slowly\n"
                                f"• ជៀសវាងសំឡេងរំខាន / Avoid background noise\n"
                                f"• ប្រើឃ្លាខ្មែរសុទ្ធ / Use pure Khmer phrases\n"
                                f"• និយាយយូរជាង / Speak for longer duration\n"
                                f"• សាកល្បងផ្ញើម្តងទៀត / Try sending again\n\n"
                            )
                        
                        # Send warning but continue with transcription
                        try:
                            await update.message.reply_text(khmer_warning, parse_mode="Markdown")
                        except Exception:
                            await update.message.reply_text(
                                f"🇰🇭 Notice: Your Khmer voice was detected as English. "
                                f"Strategy used: {strategy_used}. Please try speaking more clearly in pure Khmer."
                            )
                
                self.logger.info(f"Enhanced transcription: engine={engine_used}, confidence={confidence:.2f}")
            else:
                # Fallback to basic transcription
                text, recognition_language = await self.voice_processor.transcribe(wav_file_path, lang, is_khmer)
                metadata = {"engine": "basic", "confidence": 0.7}
                engine_used = "basic"
                confidence = 0.7

            if not text:
                # Language-specific error message
                error_text = (
                    "សូមអភ័យទោស មិនអាចយល់សំឡេងបានទេ។ សូមសាកល្បងម្តងទៀតជាមួយសំឡេងច្បាស់ជាងនេះ។\n\n"
                    "Sorry, I couldn't understand the audio. Please try again with clearer audio."
                    if is_khmer
                    else "❌ Sorry, I couldn't understand the audio.\n\n💡 **Tips:**\n"
                    "• Speak clearly and avoid background noise\n"
                    "• Try speaking in English for better accuracy\n"
                    "• Send shorter voice messages (under 30 seconds)\n"
                    "• Make sure you're in a quiet environment"
                )

                # Update status message
                try:
                    await status_message.edit_text(error_text, parse_mode="Markdown")
                except Exception:
                    await update.message.reply_text(error_text, parse_mode="Markdown")
                return

            # Delete the status message safely
            try:
                await status_message.delete()
            except Exception as msg_error:
                self.logger.warning(f"Could not delete status message: {str(msg_error)}")

            # Show the transcribed text with confidence indicator
            confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            
            # Improved formatting for voice message transcription
            if is_khmer:
                # For Khmer, show language detection status
                khmer_detected = metadata.get("khmer_detected", False)
                strategy_used = metadata.get("strategy", "unknown")
                
                if khmer_detected:
                    transcript_text = f"🎤 *សំឡេងបានបំលែងជាអក្សរ* ✅\n\n{text}"
                    if strategy_used:
                        transcript_text += f"\n\n_វិធីសាស្រ្ត: {strategy_used}_"
                else:
                    # Khmer requested but not detected
                    transcript_text = (
                        f"🎤 *Voice Transcribed* ⚠️\n\n{text}\n\n"
                        f"🇰🇭 *ចំណាំ: មិនបានសម្គាល់ជាភាសាខ្មែរទេ*\n"
                        f"*Note: Khmer not detected (got: {recognition_language})*"
                    )
            else:
                transcript_text = f"🎤 **Voice Message Transcribed** {confidence_emoji}\n\n{text}"
                
                # Add engine info only if confidence is good and not Khmer
                if confidence > 0.8:
                    transcript_text += f"\n\n_Engine: {engine_used.title()}, Confidence: {confidence:.1%}_"
                    
                # Add strategy info if available
                strategy_used = metadata.get("strategy")
                if strategy_used and strategy_used != "unknown":
                    transcript_text += f"\n_Strategy: {strategy_used}_"

            # Send transcript message
            try:
                await self.response_formatter.safe_send_message(update.message, transcript_text)
            except Exception as reply_error:
                self.logger.error(f"Error sending transcript message: {str(reply_error)}")
                await update.message.reply_text(f"🎤 Transcription: \n{text}")

            # Log the transcribed text
            self.telegram_logger.log_message(f"Transcribed {recognition_language} text: {text}", user_id)

            # Initialize user data if not already initialized
            await self.user_data_manager.initialize_user(user_id)

            # Use the TextHandler's conversation manager instead of creating a separate one
            # This ensures voice and text messages share the same conversation context
            if hasattr(self.text_handler, 'conversation_manager'):
                conversation_manager = self.text_handler.conversation_manager
            else:
                # Fallback: create our own if text handler doesn't have one
                if not hasattr(self, "_conversation_manager") or not self._conversation_manager:
                    # Create text handler for conversation manager
                    text_handler = TextHandler(
                        self.gemini_api, self.user_data_manager,
                        self.openrouter_api if hasattr(self, "openrouter_api") else None,
                        self.deepseek_api if hasattr(self, "deepseek_api") else None,
                    )
                    self._conversation_manager = ConversationManager(
                        text_handler.memory_manager, text_handler.model_history_manager
                    )
                conversation_manager = self._conversation_manager

            # Save voice interaction to shared conversation memory
            await conversation_manager.save_media_interaction(
                user_id, "voice", text, f"I've transcribed your voice message which said: {text}"
            )

            # Process the transcribed text with AI
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

            # Prepare prompt with quoted text context if it exists
            prompt = text
            if quoted_text:
                prompt = self.context_handler.format_prompt_with_quote(text, quoted_text)

            # Get user's selected model efficiently
            user_settings = await self.user_data_manager.get_user_settings(str(user_id))
            preferred_model = await self.user_data_manager.get_user_preference(user_id, "preferred_model", None)
            
            # Determine active model
            active_model = preferred_model or user_settings.get("active_model", "gemini")

            # Log model selection
            self.logger.info(f"Voice message model selection - User: {user_id}")
            self.logger.info(f" - preferred_model from preferences: {preferred_model}")
            self.logger.info(f" - active_model from user settings: {user_settings.get('active_model', 'not set')}")
            self.logger.info(f"SELECTED MODEL: '{active_model}' for voice response. Prompt: {prompt[:50]}...")

            # Get model indicator and config
            model_indicator, model_config = self.get_model_indicator_and_config(active_model)
            self.logger.info(f"Using model indicator: {model_indicator} for model: {active_model}")

            # Get conversation history for context (same as text handler)
            try:
                conversation_history = await conversation_manager.get_conversation_history(
                    user_id, max_messages=10, model=active_model
                )
                self.logger.info(f"Retrieved {len(conversation_history)} conversation history messages for voice response")
                
                # Enhanced debug: Log conversation context for troubleshooting
                if conversation_history:
                    self.logger.info(f"Voice context debug - First message: {conversation_history[0].get('content', 'N/A')[:100]}...")
                    self.logger.info(f"Voice context debug - Last message: {conversation_history[-1].get('content', 'N/A')[:100]}...")
                else:
                    self.logger.warning("No conversation history found for voice message context")
                    
            except Exception as e:
                self.logger.warning(f"Failed to retrieve conversation history: {str(e)}")
                conversation_history = None

            # Generate AI response with conversation context
            ai_response = await self.generate_ai_response(prompt, active_model, user_id, conversation_history)

            if not ai_response:
                self.logger.warning(f"Empty AI response for user {user_id} with active model {active_model}")
                ai_response = "I'm sorry, I couldn't generate a response at this time. Please try again later."

            # Log successful response generation
            self.logger.info(f"Generated AI response of length {len(ai_response)} for voice message")

            # Enhanced voice message formatting with better context awareness
            voice_intro = "🎤 **Voice Response:**"
            
            # Add context cue based on conversation content
            context_hint = ""
            if conversation_history and "name" in prompt.lower():
                # Check if we have name context
                has_name_context = any(
                    'name' in msg.get('content', '').lower()
                    for msg in conversation_history
                )
                if has_name_context:
                    context_hint = "_Based on our conversation..._\n\n"
                else:
                    context_hint = "_I don't have your name in our conversation history..._\n\n"
            elif conversation_history and len(conversation_history) > 0:
                context_hint = "_Continuing our conversation..._\n\n"
            
            # Format the response specifically for voice interaction
            voice_formatted_response = f"{voice_intro}\n\n{context_hint}{ai_response}"
            
            # Apply model indicator formatting
            formatted_response = self.response_formatter.format_with_model_indicator(
                voice_formatted_response, model_indicator, quoted_text is not None
            )

            # Format for Telegram (same as text handler) - this was missing!
            telegram_formatted_response = await self.response_formatter.format_telegram_markdown(
                formatted_response
            )

            # Send the response using the response formatter for proper formatting
            await self.response_formatter.safe_send_message(update.message, formatted_response)

            # Save the conversation pair with voice message indicator for consistency
            voice_enhanced_prompt = f"[Voice Message Transcribed]: {prompt}"
            await conversation_manager.save_message_pair(user_id, voice_enhanced_prompt, ai_response, active_model)

        except Exception as e:
            self.logger.error(f"Error processing voice message: {str(e)}", exc_info=True)
            
            # Determine appropriate error message based on processing stage
            if "text" in locals() and not text:
                error_message = "Sorry, I couldn't transcribe your voice message. Please try speaking more clearly or in a quieter environment."
            elif "prompt" in locals() and "ai_response" not in locals():
                error_message = "I understood your voice message, but I'm having trouble generating a response right now. Please try again later."
            else:
                error_message = "Sorry, there was an error processing your voice message. Please try again later."

            # Send error message
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
        """Handle incoming document messages using the new multimodal system."""
        user_id = update.effective_user.id
        self.logger.info(f"Processing document for user: {user_id}")
        self.telegram_logger.log_message("Received document message", user_id)

        try:
            # Check if we have a valid document
            if not update.message or not update.message.document:
                await update.message.reply_text(
                    "Sorry, I couldn't process this document."
                )
                return

            document = update.message.document

            # Check file size (50MB limit)
            if document.file_size and document.file_size > 50 * 1024 * 1024:
                await update.message.reply_text(
                    "📄 Sorry, this document is too large (max 50MB). Please send a smaller file."
                )
                return

            # Show processing message
            processing_message = await update.message.reply_text(
                f"📄 Processing your document: {document.file_name}. Please wait..."
            )

            # Send typing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            try:
                # Get conversation context for the user
                context_messages = []
                try:
                    if hasattr(self, "conversation_manager"):
                        context_messages = await self.conversation_manager.get_context(
                            user_id
                        )
                except Exception:
                    pass  # Continue without context if unavailable

                # Process the message with multimodal processor
                result = await self.multimodal_processor.process_telegram_message(
                    message=update.message, context=context_messages
                )

                if result.success and result.content:
                    # Format the response
                    formatted_response = await self.response_formatter.format_response(
                        result.content, user_id, model_name="gemini-2.0-flash"
                    )

                    # Delete processing message and send response
                    await processing_message.delete()

                    # Split long responses into chunks
                    response_chunks = await self.response_formatter.split_long_message(
                        formatted_response
                    )
                    for chunk in response_chunks:
                        await update.message.reply_text(
                            chunk,
                            parse_mode=(
                                "Markdown" if "```" in chunk or "*" in chunk else None
                            ),
                        )

                    # Log successful processing
                    self.telegram_logger.log_message(
                        f"Document processed successfully: {document.file_name}",
                        user_id,
                    )

                    # Update user statistics
                    try:
                        await self.user_data_manager.update_stats(
                            user_id, document=True
                        )
                    except Exception as stats_error:
                        self.logger.warning(
                            f"Failed to update stats: {str(stats_error)}"
                        )

                else:
                    # Handle error
                    await processing_message.edit_text(
                        f"❌ Sorry, I couldn't process your document: {result.error or 'Unknown error'}"
                    )

            except Exception as e:
                self.logger.error(f"Document processing error: {str(e)}")
                await processing_message.edit_text(
                    "❌ Sorry, there was an error processing your document. Please try again."
                )

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
            # Check if the message is in a group chat
            if update.effective_chat.type in ["group", "supergroup"]:
                # Process only if the bot is mentioned in the caption
                bot_username = "@" + context.bot.username
                caption = update.message.caption or ""
                if bot_username not in caption:
                    return
                else:
                    # Remove bot mention
                    caption = caption.replace(bot_username, "").strip()
            else:
                caption = update.message.caption or "Please analyze this document."

            # Get basic document information
            document = update.message.document
            file_name = document.file_name
            file_id = document.file_id
            file_extension = (
                os.path.splitext(file_name)[1][1:] if "." in file_name else ""
            )

            # Send typing action and status message
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            status_message = await update.message.reply_text(
                f"Processing your {file_extension.upper()} document... This might take a moment."
            )

            # Download and process the document
            document_file = await context.bot.get_file(file_id)
            file_content = await document_file.download_as_bytearray()
            document_file_obj = io.BytesIO(file_content)

            self.logger.info(
                f"Downloaded document: {file_name}, size: {len(file_content)} bytes, extension: {file_extension}"
            )

            # Default prompt if caption is empty
            prompt = (
                caption
                or f"Please analyze this {file_extension.upper()} file and provide a detailed summary."
            )

            # Use enhanced document processing for PDFs
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

            # Delete status message
            try:
                await status_message.delete()
            except Exception as delete_error:
                self.logger.warning(
                    f"Failed to delete document status message: {str(delete_error)}"
                )

            if response:
                # Format the response
                response_text = response.get(
                    "result", "Document processed successfully."
                )
                document_id = response.get("document_id", "Unknown")

                # Ensure the response is user-friendly
                formatted_response = (
                    f"**Document Analysis Completed**\n\n"
                    f"{response_text}\n\n"
                    f"**Document ID:** {document_id}"
                )

                # Send the formatted response to the user using safe_send_message
                await self.response_formatter.safe_send_message(
                    update.message,
                    formatted_response,
                    disable_web_page_preview=True,
                )

                # Store document interaction in memory manager
                try:
                    if (
                        hasattr(self.text_handler, "memory_manager")
                        and self.text_handler.memory_manager
                    ):
                        memory_manager = self.text_handler.memory_manager

                        # Add document interaction to memory with metadata
                        document_prompt = f"[Document submitted: {file_name}] {prompt}"
                        await memory_manager.add_user_message(
                            conversation_id,
                            document_prompt,
                            str(user_id),
                            document_type=file_extension,
                            document_name=file_name,
                            is_document=True,
                        )

                        # Add AI's response to memory
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
            # Replace the document handler with the more comprehensive handle_document method
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
            # Clear the flag
            context.user_data["awaiting_doc_text"] = False

            # Get the text
            content = update.message.text

            # Store for document generation
            context.user_data["doc_export_text"] = content

            # Offer format selection
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
                "Your text has been received. Please select the document format you want to export to:",
                reply_markup=format_markup,
            )

            return True  # Handled

        return False  # Not handled

    async def handle_awaiting_doc_image(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Process image when awaiting document content"""
        if context.user_data.get("awaiting_doc_image"):
            # Clear the flag
            context.user_data["awaiting_doc_image"] = False
            # Get the image
            image_file = update.message.photo[-1].get_file()
            # Store for document generation
            context.user_data["doc_export_image"] = image_file
            # Offer format selection
            format_options = [
                [
                    InlineKeyboardButton(
                        "📄 PDF Format", callback_data="export_format_pdf"
                    ),
                ]
            ]
