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
from services.multimodal_processor import TelegramMultimodalProcessor
from src.handlers.text_handlers import TextHandler
from src.services.ai_command_router import AICommandRouter
from src.utils.docgen.document_processor import DocumentProcessor

# Import utility classes
from src.handlers.message_context_handler import MessageContextHandler
from src.handlers.response_formatter import ResponseFormatter
from src.services.media.voice_processor import (
    VoiceProcessor,
    SpeechEngine,
    create_voice_processor,
)
from src.services.memory_context.conversation_manager import ConversationManager
from src.services.group_chat.integration import GroupChatIntegration
from src.services.model_handlers.model_configs import ModelConfigurations, Provider

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
            # Initialize enhanced voice processor if not already done
            if not hasattr(self, "voice_processor") or self.voice_processor is None:
                try:
                    self.voice_processor = await create_voice_processor(
                        engine=SpeechEngine.AUTO  # Auto-select best engine
                    )

                    # Log available engines
                    info = self.voice_processor.get_engine_info()
                    available = [
                        name
                        for name, avail in info["available_engines"].items()
                        if avail
                    ]
                    self.logger.info(
                        f"Enhanced voice processor initialized with engines: {available}"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Failed to initialize enhanced voice processor: {e}"
                    )
                    # Fallback to basic voice processor
                    self.voice_processor = VoiceProcessor()

            # Initialize preferences manager if not available
            if (
                not hasattr(self, "preferences_manager")
                or self.preferences_manager is None
            ):
                from src.services.user_preferences_manager import UserPreferencesManager

                self.preferences_manager = UserPreferencesManager(
                    self.user_data_manager
                )

            # Get user's language preference using UserPreferencesManager
            user_lang = (
                await self.preferences_manager.get_user_language_preference(user_id)
                or "en"
            )

            # If not found in preferences, try the instance user_data_manager
            if user_lang == "en" and update.effective_user.language_code:
                user_lang = update.effective_user.language_code

            # Enhanced language mapping with better Khmer support
            language_map = {
                "en": "en-US",
                "km": "km-KH",
                "kh": "km-KH",  # Alternative code sometimes used
                "ru": "ru-RU",
                "fr": "fr-FR",
                "es": "es-ES",
                "de": "de-DE",
                "ja": "ja-JP",
                "zh": "zh-CN",
                "th": "th-TH",
                "vi": "vi-VN",
            }

            # Extract language prefix properly
            lang_prefix = user_lang.split("-")[0] if "-" in user_lang else user_lang
            lang = language_map.get(lang_prefix, "en-US")

            # Set flag for Khmer processing - FORCE Khmer processing if user has set language to km/kh
            is_khmer = lang_prefix in ["km", "kh"]

            # Log the detected language for debugging
            self.logger.info(
                f"Voice recognition language set to: {lang}, is_khmer={is_khmer}"
            )

            # Show processing message in the appropriate language
            processing_text = (
                "កំពុងដំណើរការសារសំឡេងរបស់អ្នក... សូមរង់ចាំ...\n(Processing your voice message. Please wait...)"
                if is_khmer
                else "🎤 Processing your voice message with enhanced AI recognition..."
            )

            status_message = await update.message.reply_text(processing_text)

            # Use enhanced VoiceProcessor for downloading and converting voice file
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            ogg_file_path, wav_file_path = (
                await self.voice_processor.download_and_convert(
                    voice_file, str(user_id), is_khmer
                )
            )

            # Use enhanced VoiceProcessor for transcribing the voice file with best engine
            if hasattr(self.voice_processor, "get_best_transcription"):
                # Use enhanced transcription with multiple engines
                text, recognition_language, metadata = (
                    await self.voice_processor.get_best_transcription(
                        wav_file_path, language=lang, confidence_threshold=0.6
                    )
                )

                # Get engine and confidence info
                engine_used = metadata.get("engine", "unknown")
                confidence = metadata.get("confidence", 0.0)

                self.logger.info(
                    f"Enhanced transcription: engine={engine_used}, confidence={confidence:.2f}"
                )

            else:
                # Fallback to basic transcription
                text, recognition_language = await self.voice_processor.transcribe(
                    wav_file_path, lang, is_khmer
                )
                metadata = {"engine": "basic", "confidence": 0.7}
                engine_used = "basic"
                confidence = 0.7

            if not text:
                # Language-specific error message with helpful tips
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

                # Update status message instead of deleting
                try:
                    await status_message.edit_text(error_text, parse_mode="Markdown")
                except Exception as edit_error:
                    self.logger.warning(
                        f"Could not edit status message: {str(edit_error)}"
                    )
                    try:
                        await update.message.reply_text(
                            error_text, parse_mode="Markdown"
                        )
                    except:
                        pass
                return

            # Delete the status message safely
            try:
                await status_message.delete()
            except Exception as msg_error:
                self.logger.warning(
                    f"Could not delete status message: {str(msg_error)}"
                )
                try:
                    await status_message.edit_text("✓ Processing complete")
                except:
                    pass

            # Show the transcribed text to the user using ResponseFormatter with engine info
            if confidence > 0.8:
                confidence_emoji = "🟢"  # High confidence
            elif confidence > 0.6:
                confidence_emoji = "🟡"  # Medium confidence
            else:
                confidence_emoji = "🔴"  # Low confidence

            transcript_text = (
                f"🎤 *បំលែងសំឡេងទៅជាអក្សរ (Transcription)*: \n{text}"
                if is_khmer
                else f"🎤 **Voice Message Transcribed** {confidence_emoji}\n\n{text}"
            )

            # Add engine info for debugging (optional, can be disabled in production)
            if confidence > 0 and not is_khmer:
                transcript_text += (
                    f"\n\n_Engine: {engine_used.title()}, Confidence: {confidence:.1%}_"
                )

            # Use safe_send_message for better error handling
            try:
                transcript_message = await self.response_formatter.safe_send_message(
                    update.message, transcript_text
                )
            except Exception as reply_error:
                self.logger.error(
                    f"Error sending transcript message: {str(reply_error)}"
                )
                # Continue with processing since this isn't critical
                transcript_message = await update.message.reply_text(
                    f"🎤 Transcription: \n{text}"
                )

            # Log the transcribed text with engine info
            self.telegram_logger.log_message(
                f"Transcribed {recognition_language} text: {text}", user_id
            )

            # Initialize user data if not already initialized
            await self.user_data_manager.initialize_user(user_id)

            # Create text handler instance
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                self.openrouter_api if hasattr(self, "openrouter_api") else None,
                self.deepseek_api if hasattr(self, "deepseek_api") else None,
            )

            # Get or create ConversationManager
            if (
                not hasattr(self, "_conversation_manager")
                or not self._conversation_manager
            ):
                self._conversation_manager = ConversationManager(
                    text_handler.memory_manager, text_handler.model_history_manager
                )

            # Make sure model history manager knows about user's model preference
            if hasattr(text_handler, "model_history_manager"):
                # Get the user's selected model early
                user_settings = await self.user_data_manager.get_user_settings(
                    str(user_id)
                )
                preferred_model = await self.user_data_manager.get_user_preference(
                    user_id, "preferred_model", None
                )
                active_model = (
                    preferred_model
                    if preferred_model
                    else user_settings.get("active_model", "gemini")
                )

                # Set the selected model in model history manager
                if hasattr(text_handler.model_history_manager, "select_model_for_user"):
                    text_handler.model_history_manager.select_model_for_user(
                        user_id, active_model
                    )

            # Use ConversationManager to save voice interaction to memory
            language_label = "km" if is_khmer else lang.split("-")[0]
            await self._conversation_manager.save_media_interaction(
                user_id,
                "voice",
                text,
                f"I've transcribed your voice message which said: {text}",
            )

            # NEW CODE: Process the transcribed text with the AI model
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            # Treat the transcribed text as a regular text message for AI processing
            # Pass the quoted text context if it exists
            prompt = text
            if quoted_text:
                prompt = self.context_handler.format_prompt_with_quote(
                    text, quoted_text
                )

            # Get the user's selected model
            user_settings = await self.user_data_manager.get_user_settings(str(user_id))

            # First check if there's a preferred_model in the user preferences
            preferred_model = await self.user_data_manager.get_user_preference(
                user_id, "preferred_model", None
            )

            # Log detailed model selection information
            self.logger.info(f"Voice message model selection - User: {user_id}")
            self.logger.info(f"  - preferred_model from preferences: {preferred_model}")
            self.logger.info(
                f"  - active_model from user settings: {user_settings.get('active_model', 'not set')}"
            )

            # If preferred_model is set, use that, otherwise fall back to the active_model setting or default to gemini
            active_model = (
                preferred_model
                if preferred_model
                else user_settings.get("active_model", "gemini")
            )

            # Log the model being used
            self.logger.info(
                f"SELECTED MODEL: '{active_model}' for voice response. Prompt: {prompt[:50]}..."
            )

            # Get the appropriate model indicator
            model_indicator = "🔮 Gemini"  # Default
            if active_model == "deepseek":
                model_indicator = "🧠 DeepSeek"
            elif active_model == "deepcoder":
                model_indicator = "💻 DeepCoder"
            elif active_model == "llama4_maverick":
                model_indicator = "🦙 Llama-4"
            elif active_model == "llama-3.3-8b":
                model_indicator = "🦙 Llama-3.3"
            elif "llama" in active_model.lower():
                model_indicator = "🦙 Llama"
            elif "deepseek" in active_model.lower():
                model_indicator = "🧠 DeepSeek"

            # Log the model indicator for debugging
            self.logger.info(
                f"Using model indicator: {model_indicator} for model: {active_model}"
            )

            # Generate AI response based on active model
            ai_response = ""
            try:
                self.logger.info(f"Generating response with model: {active_model}")

                if active_model == "gemini":
                    ai_response = await self.gemini_api.generate_response(prompt)
                    self.logger.info("Used Gemini API for response")

                elif active_model == "deepseek" and hasattr(self, "deepseek_api"):
                    ai_response = await self.deepseek_api.generate_response(prompt)
                    self.logger.info("Used DeepSeek API for response")

                elif active_model == "llama4_maverick" and hasattr(
                    self, "openrouter_api"
                ):
                    self.logger.info(
                        "Attempting to use OpenRouter API with llama4_maverick model"
                    )
                    ai_response = await self.openrouter_api.generate_response(
                        prompt, active_model
                    )
                    self.logger.info(
                        "Successfully used OpenRouter API with llama4_maverick"
                    )

                elif active_model == "deepcoder" and hasattr(self, "openrouter_api"):
                    ai_response = await self.openrouter_api.generate_response(
                        prompt, active_model
                    )
                    self.logger.info("Used OpenRouter API with deepcoder model")

                elif active_model == "llama-3.3-8b" and hasattr(self, "openrouter_api"):
                    self.logger.info(
                        "Attempting to use OpenRouter API with llama-3.3-8b model"
                    )
                    ai_response = await self.openrouter_api.generate_response(
                        prompt, active_model
                    )
                    self.logger.info(
                        "Successfully used OpenRouter API with llama-3.3-8b"
                    )

                elif hasattr(self, "openrouter_api"):
                    self.logger.info(
                        f"Using fallback to OpenRouter API with model: {active_model}"
                    )
                    ai_response = await self.openrouter_api.generate_response(
                        prompt, active_model
                    )

                else:
                    self.logger.warning(
                        f"No suitable API found for model: {active_model}"
                    )
                    ai_response = "Sorry, I couldn't process your voice message with the selected model. Please try with a different model."
            except Exception as e:
                self.logger.error(
                    f"Error generating response with {active_model}: {str(e)}",
                    exc_info=True,
                )
                ai_response = f"Sorry, there was an error generating a response with the {active_model} model. Technical details: {str(e)}"

            if not ai_response:
                self.logger.warning(
                    f"Empty AI response for user {user_id} with active model {active_model}"
                )
                ai_response = "I'm sorry, I couldn't generate a response at this time. Please try again later."

            # Format the response with model indicator
            formatted_response = self.response_formatter.format_with_model_indicator(
                ai_response, model_indicator, quoted_text is not None
            )

            # Log successful response generation
            self.logger.info(
                f"Generated AI response of length {len(ai_response)} for voice message"
            )

            # Split long messages if needed
            response_chunks = await self.response_formatter.split_long_message(
                formatted_response
            )

            # Send the response chunks
            for chunk in response_chunks:
                await update.message.reply_text(chunk)

            # Save the conversation pair
            await self._conversation_manager.save_message_pair(
                user_id, prompt, ai_response, active_model
            )

        except Exception as e:
            self.logger.error(
                f"Error processing voice message: {str(e)}", exc_info=True
            )
            error_message = "Sorry, there was an error processing your voice message. Please try again later."

            # Check if we're at the transcription step but have no text
            if "text" in locals() and not text:
                error_message = "Sorry, I couldn't transcribe your voice message. Please try speaking more clearly or in a quieter environment."
            # Check if we're at the AI response step
            elif "prompt" in locals() and "ai_response" not in locals():
                error_message = "I understood your voice message, but I'm having trouble generating a response right now. Please try again later."

            try:
                if "status_message" in locals() and status_message:
                    try:
                        await status_message.edit_text(error_message)
                    except Exception as edit_error:
                        self.logger.warning(
                            f"Could not edit status message: {str(edit_error)}"
                        )
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
