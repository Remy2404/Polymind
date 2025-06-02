import os, io
import re
import tempfile
import logging
import uuid
import speech_recognition as sr
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update, Message
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from pydub import AudioSegment
from handlers.text_handlers import TextHandler
from services.user_data_manager import UserDataManager
from telegram.ext import MessageHandler, filters
import aiohttp
import datetime
from services.gemini_api import GeminiAPI
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional, List, Union
from functools import partial
import traceback
import gc
import time
import weakref
from telegram import Update, Message, Document
from telegram.ext import MessageHandler, filters, ContextTypes, CallbackContext
from src.services.gemini_api import GeminiAPI
from src.services.user_data_manager import user_data_manager
from src.utils.telegramlog import TelegramLogger
from src.services.document_processing import DocumentProcessor
from src.handlers.text_handlers import TextHandler

# Import new utility classes
from handlers.message_context_handler import MessageContextHandler
from handlers.response_formatter import ResponseFormatter
from handlers.media_context_extractor import MediaContextExtractor
from services.media.image_processor import ImageProcessor
from services.media.voice_processor import VoiceProcessor
from services.model_handlers.prompt_formatter import PromptFormatter
from services.user_preferences_manager import UserPreferencesManager
from services.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)


class MessageHandlers:
    def __init__(
        self,
        gemini_api,
        user_data_manager,
        telegram_logger,
        document_processor,
        text_handler,
        deepseek_api=None,
        openrouter_api=None,
    ):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.document_processor = document_processor
        self.text_handler = text_handler
        self.logger = logging.getLogger(__name__)
        self.deepseek_api = deepseek_api
        self.openrouter_api = openrouter_api

        # Initialize utility classes
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.media_context_extractor = MediaContextExtractor()
        self.image_processor = ImageProcessor(gemini_api)
        self.voice_processor = VoiceProcessor()
        self.prompt_formatter = PromptFormatter()
        self.preferences_manager = UserPreferencesManager(user_data_manager)

        # Initialize conversation manager (will be lazy-loaded with proper dependencies)
        self._conversation_manager = None

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

            # Check if the bot is mentioned but don't send an automatic reply
            # Just log it for tracking purposes
            bot_username = "@Gemini_AIAssistBot"
            if bot_username in message_text:
                self.logger.info(f"Bot mentioned by user {user_id}")
                # Remove the automatic greeting that was causing duplicate responses
                # We'll let the text handler process the full message instead

            # Create text handler instance with all required API instances
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                openrouter_api=self.openrouter_api,  # Pass the openrouter_api for models like llama4_maverick
                deepseek_api=self.deepseek_api,  # Pass the deepseek_api for deepseek model
            )

            # Process the message
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
        """Handle incoming image messages."""
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

            # Get the highest resolution photo
            photo = update.message.photo[-1]

            # Get caption as prompt or use a default prompt
            caption = update.message.caption or "Describe this image in detail."

            # Show processing message
            processing_message = await update.message.reply_text(
                "Processing your image. Please wait..."
            )

            # Send typing indicator
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            try:
                # Download the image file
                photo_file = await context.bot.get_file(photo.file_id)
                image_bytes = await photo_file.download_as_bytearray()

                # Convert to BytesIO to make it easier to process
                image_data = io.BytesIO(image_bytes)

                # Use our ImageProcessor to analyze the image
                response = await self.image_processor.analyze_image(image_data, caption)

                # Delete the processing message
                try:
                    await processing_message.delete()
                except Exception as e:
                    self.logger.warning(
                        f"Failed to delete processing message: {str(e)}"
                    )

                # Send the analysis response
                if response:
                    await self._safe_reply(
                        update.message, response, parse_mode="Markdown"
                    )

                    # Save the image interaction to memory for future reference with enhanced metadata
                    try:
                        # Get conversation manager
                        conversation_manager = self._get_conversation_manager()
                        
                        # Create timestamp for reference
                        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Prepare image metadata for better memory recall
                        image_metadata = {
                            "timestamp": timestamp_str,
                            "file_id": photo.file_id,
                            "width": photo.width,
                            "height": photo.height,
                            "file_size": photo.file_size
                        }
                        
                        # Extract key image content for user data storage
                        image_data_entry = {
                            "timestamp": timestamp_str,
                            "caption": caption,
                            "description": response,
                            "file_id": photo.file_id
                        }
                        
                        # Initialize or update user's image history in user_data
                        if "image_history" not in context.user_data:
                            context.user_data["image_history"] = []
                        
                        # Add to image history with a reasonable limit
                        context.user_data["image_history"].append(image_data_entry)
                        if len(context.user_data["image_history"]) > 10:  # Keep last 10 images
                            context.user_data["image_history"] = context.user_data["image_history"][-10:]
                        
                        # Save image interaction with both caption and response and enhanced metadata
                        await conversation_manager.save_media_interaction(
                            user_id,
                            "image",
                            caption,
                            response,
                            **image_metadata
                        )
                        
                        # Also save to model-specific history if text_handler has model history manager
                        if hasattr(self.text_handler, 'model_history_manager') and self.text_handler.model_history_manager:
                            await self.text_handler.model_history_manager.save_image_interaction(
                                user_id,
                                caption,
                                response,
                                metadata=image_metadata
                            )
                        
                        self.logger.info(f"Enhanced image interaction saved to memory for user {user_id}")
                        
                    except Exception as memory_error:
                        self.logger.error(f"Error saving image interaction to memory: {str(memory_error)}")
                else:
                    await update.message.reply_text(
                        "Sorry, I couldn't analyze this image. Please try again with a different image."
                    )

                # Update user statistics
                try:
                    await self.user_data_manager.update_stats(user_id, image=True)
                except Exception as stats_error:
                    self.logger.warning(f"Failed to update stats: {str(stats_error)}")

            except Exception as inner_error:
                self.logger.error(f"Error processing image content: {str(inner_error)}")
                await processing_message.edit_text(
                    "Sorry, I couldn't process this image. Please try another one."
                )

        except Exception as e:
            self.logger.error(f"Error processing image message: {str(e)}")
            await self._error_handler(update, context)

        # Stats are already updated in the try block, no need to call it again here

    async def _handle_voice_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming voice messages, enhanced for Khmer and multilingual support."""
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
                else "Processing your voice message. Please wait..."
            )

            status_message = await update.message.reply_text(processing_text)

            # Use VoiceProcessor for downloading and converting voice file
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            ogg_file_path, wav_file_path = (
                await self.voice_processor.download_and_convert(
                    voice_file, str(user_id), is_khmer
                )
            )

            # Use VoiceProcessor for transcribing the voice file
            text, recognition_language = await self.voice_processor.transcribe(
                wav_file_path, lang, is_khmer
            )

            if not text:
                # Language-specific error message
                error_text = (
                    "សូមអភ័យទោស មិនអាចយល់សំឡេងបានទេ។ សូមសាកល្បងម្តងទៀតជាមួយសំឡេងច្បាស់ជាងនេះ។\n\n"
                    "Sorry, I couldn't understand the audio. Please try again with clearer audio."
                    if is_khmer
                    else "Sorry, I couldn't understand the audio. Please try again with clearer audio."
                )

                # Update status message instead of deleting
                try:
                    await status_message.edit_text(error_text)
                except Exception as edit_error:
                    self.logger.warning(
                        f"Could not edit status message: {str(edit_error)}"
                    )
                    try:
                        await update.message.reply_text(error_text)
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

            # Show the transcribed text to the user using ResponseFormatter
            transcript_text = (
                f"🎤 *បំលែងសំឡេងទៅជាអក្សរ (Transcription)*: \n{text}"
                if is_khmer
                else f"🎤 *Transcription*: \n{text}"
            )

            # Use safe_reply to handle the flood control issue
            try:
                transcript_message = await self._safe_reply(
                    update.message, transcript_text, parse_mode="Markdown"
                )
            except Exception as reply_error:
                self.logger.error(
                    f"Error sending transcript message: {str(reply_error)}"
                )
                # Try without markdown
                transcript_message = await update.message.reply_text(
                    f"🎤 Transcription: \n{text}"
                )

            # Log the transcribed text
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

    async def _safe_reply(
        self, message, text, parse_mode=None, retry_delay=5, max_retries=3
    ) -> None:
        """Safely reply to a message with built-in flood control handling"""
        for attempt in range(max_retries):
            try:
                return await message.reply_text(text, parse_mode=parse_mode)
            except Exception as e:
                error_str = str(e).lower()
                if "flood" in error_str and "retry" in error_str:
                    # Parse retry time from error message like "Flood control exceeded. Retry in 134 seconds"
                    retry_seconds = 5  # Default retry time
                    try:
                        retry_match = re.search(r"retry in (\d+)", error_str)
                        if retry_match:
                            retry_seconds = int(retry_match.group(1))
                            # Cap the retry time to avoid excessive waits
                            retry_seconds = min(retry_seconds, 10)
                    except:
                        pass

                    self.logger.warning(
                        f"Hit flood control, waiting {retry_seconds} seconds"
                    )
                    # Wait the required time plus a small buffer
                    await asyncio.sleep(retry_seconds + 1)
                    continue
                elif attempt < max_retries - 1:
                    # For other errors, retry with a fixed delay
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # For the last attempt, try without parse_mode
                    if parse_mode and attempt == max_retries - 1:
                        return await message.reply_text(text, parse_mode=None)
                    raise

        # Fallback - if all retries failed, try one last time without any formatting
        try:
            return await message.reply_text(
                text.replace("*", "").replace("_", "").replace("`", "")
            )
        except Exception as last_error:
            self.logger.error(
                f"Failed to send message after all retries: {str(last_error)}"
            )
            return None

    async def _handle_document_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle incoming document messages."""
        user_id = update.effective_user.id
        self.logger.info(f"Processing document for user: {user_id}")

        try:
            document = update.message.document
            file = await context.bot.get_file(document.file_id)
            file_extension = document.file_name.split(".")[-1]

            response = await self.document_processor.process_document_from_file(
                file=await file.download_as_bytearray(),
                file_extension=file_extension,
                prompt="Analyze this document.",
            )

            formatted_response = await self.text_handler.format_telegram_markdown(
                response
            )
            await update.message.reply_text(
                formatted_response,
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )

            self.user_data_manager.update_stats(user_id, document=True)
            self.telegram_logger.log_message(
                "Document processed successfully.", user_id
            )

        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
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

            # Default prompt if caption is empty
            prompt = (
                caption
                or f"Please analyze this {file_extension.upper()} file and provide a detailed summary."
            )

            # Use enhanced document processing for PDFs
            if file_extension.lower() == "pdf":
                response = await self.document_processor.process_document_enhanced(
                    file=document_file_obj, file_extension=file_extension, prompt=prompt
                )
            else:
                response = await self.document_processor.process_document_from_file(
                    file=document_file_obj, file_extension=file_extension, prompt=prompt
                )

            # Delete status message
            await status_message.delete()

            if response:
                # Format the response
                response_text = response.get(
                    "result", "Document processed successfully."
                )
                document_id = response.get("document_id", "Unknown")

                # Escape Markdown special characters
                response_text = self._escape_markdown(response_text)
                document_id = self._escape_markdown(document_id)

                # Ensure the response is user-friendly
                formatted_response = (
                    f"**Document Analysis Completed**\n\n"
                    f"{response_text}\n\n"
                    f"**Document ID:** {document_id}"
                )

                # Send the formatted response to the user
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=formatted_response,
                    parse_mode="Markdown",
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
            await update.message.reply_text(f"Error: {str(ve)}")
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            await update.message.reply_text(
                "Sorry, I couldn't process your document. Please ensure it's in a supported format."
            )

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram Markdown."""
        escape_chars = [
            "_",
            "*",
            "[",
            "]",
            "(",
            ")",
            "~",
            "`",
            ">",
            "#",
            "+",
            "-",
            "=",
            "|",
            "{",
            "}",
            ".",
            "!",
        ]
        for char in escape_chars:
            text = text.replace(char, f"\\{char}")
        return text

    async def _error_handler(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle errors occurring in the dispatcher."""
        self.logger.error(f"Update {update} caused error: {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred while processing your request. Please try again later."
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

    async def handle_document_processing(self, update, context):
        try:
            # Extract document and user information
            document = update.message.document
            user_id = update.message.from_user.id

            # Log the start of document processing
            logger.info(f"Processing document for user {user_id}")

            # Call the document processing service
            result = await self.document_processor.process_document_enhanced(
                file=document.file_id,
                file_extension=document.file_name.split(".")[-1],
                prompt="Analyze this document",
                user_id=user_id,
            )

            # Check if the result contains an error
            if "error" in result:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"Error processing document: {result['error']}",
                )
                return

            # Format the response for the user
            response_text = result.get("result", "Document processed successfully.")
            document_id = result.get("document_id", "Unknown")

            formatted_response = (
                f"**Document Analysis Completed**\n\n"
                f"{response_text}\n\n"
                f"**Document ID:** {document_id}"
            )

            # Send the formatted response to the user
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=formatted_response,
                parse_mode="Markdown",
            )

            # Log the successful completion
            logger.info(f"Document analysis completed successfully for user {user_id}")

        except Exception as e:
            # Log the error
            logger.error(f"Error in document processing: {str(e)}")

            # Notify the user of the error
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="An error occurred while processing your document. Please try again later.",
            )

    def _get_conversation_manager(self):
        """Lazy-load conversation manager with proper dependencies."""
        if self._conversation_manager is None:
            # Get memory manager and model history manager from text_handler
            if hasattr(self.text_handler, 'memory_manager') and hasattr(self.text_handler, 'model_history_manager'):
                self._conversation_manager = ConversationManager(
                    self.text_handler.memory_manager,
                    self.text_handler.model_history_manager
                )
            else:
                # Fallback - create basic conversation manager
                from services.memory_manager import MemoryManager
                from services.model_handlers.model_history_manager import ModelHistoryManager
                
                memory_manager = MemoryManager()
                model_history_manager = ModelHistoryManager(memory_manager, self.user_data_manager)
                self._conversation_manager = ConversationManager(memory_manager, model_history_manager)
        
        return self._conversation_manager
