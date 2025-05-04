import io
import os
import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from utils.telegramlog import telegram_logger
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from typing import List, Dict, Any
import asyncio

# Import core handler utilities
from .message_context_handler import MessageContextHandler
from .response_formatter import ResponseFormatter
from .media_context_extractor import MediaContextExtractor

# Import model and service utilities
from services.memory_manager import MemoryManager
from services.model_handlers.model_history_manager import ModelHistoryManager
from services.model_handlers.factory import ModelHandlerFactory
from services.media.image_processor import ImageProcessor
from services.model_handlers.prompt_formatter import PromptFormatter
from services.conversation_manager import ConversationManager

# Import our newly modularized code
from .text_processing.intent_detector import IntentDetector
from .text_processing.media_analyzer import MediaAnalyzer
from .text_processing.utilities import MediaUtilities, MessagePreprocessor


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
        self.max_context_length = 9

        # Initialize memory and history management
        self.memory_manager = MemoryManager(
            db=user_data_manager.db if hasattr(user_data_manager, "db") else None,
        )
        self.memory_manager.short_term_limit = 15
        self.memory_manager.token_limit = 8192
        self.model_history_manager = ModelHistoryManager(self.memory_manager)

        # Initialize utility classes
        self.model_registry = None
        self.user_model_manager = None
        self.model_registry_initialized = False

        # Initialize context and formatting utilities
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.media_context_extractor = MediaContextExtractor()
        self.prompt_formatter = PromptFormatter()

        # Initialize our image processing capability
        self.image_processor = ImageProcessor(gemini_api)

        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            self.memory_manager, self.model_history_manager
        )

        # Initialize our new modular components
        self.intent_detector = IntentDetector()
        self.media_analyzer = MediaAnalyzer(gemini_api)
        self.media_utilities = MediaUtilities()
        self.message_preprocessor = MessagePreprocessor()

    async def format_telegram_markdown(self, text: str) -> str:
        # Delegate to ResponseFormatter
        return await self.response_formatter.format_telegram_markdown(text)

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        # Delegate to ResponseFormatter
        return await self.response_formatter.split_long_message(text, max_length)

    async def handle_text_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Main handler for text messages.
        Processes user messages, detects intent, and generates appropriate responses.
        """
        if not update.message and not update.edited_message:
            return

        user_id = update.effective_user.id
        message = update.message or update.edited_message
        message_text = message.text

        # Extract quoted message context if this is a reply
        quoted_text, quoted_message_id = self.context_handler.extract_reply_context(
            message
        )

        # Define conversation ID for this user
        conversation_id = f"user_{user_id}"

        try:
            # Handle edited messages
            if update.edited_message and "bot_messages" in context.user_data:
                await self._handle_edited_message(update, context)

            # In group chats, only process messages that mention the bot
            if update.effective_chat.type in ["group", "supergroup"]:
                bot_username = "@" + context.bot.username
                if bot_username not in message_text:
                    # Bot not mentioned, ignore message
                    return
                else:
                    # Remove bot username from message text
                    message_text = message_text.replace(bot_username, "").strip()

            # Extract any attached media files
            has_attached_media, media_files, media_type = (
                await self._extract_media_files(update, context)
            )

            # Send initial "thinking" message and appropriate chat action
            thinking_message = await message.reply_text("Processing your request...🧠")
            await self._send_appropriate_chat_action(
                update, context, has_attached_media, media_type
            )

            # Detect user intent (analyze, generate image, or chat)
            user_intent = await self.intent_detector.detect_user_intent(
                message_text, has_attached_media
            )

            # Get user's preferred model
            preferred_model = await self._get_user_preferred_model(user_id)

            # Get conversation history for context
            history_context = await self.conversation_manager.get_conversation_history(
                user_id, max_messages=self.max_context_length, model=preferred_model
            )

            # Handle the request based on detected intent
            if user_intent == "generate_image":
                await self._handle_image_generation(
                    update,
                    context,
                    thinking_message,
                    message_text,
                    user_id,
                    preferred_model,
                )
                return

            elif has_attached_media and user_intent == "analyze":
                await self._handle_media_analysis(
                    update,
                    context,
                    thinking_message,
                    media_files,
                    media_type,
                    message_text,
                    user_id,
                    preferred_model,
                )
                return

            # If we're here, this is a regular text conversation
            await self._handle_text_conversation(
                update,
                context,
                thinking_message,
                message_text,
                quoted_text,
                quoted_message_id,
                history_context,
                user_id,
                preferred_model,
            )

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            if "thinking_message" in locals() and thinking_message is not None:
                await thinking_message.delete()
            await update.message.reply_text(
                "Sorry, I encountered an error\\. Please try again later\\.",
                parse_mode="MarkdownV2",
            )

    async def _handle_edited_message(self, update, context):
        """Handle when a user edits their previous message"""
        original_message_id = update.edited_message.message_id
        if original_message_id in context.user_data["bot_messages"]:
            for msg_id in context.user_data["bot_messages"][original_message_id]:
                if msg_id:
                    try:
                        await context.bot.delete_message(
                            chat_id=update.effective_chat.id, message_id=msg_id
                        )
                    except Exception as e:
                        self.logger.error(f"Error deleting old message: {str(e)}")
            del context.user_data["bot_messages"][original_message_id]

    async def _extract_media_files(self, update, context):
        """Extract media files from the update"""
        has_attached_media = False
        media_files = []
        media_type = None

        if update.message:
            if update.message.photo:
                has_attached_media = True
                media_type = "photo"
                photo = update.message.photo[-1]
                photo_file = await context.bot.get_file(photo.file_id)
                photo_bytes = await photo_file.download_as_bytearray()
                media_files.append(
                    {
                        "type": "photo",
                        "data": io.BytesIO(photo_bytes),
                        "mime": "image/jpeg",
                        "filename": f"photo_{photo.file_id}.jpg",
                    }
                )

            elif update.message.video:
                has_attached_media = True
                media_type = "video"
                video = update.message.video
                video_file = await context.bot.get_file(video.file_id)
                video_bytes = await video_file.download_as_bytearray()
                media_files.append(
                    {
                        "type": "video",
                        "data": io.BytesIO(video_bytes),
                        "mime": "video/mp4",
                        "filename": (
                            video.file_name
                            if hasattr(video, "file_name")
                            else f"video_{video.file_id}.mp4"
                        ),
                    }
                )

            elif update.message.voice or update.message.audio:
                has_attached_media = True
                media_type = "audio"
                audio = update.message.voice or update.message.audio
                audio_file = await context.bot.get_file(audio.file_id)
                audio_bytes = await audio_file.download_as_bytearray()
                file_name = (
                    getattr(audio, "file_name", None) or f"audio_{audio.file_id}.ogg"
                )
                media_files.append(
                    {
                        "type": "audio",
                        "data": io.BytesIO(audio_bytes),
                        "mime": "audio/ogg",
                        "filename": file_name,
                    }
                )

            elif update.message.document:
                has_attached_media = True
                media_type = "document"
                document = update.message.document
                document_file = await context.bot.get_file(document.file_id)
                document_bytes = await document_file.download_as_bytearray()
                file_ext = os.path.splitext(document.file_name)[1].lower()
                mime_type = MediaUtilities.get_mime_type(file_ext)
                media_files.append(
                    {
                        "type": "document",
                        "data": io.BytesIO(document_bytes),
                        "mime": mime_type,
                        "filename": document.file_name,
                    }
                )

            # Check for media group (multiple files sent together)
            elif update.message.media_group_id:
                has_attached_media = True
                media_type = "media_group"

                # Store the media group ID in context for tracking
                if "media_groups" not in context.bot_data:
                    context.bot_data["media_groups"] = {}

                media_group_id = update.message.media_group_id

                # Check if we've already processed this media group partially
                if media_group_id in context.bot_data["media_groups"]:
                    # Add this media item to the existing group
                    if update.message.photo:
                        photo = update.message.photo[-1]
                        photo_file = await context.bot.get_file(photo.file_id)
                        photo_bytes = await photo_file.download_as_bytearray()
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": "photo",
                                "data": io.BytesIO(photo_bytes),
                                "mime": "image/jpeg",
                                "filename": f"photo_{photo.file_id}.jpg",
                            }
                        )

                    elif update.message.document:
                        document = update.message.document
                        document_file = await context.bot.get_file(document.file_id)
                        document_bytes = await document_file.download_as_bytearray()
                        file_ext = os.path.splitext(document.file_name)[1].lower()
                        mime_type = MediaUtilities.get_mime_type(file_ext)
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": "document",
                                "data": io.BytesIO(document_bytes),
                                "mime": mime_type,
                                "filename": document.file_name,
                            }
                        )

                    # Do not process each media file individually - return empty for now
                    # The complete group will be processed once the caption message is received
                    # or after a short delay
                    return False, [], None
                else:
                    # Start tracking this new media group
                    context.bot_data["media_groups"][media_group_id] = []

                    # Check the current message's type
                    if update.message.photo:
                        photo = update.message.photo[-1]
                        photo_file = await context.bot.get_file(photo.file_id)
                        photo_bytes = await photo_file.download_as_bytearray()
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": "photo",
                                "data": io.BytesIO(photo_bytes),
                                "mime": "image/jpeg",
                                "filename": f"photo_{photo.file_id}.jpg",
                            }
                        )

                    elif update.message.document:
                        document = update.message.document
                        document_file = await context.bot.get_file(document.file_id)
                        document_bytes = await document_file.download_as_bytearray()
                        file_ext = os.path.splitext(document.file_name)[1].lower()
                        mime_type = MediaUtilities.get_mime_type(file_ext)
                        context.bot_data["media_groups"][media_group_id].append(
                            {
                                "type": "document",
                                "data": io.BytesIO(document_bytes),
                                "mime": mime_type,
                                "filename": document.file_name,
                            }
                        )

                    # Schedule a task to process the complete media group after a delay
                    # This gives time for all media files to be received
                    asyncio.create_task(
                        self._process_complete_media_group(
                            media_group_id,
                            update.effective_chat.id,
                            update.effective_user.id,
                            update.message.caption or "",
                            context,
                        )
                    )

                    # Return empty for now - processing will happen in the scheduled task
                    return False, [], None

        return has_attached_media, media_files, media_type

    async def _process_complete_media_group(
        self, media_group_id, chat_id, user_id, caption, context
    ):
        """
        Process a complete media group after a delay to ensure all files are received
        """
        # Wait a moment for all media files to be received (Telegram typically sends them in quick succession)
        await asyncio.sleep(1.5)

        if (
            "media_groups" in context.bot_data
            and media_group_id in context.bot_data["media_groups"]
        ):
            media_files = context.bot_data["media_groups"][media_group_id]

            # Clean up the stored media group to avoid memory leaks
            del context.bot_data["media_groups"][media_group_id]

            # Only process if we have files
            if media_files:
                # Send a thinking message
                thinking_message = await context.bot.send_message(
                    chat_id=chat_id, text="Processing multiple files... 🧠"
                )

                try:
                    # Get user's preferred model
                    from services.user_preferences_manager import UserPreferencesManager

                    preferences_manager = UserPreferencesManager(self.user_data_manager)
                    preferred_model = (
                        await preferences_manager.get_user_model_preference(user_id)
                    )

                    # Process multiple files using our new MultiFileProcessor
                    from services.media.multi_file_processor import MultiFileProcessor

                    multi_processor = MultiFileProcessor(self.gemini_api)

                    result = await multi_processor.process_multiple_files(
                        media_files, caption or "Analyze these files"
                    )

                    # Delete thinking message
                    try:
                        await thinking_message.delete()
                    except Exception:
                        pass

                    # Format and send the response
                    model_indicator = (
                        "🧠 Gemini"
                        if preferred_model == "gemini"
                        else f"🤖 {preferred_model.capitalize()}"
                    )

                    if "intent" in result:
                        intent_message = f"I'm processing these files with intent: {result['intent']}\n\n"
                    else:
                        intent_message = ""

                    if "results" in result:
                        # Format each file result
                        formatted_results = []
                        for filename, content in result["results"].items():
                            if isinstance(content, str):
                                header = f"📄 *{filename}*:"
                                formatted_results.append(f"{header}\n{content}")

                        # Combine results
                        if formatted_results:
                            response = (
                                f"{intent_message}{model_indicator}\n\n"
                                + "\n\n".join(formatted_results)
                            )

                            # Split and send
                            chunks = await self.response_formatter.split_long_message(
                                response
                            )
                            for chunk in chunks:
                                formatted_chunk = await self.response_formatter.format_telegram_markdown(
                                    chunk
                                )
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    text=formatted_chunk,
                                    parse_mode="MarkdownV2",
                                    disable_web_page_preview=True,
                                )
                        else:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text="Sorry, I couldn't process these files properly. Please try again.",
                            )
                    else:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text="Sorry, I couldn't process these files. Please try again with a clearer prompt.",
                        )

                except Exception as e:
                    self.logger.error(f"Error processing media group: {e}")
                    try:
                        await thinking_message.delete()
                    except Exception:
                        pass
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="Sorry, there was an error processing your files. Please try again later.",
                    )

    async def _handle_media_analysis(
        self,
        update,
        context,
        thinking_message,
        media_files,
        media_type,
        message_text,
        user_id,
        preferred_model,
    ):
        """Handle analysis of media files"""
        # Check if we have multiple files to process
        if len(media_files) > 1:
            # Use MultiFileProcessor for multiple files
            from services.media.multi_file_processor import MultiFileProcessor

            multi_processor = MultiFileProcessor(self.gemini_api)
            result = await multi_processor.process_multiple_files(
                media_files, message_text or "Analyze these files"
            )

            # Delete thinking message
            if thinking_message is not None:
                try:
                    await thinking_message.delete()
                    thinking_message = None
                except Exception:
                    pass

            # Format and send the response
            model_indicator = (
                "🧠 Gemini"
                if preferred_model == "gemini"
                else f"🤖 {preferred_model.capitalize()}"
            )

            if "intent" in result:
                intent_message = (
                    f"I'm processing these files with intent: {result['intent']}\n\n"
                )
            else:
                intent_message = ""

            if "results" in result:
                # Format each file result
                formatted_results = []
                for filename, content in result["results"].items():
                    if isinstance(content, str):
                        header = f"📄 *{filename}*:"
                        formatted_results.append(f"{header}\n{content}")

                # Combine results
                if formatted_results:
                    response = f"{intent_message}{model_indicator}\n\n" + "\n\n".join(
                        formatted_results
                    )

                    # Split and send
                    chunks = await self.response_formatter.split_long_message(response)
                    for chunk in chunks:
                        formatted_chunk = (
                            await self.response_formatter.format_telegram_markdown(
                                chunk
                            )
                        )
                        await update.message.reply_text(
                            formatted_chunk,
                            parse_mode="MarkdownV2",
                            disable_web_page_preview=True,
                        )

                    # Update user stats
                    if self.user_data_manager:
                        await self.user_data_manager.update_stats(
                            user_id, multi_file_analysis=True
                        )

                    # Save to conversation history
                    media_description = f"[Multiple files analysis request]"
                    await self.conversation_manager.save_media_interaction(
                        user_id,
                        "multi_files",
                        media_description,
                        response,
                        preferred_model,
                    )

                    return

            # If we get here, something went wrong with the processing
            await update.message.reply_text(
                "Sorry, I couldn't analyze the content you provided. Please try again with a clearer prompt."
            )
            return

        # Single file handling (existing code)
        result = await self.media_analyzer.analyze_media(
            media_files, message_text, preferred_model
        )

        # Delete thinking message
        if thinking_message is not None:
            try:
                await thinking_message.delete()
                thinking_message = None
            except Exception:
                pass

        # Format and send the response
        if result:
            # Add model indicator
            model_indicator = (
                "🧠 Gemini"
                if preferred_model == "gemini"
                else f"🤖 {preferred_model.capitalize()}"
            )
            text_to_send = self.response_formatter.format_with_model_indicator(
                result, model_indicator
            )

            # Format for Telegram
            formatted_response = await self.response_formatter.format_telegram_markdown(
                text_to_send
            )

            # Send the response
            await update.message.reply_text(
                formatted_response,
                parse_mode="MarkdownV2",
                disable_web_page_preview=True,
            )

            # Save to conversation history
            media_description = f"[{media_type.capitalize()} analysis request]"
            await self.conversation_manager.save_media_interaction(
                user_id, media_type, media_description, result, preferred_model
            )

            # Update user stats
            if self.user_data_manager:
                await self.user_data_manager.update_stats(user_id, **{media_type: True})
        else:
            await update.message.reply_text(
                "Sorry, I couldn't analyze the content you provided. Please try again."
            )

    async def _send_appropriate_chat_action(
        self, update, context, has_attached_media, media_type
    ):
        """Send appropriate chat action based on message type"""
        action = ChatAction.TYPING

        if has_attached_media:
            if media_type == "photo":
                action = ChatAction.UPLOAD_PHOTO
            elif media_type == "video":
                action = ChatAction.UPLOAD_VIDEO
            elif media_type == "audio":
                action = ChatAction.RECORD_AUDIO

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=action
        )

    async def _get_user_preferred_model(self, user_id):
        """Get user's preferred model"""
        from services.user_preferences_manager import UserPreferencesManager

        preferences_manager = UserPreferencesManager(self.user_data_manager)
        preferred_model = await preferences_manager.get_user_model_preference(user_id)
        self.logger.info(f"Preferred model for user {user_id}: {preferred_model}")
        return preferred_model

    async def _handle_image_generation(
        self, update, context, thinking_message, message_text, user_id, preferred_model
    ):
        """Handle image generation requests"""
        # Use our enhanced prompt extraction instead of keyword replacement
        image_prompt = self.intent_detector.extract_image_prompt(message_text)

        if not image_prompt:
            # Fallback to simpler extraction if needed
            image_prompt = (
                message_text.replace("generate image", "")
                .replace("create image", "")
                .strip()
            )

        # Try to delete the thinking message
        if thinking_message is not None:
            try:
                await thinking_message.delete()
                thinking_message = None
            except Exception:
                pass

        # Inform the user that image generation is starting
        status_message = await update.message.reply_text(
            "Generating image based on your description... This may take a moment. 🎨"
        )

        try:
            # Generate the image
            image_bytes = await self.gemini_api.generate_image(image_prompt)

            if image_bytes and len(image_bytes) > 0:
                # Try to delete the status message
                if status_message is not None:
                    try:
                        await status_message.delete()
                        status_message = None
                    except Exception:
                        try:
                            await status_message.edit_text(
                                "✅ Image generated successfully!"
                            )
                            status_message = None
                        except Exception:
                            pass

                # Send the generated image
                caption = f"Generated image of: {image_prompt}"
                await update.message.reply_photo(
                    photo=io.BytesIO(image_bytes), caption=caption
                )

                # Update user stats
                if self.user_data_manager:
                    await self.user_data_manager.update_stats(
                        user_id, image_generation=True
                    )

                # Save to conversation history
                await self.conversation_manager.save_media_interaction(
                    user_id,
                    "generated_image",
                    f"Generate an image of: {image_prompt}",
                    f"Here's the image I generated of {image_prompt}.",
                    preferred_model,
                )
            else:
                # Image generation failed
                if status_message is not None:
                    try:
                        await status_message.edit_text(
                            "Sorry, I couldn't generate that image. Please try with a different description."
                        )
                    except Exception:
                        pass
        except Exception as e:
            self.logger.error(f"Error generating image: {e}")
            if status_message is not None:
                try:
                    await status_message.edit_text(
                        "Sorry, there was an error generating your image. Please try again later."
                    )
                except Exception:
                    pass

    async def _handle_text_conversation(
        self,
        update,
        context,
        thinking_message,
        message_text,
        quoted_text,
        quoted_message_id,
        history_context,
        user_id,
        preferred_model,
    ):
        """Handle regular text conversation"""
        message = update.message or update.edited_message

        # Build enhanced prompt with context
        enhanced_prompt = message_text

        # Add quoted text context if this is a reply
        if quoted_text:
            enhanced_prompt = self.prompt_formatter.add_context(
                message_text, "quote", quoted_text
            )

        # Add any reference to previously shared media
        if (
            self.context_handler.detect_reference_to_image(message_text)
            and "image_history" in context.user_data
        ):
            image_context = await self.media_context_extractor.get_image_context(
                context.user_data
            )
            enhanced_prompt = self.prompt_formatter.add_context(
                enhanced_prompt, "image", image_context
            )

        if (
            self.context_handler.detect_reference_to_document(message_text)
            and "document_history" in context.user_data
        ):
            document_context = await self.media_context_extractor.get_document_context(
                context.user_data
            )
            enhanced_prompt = self.prompt_formatter.add_context(
                enhanced_prompt, "document", document_context
            )

        # Apply response style guidelines
        enhanced_prompt_with_guidelines = (
            await self.prompt_formatter.apply_response_guidelines(
                enhanced_prompt,
                ModelHandlerFactory.get_model_handler(
                    preferred_model,
                    gemini_api=self.gemini_api,
                    openrouter_api=self.openrouter_api,
                    deepseek_api=self.deepseek_api,
                ),
                context,
            )
        )

        # Get model timeout
        model_timeout = 60.0
        if self.user_model_manager:
            model_config = self.user_model_manager.get_user_model_config(user_id)
            model_timeout = model_config.timeout_seconds if model_config else 60.0

        try:
            # Get the model handler
            model_handler = ModelHandlerFactory.get_model_handler(
                preferred_model,
                gemini_api=self.gemini_api,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )

            # Generate response
            response = await asyncio.wait_for(
                model_handler.generate_response(
                    prompt=enhanced_prompt_with_guidelines,
                    context=history_context,
                    temperature=0.7,
                    max_tokens=4000,
                    quoted_message=quoted_text,
                ),
                timeout=model_timeout,
            )

            # Delete thinking message
            if thinking_message is not None:
                try:
                    await thinking_message.delete()
                    thinking_message = None
                except Exception:
                    pass

            if response is None:
                await message.reply_text(
                    "Sorry, I couldn't generate a response\\. Please try rephrasing your message\\.",
                    parse_mode="MarkdownV2",
                )
                return

            # Send the response
            await self._send_formatted_response(
                update,
                context,
                message,
                response,
                model_handler.get_model_indicator(),
                quoted_text,
                quoted_message_id,
            )

            # Save to conversation history
            if response:
                if quoted_text:
                    await self.conversation_manager.add_quoted_message_context(
                        user_id, quoted_text, message_text, response, preferred_model
                    )
                else:
                    await self.conversation_manager.save_message_pair(
                        user_id, message_text, response, preferred_model
                    )

            telegram_logger.log_message("Text response sent successfully", user_id)

        except asyncio.TimeoutError:
            if thinking_message is not None:
                await thinking_message.delete()
            await message.reply_text(
                "Sorry, the request took too long to process. Please try again later.",
                parse_mode="MarkdownV2",
            )
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            if thinking_message is not None:
                await thinking_message.delete()
            await message.reply_text(
                "Sorry, there was an error processing your request. Please try again later.",
                parse_mode="MarkdownV2",
            )

    async def _send_formatted_response(
        self,
        update,
        context,
        message,
        response,
        model_indicator,
        quoted_text,
        quoted_message_id,
    ):
        """Format and send the AI response"""
        # Split long messages
        message_chunks = await self.response_formatter.split_long_message(response)
        sent_messages = []

        # Store indicator for editing functionality
        context.user_data["last_message_indicator"] = model_indicator

        # Determine if this is a reply
        is_reply = self.context_handler.should_use_reply_format(
            quoted_text, quoted_message_id
        )

        # Send each chunk
        for i, chunk in enumerate(message_chunks):
            try:
                # Format first chunk with model indicator
                if i == 0:
                    text_to_send = self.response_formatter.format_with_model_indicator(
                        chunk, model_indicator, is_reply
                    )
                else:
                    text_to_send = chunk

                # Format with markdown
                formatted_chunk = (
                    await self.response_formatter.format_telegram_markdown(text_to_send)
                )

                # Send the message with appropriate reply if needed
                if i == 0 and is_reply:
                    last_message = await message.reply_text(
                        formatted_chunk,
                        parse_mode="MarkdownV2",
                        disable_web_page_preview=True,
                        reply_to_message_id=quoted_message_id,
                    )
                elif i == 0:
                    last_message = await message.reply_text(
                        formatted_chunk,
                        parse_mode="MarkdownV2",
                        disable_web_page_preview=True,
                    )
                else:
                    last_message = await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=formatted_chunk,
                        parse_mode="MarkdownV2",
                        disable_web_page_preview=True,
                    )
                sent_messages.append(last_message)
            except Exception as formatting_error:
                # Log the error with the problematic text
                self.logger.error(
                    f"Formatting error: {str(formatting_error)}\nProblematic text: {text_to_send[:100]}..."
                )

                try:
                    # Try with simpler formatting - strip all markdown formatting characters
                    simplified_text = text_to_send
                    for char in [
                        "*",
                        "_",
                        "`",
                        "[",
                        "]",
                        "(",
                        ")",
                        "~",
                        ">",
                        "#",
                        "+",
                        "=",
                        "|",
                        "{",
                        "}",
                        "!",
                    ]:
                        simplified_text = simplified_text.replace(char, "")

                    if i == 0 and is_reply:
                        last_message = await message.reply_text(
                            simplified_text,
                            parse_mode=None,
                            disable_web_page_preview=True,
                            reply_to_message_id=quoted_message_id,
                        )
                    elif i == 0:
                        last_message = await message.reply_text(
                            simplified_text,
                            parse_mode=None,
                            disable_web_page_preview=True,
                        )
                    else:
                        last_message = await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=simplified_text,
                            parse_mode=None,
                            disable_web_page_preview=True,
                        )
                    sent_messages.append(last_message)
                except Exception as final_error:
                    # Last resort fallback - send a generic message
                    self.logger.error(f"Final fallback failed: {str(final_error)}")
                    fallback_text = "Sorry, I had trouble formatting the response. Please try again."

                    if i == 0:
                        last_message = await message.reply_text(fallback_text)
                    else:
                        last_message = await context.bot.send_message(
                            chat_id=update.effective_chat.id, text=fallback_text
                        )
                    sent_messages.append(last_message)

        # Store message IDs for future editing
        if sent_messages:
            if "bot_messages" not in context.user_data:
                context.user_data["bot_messages"] = {}
            context.user_data["bot_messages"][message.message_id] = [
                msg.message_id for msg in sent_messages
            ]
