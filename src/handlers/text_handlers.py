import io
import os
import aiofiles
from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, filters
from telegram.constants import ChatAction
from utils.telegramlog import telegram_logger
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from typing import List
import datetime
import logging
from services.model_handlers.factory import ModelHandlerFactory
import asyncio
from services.memory_manager import MemoryManager
from services.model_handlers.model_history_manager import ModelHistoryManager

# Use relative imports for handler utilities
from .message_context_handler import MessageContextHandler
from .response_formatter import ResponseFormatter
from .media_context_extractor import MediaContextExtractor
from services.media.image_processor import ImageProcessor
from services.model_handlers.prompt_formatter import PromptFormatter
from services.conversation_manager import ConversationManager


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

        # Initialize MemoryManager
        self.memory_manager = MemoryManager(
            db=user_data_manager.db if hasattr(user_data_manager, "db") else None,
        )
        self.memory_manager.short_term_limit = 15
        self.memory_manager.token_limit = 8192

        # Initialize model components with safe defaults
        self.model_registry = None
        self.user_model_manager = None

        # Instantiate the ModelHistoryManager with safe initialization
        self.model_history_manager = ModelHistoryManager(self.memory_manager)

        # Initialize utility classes
        self.context_handler = MessageContextHandler()
        self.response_formatter = ResponseFormatter()
        self.media_context_extractor = MediaContextExtractor()
        self.image_processor = ImageProcessor(gemini_api)
        self.prompt_formatter = PromptFormatter()
        self.conversation_manager = ConversationManager(
            self.memory_manager, self.model_history_manager
        )

        # Flag to track if model_registry has been initialized
        self.model_registry_initialized = False

    async def format_telegram_markdown(self, text: str) -> str:
        # Delegate to ResponseFormatter
        return await self.response_formatter.format_telegram_markdown(text)

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        # Delegate to ResponseFormatter
        return await self.response_formatter.split_long_message(text, max_length)

    async def handle_text_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message and not update.edited_message:
            return

        user_id = update.effective_user.id
        message = update.message or update.edited_message
        message_text = message.text

        # Extract quoted message using MessageContextHandler
        quoted_text, quoted_message_id = self.context_handler.extract_reply_context(
            message
        )

        # Define a unique conversation ID for this user
        conversation_id = f"user_{user_id}"

        try:
            # Delete old message if this is an edited message
            if update.edited_message and "bot_messages" in context.user_data:
                original_message_id = update.edited_message.message_id
                if original_message_id in context.user_data["bot_messages"]:
                    for msg_id in context.user_data["bot_messages"][
                        original_message_id
                    ]:
                        if msg_id:
                            try:
                                await context.bot.delete_message(
                                    chat_id=update.effective_chat.id, message_id=msg_id
                                )
                            except Exception as e:
                                self.logger.error(
                                    f"Error deleting old message: {str(e)}"
                                )
                    del context.user_data["bot_messages"][original_message_id]

            # In group chats, process only messages that mention the bot
            if update.effective_chat.type in ["group", "supergroup"]:
                bot_username = "@" + context.bot.username
                if bot_username not in message_text:
                    # Bot not mentioned, ignore message
                    return
                else:
                    # Remove all mentions of bot_username from the message text
                    message_text = message_text.replace(bot_username, "").strip()

            # Check if user is referring to images or documents using MessageContextHandler
            referring_to_image = self.context_handler.detect_reference_to_image(
                message_text
            )
            referring_to_document = self.context_handler.detect_reference_to_document(
                message_text
            )

            # Send initial "thinking" message
            thinking_message = await message.reply_text("Thinking...🧠")
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            # Get the preferred model with the ModelHandler system
            from services.user_preferences_manager import UserPreferencesManager

            preferences_manager = UserPreferencesManager(self.user_data_manager)

            # Get the preferred model using the UserPreferencesManager first so we can use it for history
            preferred_model = await preferences_manager.get_user_model_preference(
                user_id
            )
            self.logger.info(f"Preferred model for user {user_id}: {preferred_model}")

            # Get conversation history using ConversationManager - IMPROVED TO USE MODEL-SPECIFIC HISTORY
            history_context = []
            if preferred_model:
                # Get model-specific conversation history
                history_context = await self.conversation_manager.get_conversation_history(
                    user_id,
                    max_messages=self.max_context_length,
                    model=preferred_model,  # Pass the specific model to get its history
                )

                if not history_context:
                    self.logger.info(
                        f"No model-specific history found for {preferred_model}, using default history"
                    )
                    # Fall back to default history if model-specific history is not available
                    history_context = (
                        await self.conversation_manager.get_conversation_history(
                            user_id, max_messages=self.max_context_length
                        )
                    )
            else:
                # Use default history if no preferred model is specified
                history_context = (
                    await self.conversation_manager.get_conversation_history(
                        user_id, max_messages=self.max_context_length
                    )
                )

            self.logger.info(
                f"Retrieved {len(history_context)} message(s) from history for user {user_id}"
            )

            # Build enhanced prompt with relevant context
            enhanced_prompt = message_text
            context_added = False

            # Add quoted text context if this is a reply to another message
            if quoted_text:
                enhanced_prompt = self.prompt_formatter.add_context(
                    message_text, "quote", quoted_text
                )
                context_added = True

            # Add image context if relevant using MediaContextExtractor
            if (
                referring_to_image
                and "image_history" in context.user_data
                and context.user_data["image_history"]
            ):
                image_context = await self.media_context_extractor.get_image_context(
                    context.user_data
                )
                if context_added:
                    # We already have other context, so add this as additional context
                    enhanced_prompt += f"\n\nThe user is also referring to previously shared images. Image context:\n\n{image_context}"
                else:
                    enhanced_prompt = self.prompt_formatter.add_context(
                        message_text, "image", image_context
                    )
                    context_added = True

            # Add document context if relevant using MediaContextExtractor
            if (
                referring_to_document
                and "document_history" in context.user_data
                and context.user_data["document_history"]
            ):
                document_context = (
                    await self.media_context_extractor.get_document_context(
                        context.user_data
                    )
                )
                if context_added:
                    # We already have other context, so add this as additional context
                    enhanced_prompt += f"\n\nThe user is also referring to previously processed documents. Document context:\n\n{document_context}"
                else:
                    enhanced_prompt = self.prompt_formatter.add_context(
                        message_text, "document", document_context
                    )
                    context_added = True

            # Check if this is an image generation request using ImageProcessor
            is_image_request, image_prompt = (
                await self.image_processor.detect_image_generation_request(message_text)
            )

            if is_image_request and image_prompt:
                # Try to delete the thinking message first, but continue even if it fails
                if thinking_message is not None:
                    try:
                        await thinking_message.delete()
                        thinking_message = (
                            None  # Mark as deleted to avoid double deletion attempts
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not delete thinking message: {e}")
                        # Continue processing even if deletion fails
                        thinking_message = (
                            None  # Mark as deleted to avoid double deletion attempts
                        )

                # Inform the user that image generation is starting
                status_message = await update.message.reply_text(
                    "Generating image... This may take a moment."
                )

                try:
                    # Use the ImageProcessor to generate the image
                    image_bytes = await self.image_processor.generate_image(
                        image_prompt
                    )

                    if image_bytes and len(image_bytes) > 0:
                        # Delete the status message if it exists
                        if status_message is not None:
                            try:
                                await status_message.delete()
                                status_message = None  # Mark as deleted
                            except Exception as e:
                                self.logger.warning(
                                    f"Could not delete status message: {e}"
                                )
                                # Try to edit if delete fails
                                try:
                                    await status_message.edit_text(
                                        "✅ Image generated successfully!"
                                    )
                                    status_message = None  # Consider handled
                                except Exception:
                                    status_message = (
                                        None  # Consider handled even if edit fails
                                    )
                                    pass

                        try:
                            # Send the image
                            caption = f"Generated image of: {image_prompt}"
                            await update.message.reply_photo(
                                photo=io.BytesIO(image_bytes), caption=caption
                            )

                            # Update user stats
                            if self.user_data_manager:
                                await self.user_data_manager.update_stats(
                                    user_id, image_generation=True
                                )

                            # Save interaction to conversation history using ConversationManager
                            await self.conversation_manager.save_media_interaction(
                                user_id,
                                "generated_image",
                                f"Generate an image of: {image_prompt}",
                                f"Here's the image I generated of {image_prompt}.",
                            )
                        except Exception as send_error:
                            self.logger.error(
                                f"Error sending generated image: {str(send_error)}"
                            )
                            await update.message.reply_text(
                                f"I generated the image but couldn't send it due to an error: {str(send_error)}"
                            )

                        # Return early since we've handled the request
                        return
                    else:
                        # Update status message if image generation failed
                        self.logger.warning(
                            f"Image generation returned empty result for prompt: {image_prompt}"
                        )
                        if status_message is not None:
                            try:
                                await status_message.edit_text(
                                    "Sorry, I couldn't generate that image. Please try with a different description."
                                )
                                status_message = None  # Mark as handled
                            except Exception as edit_error:
                                self.logger.warning(
                                    f"Could not edit status message: {str(edit_error)}"
                                )
                                try:
                                    # If edit fails, try sending a new message
                                    await update.message.reply_text(
                                        "Sorry, I couldn't generate that image. Please try with a different description."
                                    )
                                except Exception:
                                    pass
                        # Continue with normal text response as fallback
                except Exception as e:
                    self.logger.error(f"Error generating image: {e}")
                    if status_message is not None:
                        try:
                            await status_message.edit_text(
                                "Sorry, there was an error generating your image. Please try again later."
                            )
                            status_message = None  # Mark as handled
                        except Exception as edit_error:
                            self.logger.warning(
                                f"Could not edit status message: {str(edit_error)}"
                            )
                            try:
                                # If edit fails, try sending a new message
                                await update.message.reply_text(
                                    "Sorry, there was an error generating your image. Please try again later."
                                )
                            except Exception:
                                pass

            # Try to get model registry and user model manager from application context
            if hasattr(context, "application") and hasattr(
                context.application, "bot_data"
            ):
                if not self.model_registry:
                    self.model_registry = context.application.bot_data.get(
                        "model_registry"
                    )
                if not self.user_model_manager:
                    self.user_model_manager = context.application.bot_data.get(
                        "user_model_manager"
                    )

                # If we have the model registry and user model manager, update the ModelHistoryManager
                if (
                    self.model_registry
                    and self.user_model_manager
                    and hasattr(self, "model_history_manager")
                ):
                    # Only update the ModelHistoryManager once
                    if (
                        not hasattr(self.model_history_manager, "model_registry")
                        or not self.model_history_manager.model_registry
                    ):
                        # Create a new ModelHistoryManager with model_registry and user_model_manager
                        from services.model_handlers.model_history_manager import (
                            ModelHistoryManager,
                        )

                        self.model_history_manager = ModelHistoryManager(
                            self.memory_manager,
                            self.user_model_manager,
                            self.model_registry,
                        )
                        self.logger.info(
                            "Updated ModelHistoryManager with model_registry and user_model_manager"
                        )

                        # Update the ConversationManager with the new ModelHistoryManager
                        self.conversation_manager = ConversationManager(
                            self.memory_manager, self.model_history_manager
                        )

            # Safe check for model_registry before accessing it
            if self.model_registry and self.model_history_manager:
                try:
                    selected_model = self.model_history_manager.get_selected_model(
                        user_id
                    )
                    model_config = self.model_registry.get_model_config(selected_model)
                    self.logger.info(f"Current model: {model_config}")
                except Exception as e:
                    self.logger.error(f"Error accessing model config: {e}")
                    # Fall back to using preferred_model directly
                    self.logger.info(
                        f"Falling back to preferred model: {preferred_model}"
                    )
            else:
                self.logger.info(
                    "model_registry or model_history_manager is not available, using preferred_model directly"
                )

            # Apply response style guidelines using PromptFormatter
            enhanced_prompt_with_guidelines = await self.prompt_formatter.apply_response_guidelines(
                enhanced_prompt,
                ModelHandlerFactory.get_model_handler(
                    preferred_model,
                    gemini_api=self.gemini_api,
                    openrouter_api=self.openrouter_api,
                    deepseek_api=self.deepseek_api,  # Pass the deepseek_api instance
                ),
                context,
            )

            # Get model timeout
            model_timeout = 60.0  # Default timeout
            if self.user_model_manager:
                model_config = self.user_model_manager.get_user_model_config(user_id)
                model_timeout = model_config.timeout_seconds if model_config else 60.0

            try:
                # Get the model handler with all API instances properly provided
                model_handler = ModelHandlerFactory.get_model_handler(
                    preferred_model,
                    gemini_api=self.gemini_api,
                    openrouter_api=self.openrouter_api,
                    deepseek_api=self.deepseek_api,  # Pass the deepseek_api instance
                )

                # Generate response using the model handler with proper timeout from model_config
                response = await asyncio.wait_for(
                    model_handler.generate_response(
                        prompt=enhanced_prompt_with_guidelines,
                        context=history_context,
                        temperature=0.7,
                        max_tokens=4000,
                        quoted_message=quoted_text,  # Pass the quoted message to the model handler
                    ),
                    timeout=model_timeout,
                )
            except asyncio.TimeoutError:
                await thinking_message.delete()
                await message.reply_text(
                    "Sorry, the request took too long to process. Please try again later.",
                    parse_mode="MarkdownV2",
                )
                return
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                await thinking_message.delete()
                await message.reply_text(
                    "Sorry, there was an error processing your request. Please try again later.",
                    parse_mode="MarkdownV2",
                )
                return

            if response is None:
                await thinking_message.delete()
                await message.reply_text(
                    "Sorry, I couldn't generate a response\\. Please try rephrasing your message\\.",
                    parse_mode="MarkdownV2",
                )
                return

            # Split long messages using ResponseFormatter and send them
            message_chunks = await self.response_formatter.split_long_message(response)
            await thinking_message.delete()

            # Store the message IDs for potential editing
            sent_messages = []

            # Get model indicator from the model handler
            model_indicator = model_handler.get_model_indicator()
            context.user_data["last_message_indicator"] = model_indicator

            # Determine if this is a reply
            is_reply = self.context_handler.should_use_reply_format(
                quoted_text, quoted_message_id
            )

            for i, chunk in enumerate(message_chunks):
                try:
                    # Format response with model indicator using ResponseFormatter
                    if i == 0:
                        text_to_send = (
                            self.response_formatter.format_with_model_indicator(
                                chunk, model_indicator, is_reply
                            )
                        )
                    else:
                        text_to_send = chunk

                    # Format with telegramify-markdown using ResponseFormatter
                    formatted_chunk = (
                        await self.response_formatter.format_telegram_markdown(
                            text_to_send
                        )
                    )

                    if i == 0:
                        # For first chunk, use reply_to_message_id if this is a reply
                        if is_reply:
                            last_message = await message.reply_text(
                                formatted_chunk,
                                parse_mode="MarkdownV2",
                                disable_web_page_preview=True,
                                reply_to_message_id=quoted_message_id,
                            )
                        else:
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
                    self.logger.error(f"Formatting failed: {str(formatting_error)}")
                    if i == 0:
                        last_message = await message.reply_text(
                            chunk.replace("*", "").replace("_", "").replace("`", ""),
                            parse_mode=None,
                        )
                    else:
                        last_message = await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=chunk.replace("*", "")
                            .replace("_", "")
                            .replace("`", ""),
                            parse_mode=None,
                        )
                    sent_messages.append(last_message)

            # Save interaction to conversation history using ConversationManager
            if response:
                if quoted_text:
                    # If there was a quoted message, use special handling
                    await self.conversation_manager.add_quoted_message_context(
                        user_id, quoted_text, message_text, response
                    )
                else:
                    # Regular message pair
                    await self.conversation_manager.save_message_pair(
                        user_id, message_text, response, preferred_model
                    )

                # Store the message IDs in context for future editing
                if "bot_messages" not in context.user_data:
                    context.user_data["bot_messages"] = {}
                context.user_data["bot_messages"][message.message_id] = [
                    msg.message_id for msg in sent_messages
                ]

            telegram_logger.log_message(f"Text response sent successfully", user_id)
        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            if "thinking_message" in locals():
                await thinking_message.delete()
            await update.message.reply_text(
                "Sorry, I encountered an error\\. Please try again later\\.",
                parse_mode="MarkdownV2",
            )

    async def handle_image(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming image messages with AI analysis."""
        user_id = update.effective_user.id
        self.logger.info(f"Processing image from user {user_id}")

        try:
            # Get the largest available photo
            photo = update.message.photo[-1]
            caption = update.message.caption or "What's in this image?"

            # Send typing action and initial status message
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )
            status_message = await update.message.reply_text(
                "Analyzing image... This may take a moment."
            )

            # Download the image
            image_file = await context.bot.get_file(photo.file_id)
            image_bytes_io = io.BytesIO()
            await image_file.download_to_memory(image_bytes_io)
            image_bytes_io.seek(0)

            # Store in user context for future reference
            if "image_history" not in context.user_data:
                context.user_data["image_history"] = []

            # Add to the beginning (most recent first)
            context.user_data["image_history"].insert(
                0,
                {
                    "file_id": photo.file_id,
                    "caption": caption,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "width": photo.width,
                    "height": photo.height,
                },
            )

            # Limit the history size
            if len(context.user_data["image_history"]) > 5:
                context.user_data["image_history"] = context.user_data["image_history"][
                    :5
                ]

            # Analyze the image with a prompt
            response = await self.image_processor.analyze_image(image_bytes_io, caption)

            # Delete status message
            await status_message.delete()

            if response:
                # Format the response with model indicator
                model_indicator = "🧠 Gemini"
                text_to_send = self.response_formatter.format_with_model_indicator(
                    response, model_indicator
                )

                # Format with telegramify-markdown
                formatted_response = (
                    await self.response_formatter.format_telegram_markdown(text_to_send)
                )

                # Send the formatted response
                await update.message.reply_text(
                    formatted_response,
                    parse_mode="MarkdownV2",
                    disable_web_page_preview=True,
                )

                # Save interaction to conversation history using ConversationManager
                if hasattr(self, "conversation_manager"):
                    await self.conversation_manager.save_media_interaction(
                        user_id,
                        "image",
                        f"[Image shared with caption: {caption}]",
                        response,
                    )

                # Update user stats
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, image=True)
            else:
                await update.message.reply_text(
                    "Sorry, I couldn't analyze this image. Please try again with a different image or question."
                )
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error while processing your image. Please try again later."
            )
