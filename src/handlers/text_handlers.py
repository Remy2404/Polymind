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

        # These will be set later when application context is available
        self.model_registry = None
        self.user_model_manager = None

        # Instantiate the ModelHistoryManager (will be updated with model_registry later if available)
        self.model_history_manager = ModelHistoryManager(self.memory_manager)

    async def format_telegram_markdown(self, text: str) -> str:
        try:
            from telegramify_markdown import convert

            formatted_text = convert(text)
            return formatted_text
        except Exception as e:
            self.logger.error(f"Error formatting markdown: {str(e)}")
            return text.replace("*", "").replace("_", "").replace("`", "")

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        if len(text) <= max_length:
            return [text]

        chunks = []
        current_chunk = ""

        for line in text.split("\n"):
            if len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def handle_text_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.message and not update.edited_message:
            return

        user_id = update.effective_user.id
        message = update.message or update.edited_message
        message_text = message.text

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
                        try:
                            await context.bot.delete_message(
                                chat_id=update.effective_chat.id, message_id=msg_id
                            )
                        except Exception as e:
                            self.logger.error(f"Error deleting old message: {str(e)}")
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

            # Check if user is referring to images or documents
            image_related_keywords = [
                "image",
                "picture",
                "photo",
                "pic",
                "img",
                "that image",
                "the picture",
            ]
            document_related_keywords = [
                "document",
                "doc",
                "file",
                "pdf",
                "that document",
                "the file",
                "the pdf",
                "tell me more",
                "more information",
                "more details",
                "explain further",
                "tell me about it",
                "what else",
                "elaborate",
            ]

            referring_to_image = any(
                keyword in message_text.lower() for keyword in image_related_keywords
            )
            referring_to_document = any(
                keyword in message_text.lower() for keyword in document_related_keywords
            )

            # Send initial "thinking" message
            thinking_message = await message.reply_text("Thinking...🧠")
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id, action=ChatAction.TYPING
            )

            # Get conversation history using the new manager
            history_context = await self.model_history_manager.get_history(
                user_id, max_messages=self.max_context_length
            )

            # Build enhanced prompt with relevant context
            enhanced_prompt = message_text
            context_added = False

            # Add image context if relevant
            if (
                referring_to_image
                and "image_history" in context.user_data
                and context.user_data["image_history"]
            ):
                image_context = await self.get_image_context(user_id, context)
                enhanced_prompt = f"The user is referring to previously shared images. Here's the context of those images:\n\n{image_context}\n\nUser's question: {message_text}"
                context_added = True

            # Add document context if relevant
            if (
                referring_to_document
                and "document_history" in context.user_data
                and context.user_data["document_history"]
            ):
                document_context = await self.get_document_context(user_id, context)
                if context_added:
                    enhanced_prompt += f"\n\nThe user is also referring to previously processed documents. Document context:\n\n{document_context}"
                else:
                    enhanced_prompt = f"The user is referring to previously processed documents. Here's the context of those documents:\n\n{document_context}\n\nUser's question: {message_text}"
                    context_added = True

            # Add this after existing document context check
            if (
                (
                    "tell me more" in message_text.lower()
                    or "more details" in message_text.lower()
                )
                and "document_history" in context.user_data
                and context.user_data["document_history"]
            ):
                document_context = await self.get_document_context(user_id, context)
                enhanced_prompt = f"The user wants more information about the previously analyzed document. Here's the document context:\n\n{document_context}\n\nProvide more detailed analysis focusing on aspects not covered in the initial response."
                context_added = True

            # Check if this is an image generation request
            is_image_request, image_prompt = await self.detect_image_generation_request(
                message_text
            )

            if is_image_request and image_prompt:
                # Try to delete the thinking message first, but continue even if it fails
                try:
                    await thinking_message.delete()
                except Exception as e:
                    self.logger.warning(f"Could not delete thinking message: {e}")
                    # Continue processing even if deletion fails

                # Inform the user that image generation is starting
                status_message = await update.message.reply_text(
                    "Generating image... This may take a moment."
                )

                try:
                    # First try with the primary image generation method (using gemini-2.0-flash-exp-image-generation)
                    image_bytes = await self.gemini_api.generate_image(image_prompt)

                    if image_bytes:
                        # Delete the status message
                        await status_message.delete()

                        # Send the image
                        caption = f"Generated image of: {image_prompt}"
                        await update.message.reply_photo(
                            photo=image_bytes, caption=caption
                        )

                        # Update user stats
                        if self.user_data_manager:
                            self.user_data_manager.update_stats(
                                user_id, image_generation=True
                            )

                        # Store the response in both storage systems
                        # Add to MemoryManager
                        await self.memory_manager.add_user_message(
                            conversation_id,
                            f"Generate an image of: {image_prompt}",
                            str(user_id),
                        )
                        await self.memory_manager.add_assistant_message(
                            conversation_id,
                            f"Here's the image I generated of {image_prompt}.",
                        )

                        # Also add to existing system for backward compatibility
                        self.user_data_manager.add_to_context(
                            user_id,
                            {
                                "role": "user",
                                "content": f"Generate an image of: {image_prompt}",
                            },
                        )
                        self.user_data_manager.add_to_context(
                            user_id,
                            {
                                "role": "assistant",
                                "content": f"Here's the image I generated of {image_prompt}.",
                            },
                        )

                        # Return early since we've handled the request
                        return
                    else:
                        # Update status message if image generation failed
                        await status_message.edit_text(
                            "Sorry, I couldn't generate that image. Please try a different description or use the /imagen3 command."
                        )
                        # Continue with normal text response as fallback
                except Exception as e:
                    self.logger.error(f"Error generating image: {e}")
                    await status_message.edit_text(
                        "Sorry, there was an error generating your image. Please try again later."
                    )  # Get the preferred model with the new ModelManager system

            # Try to get the model registry and user model manager from application context
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

            # Get the preferred model using the new system if available, otherwise fall back to legacy approach
            if self.user_model_manager:
                preferred_model = self.user_model_manager.get_user_model(user_id)
                self.logger.info(
                    f"Using model from UserModelManager for user {user_id}: {preferred_model}"
                )

                # Get model config for additional info like timeout
                model_config = self.user_model_manager.get_user_model_config(user_id)
                model_timeout = model_config.timeout_seconds if model_config else 60
            else:
                # Legacy fallback method
                preferred_model = await self.user_data_manager.get_user_preference(
                    user_id, "preferred_model", default="gemini"
                )
                self.logger.info(
                    f"Preferred model for user {user_id}: {preferred_model}"
                )

                # Check for temporary flags from a recent model switch
                if (
                    "just_switched_model" in context.user_data
                    and context.user_data["just_switched_model"]
                ):
                    preferred_model = context.user_data.get(
                        "switched_to_model", preferred_model
                    )
                    self.logger.info(
                        f"Using recently switched model for user {user_id}: {preferred_model}"
                    )
                    # Clear the temporary flags after a few messages
                    if "model_switch_counter" not in context.user_data:
                        context.user_data["model_switch_counter"] = 1
                    else:
                        context.user_data["model_switch_counter"] += 1
                    # After 5 messages, we can trust the DB to have the correct preference
                    if context.user_data["model_switch_counter"] >= 5:
                        context.user_data["just_switched_model"] = False
                        context.user_data["model_switch_counter"] = 0
                        self.logger.info(
                            f"Cleared temporary model switch flags for user {user_id}"
                        )

                # Set timeout based on model
                model_timeout = (
                    300.0 if preferred_model in ["deepseek", "optimus-alpha"] else 60.0
                )

            # Apply response style guidelines
            enhanced_prompt_with_guidelines = await self._apply_response_guidelines(
                enhanced_prompt, preferred_model, context
            )
            try:
                # Get the model handler for the preferred model
                # Use the class's API instances directly
                model_handler = ModelHandlerFactory.get_model_handler(
                    preferred_model,
                    gemini_api=self.gemini_api,
                    openrouter_api=self.openrouter_api,
                    deepseek_api=self.deepseek_api,
                )

                # Generate response using the model handler with proper timeout from model_config
                response = await asyncio.wait_for(
                    model_handler.generate_response(
                        prompt=enhanced_prompt_with_guidelines,
                        context=history_context,
                        temperature=0.7,
                        max_tokens=4000,
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

            # Split long messages and send them, then delete the thinking message
            message_chunks = await self.split_long_message(response)
            await thinking_message.delete()

            # Store the message IDs for potential editing
            sent_messages = []

            # Get the model indicator from the model handler
            model_handler = ModelHandlerFactory.get_model_handler(
                preferred_model,
                gemini_api=self.gemini_api,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )
            model_indicator = model_handler.get_model_indicator()

            # Store the model indicator for model consistency checking in future messages
            context.user_data["last_message_indicator"] = model_indicator

            for i, chunk in enumerate(message_chunks):
                try:
                    # Add model indicator to first message only
                    text_to_send = chunk
                    if i == 0:
                        text_to_send = f"{model_indicator}\n\n{chunk}"

                    # Format with telegramify-markdown
                    formatted_chunk = await self.format_telegram_markdown(text_to_send)
                    if i == 0:
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

            # Save user message and assistant response using the new manager
            if response:
                await self.model_history_manager.save_message_pair(
                    user_id, message_text, response
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

    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        # Define conversation ID for this user - Handled by ModelHistoryManager now
        # conversation_id = f"user_{user_id}"
        telegram_logger.log_message("Processing an image", user_id)

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.TYPING
        )

        try:
            # In group chats, process only images that mention the bot
            if update.effective_chat.type in ["group", "supergroup"]:
                bot_username = "@" + context.bot.username
                caption = update.message.caption or ""
                if bot_username not in caption:
                    # Bot not mentioned, ignore message
                    return
                else:
                    # Remove all mentions of bot_username from the caption
                    caption = caption.replace(bot_username, "").strip()
            else:
                caption = (
                    update.message.caption
                    or "Please analyze this image and describe it."
                )

            photo = update.message.photo[-1]
            image_file = await context.bot.get_file(photo.file_id)
            if image_file is None:
                await update.message.reply_text(
                    "Failed to retrieve image file. Please try again."
                )
                return
            image_bytes = await image_file.download_as_bytearray()

            # Use the updated analyze_image method
            response = await self.gemini_api.analyze_image(image_bytes, caption)

            if response:
                # Split the response into chunks
                response_chunks = await self.split_long_message(response)
                sent_messages = []

                # Send each chunk
                for chunk in response_chunks:
                    try:
                        # Format with telegramify-markdown
                        formatted_chunk = await self.format_telegram_markdown(chunk)
                        sent_message = await update.message.reply_text(
                            formatted_chunk,
                            parse_mode="MarkdownV2",
                            disable_web_page_preview=True,
                        )
                        sent_messages.append(sent_message.message_id)
                    except Exception as formatting_error:
                        self.logger.warning(
                            f"Markdown formatting failed: {formatting_error}"
                        )
                        # Try without markdown formatting and remove special characters
                        sent_message = await update.message.reply_text(
                            chunk.replace("*", "").replace("_", "").replace("`", ""),
                            parse_mode=None,
                        )
                        sent_messages.append(sent_message.message_id)

                # Store the image interaction using ModelHistoryManager
                await self.model_history_manager.save_image_interaction(
                    user_id, caption, response
                )

                # Store image reference in user data for future reference (non-history context)
                if "image_history" not in context.user_data:
                    context.user_data["image_history"] = []

                # Store image metadata
                context.user_data["image_history"].append(
                    {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "file_id": photo.file_id,
                        "caption": caption,
                        "description": response,
                        "message_id": update.message.message_id,
                        "response_message_ids": sent_messages,  # Now storing all message IDs
                    }
                )

                # Update user stats for image
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, image=True)

                telegram_logger.log_message(
                    f"Image analysis completed successfully", user_id
                )
            else:
                await update.message.reply_text(
                    "Sorry, I couldn't analyze the image\\. Please try again\\.",
                    parse_mode="MarkdownV2",
                )

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            await update.message.reply_text(
                "Sorry, I couldn't process your image\\. The response might be too long or there might be an issue with the image format\\. Please try a different image or a more specific question\\.",
                parse_mode="MarkdownV2",
            )

    async def show_history(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_id = update.effective_user.id
        # conversation_id = f"user_{user_id}" # Not needed directly

        # Get messages using ModelHistoryManager
        messages = await self.model_history_manager.get_history(
            user_id, max_messages=50
        )  # Get more messages for display

        if not messages:
            await update.message.reply_text(
                "You don't have any conversation history yet."
            )
            return

        # Build history text
        history_text = "Your conversation history:"
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            history_text += f"{role}: {content}\n\n"

        # Split long messages
        message_chunks = await self.split_long_message(history_text)

        for chunk in message_chunks:
            await update.message.reply_text(chunk)

    async def get_image_context(
        self, user_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> str:
        """Generate context from previously processed images"""
        if (
            "image_history" not in context.user_data
            or not context.user_data["image_history"]
        ):
            return ""

        # Get the 3 most recent images
        recent_images = context.user_data["image_history"][-3:]

        image_context = "Recently analyzed images:\n"
        for idx, img in enumerate(recent_images):
            image_context += f"[Image {idx+1}]: Caption: {img['caption']}\nDescription: {img['description'][:100]}...\n\n"

        return image_context

    async def get_document_context(
        self, user_id: int, context: ContextTypes.DEFAULT_TYPE
    ) -> str:
        """Generate richer context from previously processed documents"""
        if (
            "document_history" not in context.user_data
            or not context.user_data["document_history"]
        ):
            return ""

        # Get the most recent document (the one the user is likely referring to)
        most_recent = context.user_data["document_history"][-1]

        document_context = f"Recently analyzed document: {most_recent['file_name']}\n\n"
        document_context += f"Full content summary:\n{most_recent['full_response']}\n\n"

        # Add a special instruction for the AI
        document_context += "Please provide additional details or answer follow-up questions about this document."

        return document_context

    async def detect_image_generation_request(self, text: str) -> tuple[bool, str]:
        """
        Detect if a message is requesting image generation and extract the prompt.

        Returns:
            tuple: (is_image_request, image_prompt)
        """
        # Lowercase for easier matching
        text_lower = text.lower().strip()

        # Define image generation trigger phrases
        image_triggers = [
            "generate an image",
            "generate image",
            "create an image",
            "create image",
            "make an image",
            "make image",
            "draw",
            "generate a picture",
            "create a picture",
            "generate img",
            "generate an img",
            "create img",
            "make img",
            "generate a photo",
            "image of",
            "picture of",
            "photo of",
            "draw me",
            "generate me an image",
            "create me an image",
            "make me an image",
            "generate me a picture",
            "can you generate an image",
            "can you create an image",
            "i want an image of",
            "please make an image",
        ]

        # Check if any trigger phrase is in the message
        is_image_request = any(trigger in text_lower for trigger in image_triggers)

        if is_image_request:
            # Extract the prompt: Find the first trigger that matches and get everything after it
            image_prompt = text
            for trigger in sorted(image_triggers, key=len, reverse=True):
                if trigger in text_lower:
                    # Find the trigger position and extract everything after it
                    trigger_pos = text_lower.find(trigger)
                    prompt_start = trigger_pos + len(trigger)

                    # Clean up the prompt - remove words like "of", "about", etc. at the beginning
                    raw_prompt = text[prompt_start:].strip()
                    clean_words = [
                        "of",
                        "about",
                        "showing",
                        "depicting",
                        "that shows",
                        "with",
                        ":",
                        "-",
                    ]

                    for word in clean_words:
                        if raw_prompt.lower().startswith(word + " "):
                            raw_prompt = raw_prompt[len(word) :].strip()

                    image_prompt = raw_prompt
                    break

            # If we have a non-empty prompt, it's an image request
            if image_prompt and len(image_prompt) > 0:
                return True, image_prompt

        return False, ""

    async def _apply_response_guidelines(
        self, prompt: str, preferred_model: str, context=None
    ) -> str:
        """Apply appropriate response style guidelines based on the selected model."""
        try:
            guidelines_path = os.path.join(
                os.path.dirname(__file__), "..", "doc", "dataset.md"
            )
            async with aiofiles.open(guidelines_path, "r", encoding="utf-8") as file:
                content = await file.read()

            # Get the model handler for the preferred model
            # Use the class's API instances directly
            model_handler = ModelHandlerFactory.get_model_handler(
                preferred_model,
                gemini_api=self.gemini_api,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )

            # Get system message from the model handler
            system_message = model_handler.get_system_message()

            # Create enhanced prompt with system message and user query
            enhanced_prompt = f"{system_message}\n\nUser query: {prompt}"

            return enhanced_prompt
        except Exception as e:
            self.logger.error(f"Error applying response guidelines: {str(e)}")
            return prompt  # Return original prompt if there was an error

    def get_handlers(self):
        return [
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message),
            MessageHandler(filters.PHOTO, self.handle_image),
        ]

    async def reset_conversation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset a user's conversation history using ModelHistoryManager."""
        user_id = update.effective_user.id
        # conversation_id = f"user_{user_id}" # Not needed directly

        # Clear conversation using ModelHistoryManager
        await self.model_history_manager.clear_history(user_id)

        # Clear other non-history context data in user_data (this remains)
        if "image_history" in context.user_data:
            context.user_data["image_history"] = []
        if "document_history" in context.user_data:
            context.user_data["document_history"] = []
        if "bot_messages" in context.user_data:
            context.user_data["bot_messages"] = {}

        await update.message.reply_text("✅ Your conversation history has been reset.")
        telegram_logger.log_message(f"Conversation history cleared", user_id)
