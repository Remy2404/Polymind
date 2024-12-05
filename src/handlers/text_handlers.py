from telegram import Update
from telegram.constants import ChatAction
from utils.telegramlog import TelegramLogger
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from typing import List
import logging
import io
from telegram.ext import (
    ContextTypes,
    MessageHandler,
    filters,
    CommandHandler,
    ConversationHandler,
    CallbackQueryHandler
)

# Define conversation states
GENERATE_IMG_PROMPT = 1
GENERATE_IMG_CONFIRM = 2

class TextHandler:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager, telegram_logger: TelegramLogger):
        self.logger = logging.getLogger(__name__)
        self.gemini_api = gemini_api
        self.telegram_logger = telegram_logger  # Corrected attribute name
        self.user_data_manager = user_data_manager
        self.max_context_length = 5

    async def format_telegram_markdown(self, text: str) -> str:
        try:
            from telegramify_markdown import convert
            formatted_text = convert(text)
            return formatted_text
        except Exception as e:
            self.logger.error(f"Error formatting markdown: {str(e)}")
            return text.replace('*', '').replace('_', '').replace('`', '')

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            if len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line
        
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        message_text = update.message.text

        try:
            # In group chats, process only messages that mention the bot
            if update.effective_chat.type in ['group', 'supergroup']:
                bot_username = '@' + context.bot.username
                if bot_username not in message_text:
                    # Bot not mentioned, ignore message
                    return
                else:
                    # Remove all mentions of bot_username from the message text
                    message_text = message_text.replace(bot_username, '').strip()

            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            # Get user context
            user_context = self.user_data_manager.get_user_context(user_id)
            
            # Generate response
            response = await self.gemini_api.generate_response(
                prompt=message_text,
                context=user_context[-self.max_context_length:]
            )

            if response is None:
                raise ValueError("Gemini API returned None response")

            # Split long messages
            message_chunks = await self.split_long_message(response)

            for chunk in message_chunks:
                try:
                    # Format with telegramify-markdown
                    formatted_chunk = await self.format_telegram_markdown(chunk)
                    await update.message.reply_text(
                        formatted_chunk,
                        parse_mode='MarkdownV2',
                        disable_web_page_preview=False,
                    )
                except Exception as formatting_error:
                    self.logger.error(f"Formatting failed: {str(formatting_error)}")
                    await update.message.reply_text(chunk.replace('*', '').replace('_', '').replace('`', ''), parse_mode=None)

            # Update user context
            self.user_data_manager.add_to_context(user_id, {"role": "user", "content": message_text})
            self.user_data_manager.add_to_context(user_id, {"role": "assistant", "content": response})

            self.telegram_logger.log_message(f"Text response sent successfully", user_id)

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error\\. Please try again\\.",
                parse_mode='MarkdownV2'
            )
        else:
            self.logger.info("Message processed successfully")

    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        self.telegram_logger.log_message("Processing an image", user_id)
    
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
        try:
            # In group chats, process only images that mention the bot
            if update.effective_chat.type in ['group', 'supergroup']:
                bot_username = '@' + context.bot.username
                caption = update.message.caption or ""
                if bot_username not in caption:
                    # Bot not mentioned, ignore message
                    return
                else:
                    # Remove all mentions of bot_username from the caption
                    caption = caption.replace(bot_username, '').strip()
            else:
                caption = update.message.caption or "Please analyze this image and describe it."

            photo = update.message.photo[-1]
            image_file = await context.bot.get_file(photo.file_id)
            image_bytes = await image_file.download_as_bytearray()
            
            response = await self.gemini_api.analyze_image(image_bytes, caption)

            if response:
                try:
                    formatted_response = await self.format_telegram_markdown(response)
                    await update.message.reply_text(
                        formatted_response,
                        parse_mode='MarkdownV2',
                        disable_web_page_preview=False,
                    )
                except Exception as markdown_error:
                    self.logger.warning(f"Markdown formatting failed: {markdown_error}")
                    await update.message.reply_text(response, parse_mode=None)

                # Update user stats for image
                if self.user_data_manager:
                    await self.user_data_manager.update_stats(user_id, image=True)

                self.telegram_logger.log_message(f"Image analysis completed: {response}", user_id)
            else:
                await update.message.reply_text("Sorry, I couldn't analyze the image. Please try again.")

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            await update.message.reply_text(
                "Sorry, I couldn't process your image. Please try a different one or ensure it's in JPEG/PNG format."
            )

    async def handle_generate_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        prompt = ' '.join(context.args)  # Join all arguments as the prompt
    
        if not prompt:
            await update.message.reply_text("Please provide a prompt after the /generate_image command.")
            return
    
        self.telegram_logger.log_message(f"Generating image for user {user_id} with prompt: {prompt}", user_id)
    
        # Show typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
        try:
            image_bytes = await self.gemini_api.generate_image(prompt)
    
            if image_bytes:
                # Indicate uploading photo
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
                
                await update.message.reply_photo(photo=io.BytesIO(image_bytes))
                self.telegram_logger.log_message(f"Image sent to user {user_id}", user_id)
            else:
                await update.message.reply_text("❌ Failed to generate image. Please try again later.")
    
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            await update.message.reply_text("An error occurred while generating the image. Please try again later.")
    
        # Update user stats
        if self.user_data_manager:
            await self.user_data_manager.update_stats(user_id, generated_images=True)

    async def show_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        history = self.user_data_manager.get_user_context(user_id)
        
        if not history:
            await update.message.reply_text("You don't have any conversation history yet.")
            return

        history_text = "Your conversation history:\n\n"
        for entry in history:
            role = entry['role'].capitalize()
            content = entry['content']
            history_text += f"{role}: {content}\n\n"

        # Split long messages
        message_chunks = await self.split_long_message(history_text)

        for chunk in message_chunks:
            await update.message.reply_text(chunk)

    async def start_generate_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Initiate the image generation process."""
        user_id = update.effective_user.id
        self.logger.info(f"User {user_id} initiated image generation.")

        await update.message.reply_text(
            "🎨 *Image Generation*\nPlease enter a descriptive prompt for the image you want to generate:",
            parse_mode='MarkdownV2'
        )

        return GENERATE_IMG_PROMPT

    async def receive_generate_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Receive the image prompt from the user and generate the image."""
        user_id = update.effective_user.id
        prompt = update.message.text.strip()
        self.logger.info(f"User {user_id} provided prompt for image generation: {prompt}")

        if not prompt:
            await update.message.reply_text(
                "❌ The prompt cannot be empty. Please enter a valid description:"
            )
            return GENERATE_IMG_PROMPT

        # Rate limiting
        try:
            await self.user_data_manager.acquire_rate_limit(user_id)
        except Exception as e:
            self.logger.warning(f"Rate limit exceeded for user {user_id}: {e}")
            await update.message.reply_text(
                "⚠️ You have reached the maximum number of image generation requests for this hour. Please try again later."
            )
            return ConversationHandler.END

        # Show 'typing' action while processing
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        try:
            # Generate image using Gemini API
            image_bytes = await self.gemini_api.generate_image(prompt)

            if image_bytes:
                # Show 'uploading photo' action
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)

                await update.message.reply_photo(photo=io.BytesIO(image_bytes))
                self.logger.info(f"Image successfully generated and sent to user {user_id}.")
                
                # After successfully generating and sending the image
                if self.user_data_manager:
                    await self.user_data_manager.update_stats(user_id, image=True)  # Use 'image' instead of 'generated_images'
            else:
                await update.message.reply_text("❌ Failed to generate image. Please try again later.")

        except Exception as e:
            self.logger.error(f"Error generating image for user {user_id}: {e}")
            await update.message.reply_text(
                "❌ An error occurred while generating the image. Please try again later."
            )

        return ConversationHandler.END

    async def confirm_generate_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Confirm image generation based on user input."""
        user_id = update.effective_user.id
        response = update.message.text.lower()

        if response == 'yes':
            prompt = context.user_data.get('prompt')
            if not prompt:
                await update.message.reply_text("❌ No prompt found. Please start again.")
                return ConversationHandler.END

            self.logger.info(f"User {user_id} confirmed image generation for prompt: {prompt}")

            # Rate limiting
            try:
                await self.user_data_manager.acquire_rate_limit(user_id)
                
                # Send processing message
                processing_msg = await update.message.reply_text("🎨 Generating your image, please wait...")
                
                try:
                    # Assuming you're using an image generation API like DALL-E or Stable Diffusion
                    image_url = await self.image_generator.generate_image(prompt)
                    
                    if image_url:
                        # Send the generated image
                        await update.message.reply_photo(
                            photo=image_url,
                            caption=f"🎨 Generated image for: {prompt}"
                        )
                        await processing_msg.delete()
                        # After successfully generating and sending the image
                        if self.user_data_manager:
                            await self.user_data_manager.update_stats(user_id, image=True)  # Use 'image' instead of 'generated_images'
                    else:
                        await processing_msg.edit_text("❌ Failed to generate image. Please try again.")
                except Exception as e:
                    self.logger.error(f"Image generation error: {str(e)}")
                    await processing_msg.edit_text("❌ Error during image generation. Please try again later.")
            except Exception as e:
                self.logger.error(f"Error generating image for user {user_id}: {e}")
                await update.message.reply_text("❌ An error occurred while generating the image.")
 
        return ConversationHandler.END

    async def cancel_generate_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancel the image generation process."""
        user_id = update.effective_user.id
        self.logger.info(f"User {user_id} canceled the image generation process.")

        await update.message.reply_text("🛑 Image generation process has been canceled.")

        return ConversationHandler.END

    def get_handlers(self):
        """Return a list of handlers to be added to the application."""
        return [
            CommandHandler("history", self.show_history),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message),
            MessageHandler(filters.PHOTO, self.handle_image),
    
            # ConversationHandler for /generate_img
            ConversationHandler(
                entry_points=[CommandHandler('generate_img', self.start_generate_image)],
                states={
                    GENERATE_IMG_PROMPT: [
                        MessageHandler(filters.TEXT & ~filters.COMMAND, self.receive_generate_prompt)
                    ],
                },
                fallbacks=[CommandHandler('cancel', self.cancel_generate_image)],
                name="generate_image_conversation",
                persistent=True,
                allow_reentry=True
            ),
    
            # Handler for direct image generation without conversation
            CommandHandler('generate_image', self.handle_generate_image),
    
            # You can add more handlers here
        ]