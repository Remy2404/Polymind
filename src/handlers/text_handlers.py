from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, filters
from telegram.constants import ChatAction
from utils.telegramlog import telegram_logger
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from typing import List
import logging

class TextHandler:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager):
        self.logger = logging.getLogger(__name__)
        self.gemini_api = gemini_api
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

            # Send initial "thinking" message
            thinking_message = await update.message.reply_text("Thinking...")
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

            # Delete thinking message
            await thinking_message.delete()

            last_message = None
            for i, chunk in enumerate(message_chunks):
                try:
                    # Format with telegramify-markdown
                    formatted_chunk = await self.format_telegram_markdown(chunk)
                    if i == 0:
                        last_message = await update.message.reply_text(
                            formatted_chunk,
                            parse_mode='MarkdownV2',
                            disable_web_page_preview=True,
                        )
                    else:
                        last_message = await context.bot.edit_message_text(
                            chat_id=update.effective_chat.id,
                            message_id=last_message.message_id,
                            text=formatted_chunk,
                            parse_mode='MarkdownV2',
                            disable_web_page_preview=True,
                        )
                except Exception as formatting_error:
                    self.logger.error(f"Formatting failed: {str(formatting_error)}")
                    if i == 0:
                        last_message = await update.message.reply_text(
                            chunk.replace('*', '').replace('_', '').replace('`', ''),
                            parse_mode=None
                        )
                    else:
                        last_message = await context.bot.edit_message_text(
                            chat_id=update.effective_chat.id,
                            message_id=last_message.message_id,
                            text=chunk.replace('*', '').replace('_', '').replace('`', ''),
                            parse_mode=None
                        )

            # Update user context
            self.user_data_manager.add_to_context(user_id, {"role": "user", "content": message_text})
            self.user_data_manager.add_to_context(user_id, {"role": "assistant", "content": response})

            telegram_logger.log_message(f"Text response sent successfully", user_id)

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error\\. Please try again\\.",
                parse_mode='MarkdownV2'
            )
        else:
            self.logger.error("message processing failed")
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        telegram_logger.log_message("Processing an image", user_id)
    
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
                        disable_web_page_preview=True
                    )
                except Exception as markdown_error:
                    self.logger.warning(f"Markdown formatting failed: {markdown_error}")
                    await update.message.reply_text(response, parse_mode=None)
    
                # Update user stats for image
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, image=True)
    
                telegram_logger.log_message(f"Image analysis completed: {response}", user_id)
            else:
                await update.message.reply_text("Sorry, I couldn't analyze the image. Please try again.")
    
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            await update.message.reply_text(
                "Sorry, I couldn't process your image. Please try a different one or ensure it's in JPEG/PNG format."
            )    
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

    def get_handlers(self):
        return [
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message),
            MessageHandler(filters.PHOTO, self.handle_image),
        ]