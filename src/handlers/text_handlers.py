import logging
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes, MessageHandler, filters
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from utils.telegramlog import telegram_logger
from telegramify_markdown import customize, convert  # Import the customization options from the library
from typing import List

logger = logging.getLogger(__name__)

class TextHandler:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager):
        """ Initialize the TextHandler class with dependencies for API and user data management. """
        self.logger = logging.getLogger(__name__)
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.max_context_length = 5  # Limit user context to improve speed
        self.max_retries = 2  # Retry API calls on failure
        
        # Configure telegramify-markdown settings
        customize.strict_markdown = False  # Allow more markdown features
        customize.cite_expandable = True  # Enable expandable citations
        customize.latex_escape = True  # Enable LaTeX escaping

        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO,
        )

    async def format_telegram_markdown(self, text: str) -> str:
        """ Format text to be compatible with Telegram's MarkdownV2 format using telegramify-markdown. """
        try:
            formatted_text = convert(text)
            return formatted_text
        except Exception as e:
            self.logger.error(f"Error formatting markdown: {str(e)}")
            return text.replace('*', '').replace('_', '').replace('`', '')

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        """ Split long messages into smaller chunks while preserving code blocks and formatting. """
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
        """ Handle incoming text messages with AI-powered responses. """
        user_id = update.effective_user.id
        message_text = update.message.text
        
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        try:
            response = await self.gemini_api.generate_response(
                prompt=message_text,
                context=self.user_data_manager.get_user_context(user_id)[-self.max_context_length:]
            )

            # Split long messages
            message_chunks = await self.split_long_message(response)

            for chunk in message_chunks:
                try:
                    # Format with telegramify-markdown
                    formatted_chunk = await self.format_telegram_markdown(chunk)
                    await update.message.reply_text(
                        formatted_chunk,
                        parse_mode='MarkdownV2',
                        disable_web_page_preview=True
                    )
                except Exception as formatting_error:
                    self.logger.error(f"Formatting failed: {str(formatting_error)}")
                    await update.message.reply_text(chunk.replace('*', '').replace('_', '').replace('`', ''), parse_mode=None)

            telegram_logger.log_message(f"Text response sent successfully", user_id)

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error\\. Please try again\\.",
                parse_mode='MarkdownV2'
            )

    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """ Handle incoming image messages and analyze them using AI. """
        user_id = update.effective_user.id
        telegram_logger.log_message("Processing an image", user_id)

        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        try:
            photo = update.message.photo[-1]
            image_file = await context.bot.get_file(photo.file_id)
            image_bytes = await image_file.download_as_bytearray()

            caption = update.message.caption or "Please analyze this image and describe it."
            
            response = await self.gemini_api.analyze_image(image_bytes, caption)

            try:
                formatted_response = await self.format_telegram_markdown(response)
                await update.message.reply_text(
                    formatted_response,
                    parse_mode='MarkdownV2',
                    disable_web_page_preview=True
                )
            except Exception as markdown_error:
                self.logger.warning(f"Markdown formatting failed: {markdown_error}")
                await update.message.reply_text(response.replace('\\', ''), parse_mode=None)

            telegram_logger.log_message(f"Image analysis completed: {response}", user_id)

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            await update.message.reply_text(
                "Sorry, I couldn't process your image. Please try a different one or ensure it's in JPEG/PNG format."
            )

    def get_handlers(self):
        """ Return the list of message handlers for this bot. """
        return [
           MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message),
           MessageHandler(filters.PHOTO, self.handle_image),
       ]