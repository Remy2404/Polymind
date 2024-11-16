import logging
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes, MessageHandler, filters
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from utils.telegramlog import telegram_logger
from telegram import Update
from typing import List

logger = logging.getLogger(__name__)


class TextHandler:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager):
        """
        Initialize the TextHandler class with dependencies for API and user data management.
        """
        self.logger = logging.getLogger(__name__)
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.max_context_length = 5  # Limit user context to improve speed
        self.max_retries = 2  # Retry API calls on failure
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO,
        )

    async def format_telegram_markdown(self, text: str) -> str:
        """
        Format text to be compatible with Telegram's MarkdownV2 format.
        """
        try:
            # Characters that need to be escaped in MarkdownV2
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
            
            # Split into code blocks and regular text
            parts = text.split('```')
            formatted_parts = []
            
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Regular text
                    # First escape all special characters
                    escaped_part = part
                    for char in special_chars:
                        escaped_part = escaped_part.replace(char, f'\\{char}')
                    
                    # Handle bold text
                    escaped_part = escaped_part.replace('\\*\\*', '*')  # Unescape double asterisks
                    escaped_part = escaped_part.replace('**', '*')      # Convert to single asterisk
                    
                    # Handle italic text
                    escaped_part = escaped_part.replace('\\_\\_', '_')  # Unescape double underscores
                    escaped_part = escaped_part.replace('__', '_')      # Convert to single underscore
                    
                    formatted_parts.append(escaped_part)
                else:  # Code block
                    # Extract language identifier if present
                    if '\n' in part:
                        lang, code = part.split('\n', 1)
                        formatted_parts.append(f'```{lang}\n{code}```')
                    else:
                        formatted_parts.append(f'```{part}```')

            formatted_text = ''.join(formatted_parts)

            # Handle inline code
            inline_code_parts = formatted_text.split('`')
            for i in range(1, len(inline_code_parts) - 1, 2):
                # Remove escapes inside inline code
                inline_code_parts[i] = inline_code_parts[i].replace('\\', '')
            formatted_text = '`'.join(inline_code_parts)

            # Clean up any double escapes
            for char in special_chars:
                formatted_text = formatted_text.replace(f'\\\\{char}', f'\\{char}')

            return formatted_text

        except Exception as e:
            self.logger.error(f"Error formatting markdown: {str(e)}")
            return text.replace('*', '').replace('_', '').replace('`', '')

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        """
        Split long messages into smaller chunks while preserving code blocks and formatting.
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        in_code_block = False
        
        lines = text.split('\n')
        
        for line in lines:
            # Check for code block markers
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
            
            # If adding this line would exceed the limit
            if len(current_chunk) + len(line) + 1 > max_length:
                if in_code_block:
                    # Close code block in current chunk and reopen in next chunk
                    current_chunk += "\n```"
                    chunks.append(current_chunk)
                    current_chunk = "```\n" + line
                else:
                    chunks.append(current_chunk)
                    current_chunk = line
            else:
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle incoming text messages with AI-powered responses.
        """
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
                    # Format with HTML
                    formatted_chunk = await self.format_code_block(chunk)
                    await update.message.reply_text(
                        formatted_chunk,
                        parse_mode='HTML',
                        disable_web_page_preview=True
                    )
                except Exception as html_error:
                    self.logger.warning(f"HTML formatting failed: {html_error}")
                    # Fallback to plain text with monospace for code
                    try:
                        # Fallback to MarkdownV2
                        formatted_chunk = await self.format_telegram_markdown(chunk)
                        await update.message.reply_text(
                            formatted_chunk,
                            parse_mode='MarkdownV2',
                            disable_web_page_preview=True
                        )
                    except Exception as markdown_error:
                        # Final fallback: plain text
                        plain_chunk = chunk.replace('```', '').replace('`', '')
                        await update.message.reply_text(
                            plain_chunk,
                            parse_mode=None
                        )

            telegram_logger.log_message(f"Text response sent successfully", user_id)

        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error processing your request\\. Please try again later\\.",
                parse_mode='MarkdownV2'
            )

    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle incoming image messages and analyze them using AI.
        """
        user_id = update.effective_user.id
        telegram_logger.log_message("Processing an image", user_id)

        # Send typing action
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

        try:
            # Retrieve the largest photo version
            photo = update.message.photo[-1]
            image_file = await context.bot.get_file(photo.file_id)
            image_bytes = await image_file.download_as_bytearray()

            # Use caption or provide a default prompt
            caption = update.message.caption or "Please analyze this image and describe it."

            # Analyze the image using Gemini API
            response = await self.gemini_api.analyze_image(image_bytes, caption)

            # Attempt to send the response with Markdown formatting
            try:
                await update.message.reply_text(
                    response,
                    parse_mode='MarkdownV2',
                    disable_web_page_preview=True
                )
            except Exception as markdown_error:
                self.logger.warning(f"Markdown formatting failed: {markdown_error}")
                await update.message.reply_text(response.replace('\\', ''), parse_mode=None)

            # Log the successful handling of the image
            telegram_logger.log_message(f"Image analysis completed: {response}", user_id)

        except Exception as e:
            # Log the error and notify the user
            self.logger.error(f"Error processing image: {e}")
            await update.message.reply_text(
                "Sorry, I couldn't process your image. Please try a different one or ensure it's in JPEG/PNG format."
            )

    def get_handlers(self):
        """
        Return the list of message handlers for this bot.
        """
        return [
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message),
            MessageHandler(filters.PHOTO, self.handle_image),
        ]