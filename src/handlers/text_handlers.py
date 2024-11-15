import logging
from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, filters
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from utils.telegramlog import telegram_logger

logger = logging.getLogger(__name__)

class TextHandler:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager

    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = update.effective_user.id
            message_text = update.message.text
            
            # Get conversation history from user data manager if needed
            conversation_history = self.user_data_manager.get_user_history(user_id)
            
            # Generate response using Gemini API
            response = await self.gemini_api.generate_response(
                prompt=message_text,
                context=conversation_history
            )
            
            # Update conversation history
            
            # Send response back to user
            await update.message.reply_text(response)
            
        except Exception as e:
            error_message = f"Error processing message: {str(e)}"
            await update.message.reply_text("I apologize, but I encountered an error processing your message. Please try again.")
            raise

    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        telegram_logger.log_message(f"Processing image", user_id)
        
        async with context.application.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing"):
            try:
                photo = update.message.photo[-1]
                image_file = await context.bot.get_file(photo.file_id)
                image_bytes = await image_file.download_as_bytearray()
                
                caption = update.message.caption or "Describe this image"
                response = await self.gemini_api.analyze_image(image_bytes, caption)
                
                await update.message.reply_text(response)
                telegram_logger.log_message(f"Image analysis sent", user_id)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                await update.message.reply_text("I'm analyzing your image. Please try again in a moment.")

    def get_handlers(self):
        return [
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message),
            MessageHandler(filters.PHOTO, self.handle_image)
        ]