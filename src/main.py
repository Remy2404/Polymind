import logging
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
from typing import Optional

from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
import handlers.text_handlers as text_handlers
from utils.telegramlog import telegram_logger

# Load environment variables from .env file
load_dotenv()

class TelegramBot:
    def __init__(self):
        # Initialize logger
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        
        # Get tokens from .env file
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')  # Changed to match your .env variable name
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in .env file")

        # Verify Gemini API key is present
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize components
        self.gemini_api = GeminiAPI()
        self.user_data_manager = UserDataManager()
        self.telegram_logger = telegram_logger

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        user = update.effective_user
        welcome_message = (
            f"Hi {user.first_name}! 👋\n\n"
            "I'm your AI assistant powered by Gemini. I can help you with:\n"
            "• Answering questions\n"
            "• General conversation\n"
            "• Analysis and explanations\n\n"
            "Just send me a message and I'll do my best to help!"
        )
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        help_text = (
            "Here's how to use me:\n\n"
            "1. Simply send me any message or question\n"
            "2. I'll process it and respond accordingly\n"
            "3. For images, send them with a description\n\n"
            "Commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/reset - Reset our conversation"
        )
        await update.message.reply_text(help_text)

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reset the conversation history when /reset is issued."""
        user_id = update.effective_user.id
        self.user_data_manager.clear_history(user_id)
        await update.message.reply_text("Conversation history has been reset!")

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming text messages."""
        try:
            user_id = update.effective_user.id
            self.telegram_logger.log_message(user_id, f"Received text message: {update.message.text}")
            
            # Create text handler instance
            text_handler = text_handlers.TextHandler(self.gemini_api, self.user_data_manager)
            
            # Process the message
            await text_handler.handle_text_message(update, context)
            
        except Exception as e:
            self.logger.error(f"Error processing text message: {str(e)}")
            await self._error_handler(update, context)

    async def _error_handler(self, update: Optional[Update], context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors occurring in the dispatcher."""
        self.logger.error(f"Update {update} caused error {context.error}")
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred while processing your request."
            )

    def run(self) -> None:
        """Start the bot."""
        try:
            # Create application
            application = Application.builder().token(self.token).build()

            # Add handlers
            application.add_handler(CommandHandler("start", self.start))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("reset", self.reset_command))
            application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message))
            
            # Add error handler
            application.add_error_handler(self._error_handler)

            # Start the bot
            self.logger.info("Starting bot...")
            application.run_polling(allowed_updates=Update.ALL_TYPES)
            
        except Exception as e:
            self.logger.error(f"Failed to start bot: {str(e)}")
            raise

def main() -> None:
    """Main function to run the bot."""
    try:
        bot = TelegramBot()
        bot.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == '__main__':
    main()
