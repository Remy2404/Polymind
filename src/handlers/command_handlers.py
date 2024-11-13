import logging
import sys
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import WELCOME_MESSAGE, HELP_MESSAGE
from src.services.user_data_manager import UserDataManager
from src.utils.telegramlog import telegram_logger

logger = logging.getLogger(__name__)

class CommandHandler:
    def __init__(self, user_manager: UserDataManager = None):
        self.user_manager = user_manager or UserDataManager()
        telegram_logger.log_message(0, "CommandHandler initialized")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        telegram_logger.log_command(user_id, "/start")
        
        keyboard = [
            [
                InlineKeyboardButton("Help 📚", callback_data='help'),
                InlineKeyboardButton("Settings ⚙️", callback_data='settings')
            ],
            [InlineKeyboardButton("Support Channel 📢", url='https://t.me/Gemini_AIAssistBot')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            WELCOME_MESSAGE,
            reply_markup=reply_markup,
            parse_mode='HTML'
        )
        telegram_logger.log_message(user_id, "Welcome message sent")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        telegram_logger.log_command(user_id, "/help")
        
        await update.message.reply_text(
            HELP_MESSAGE,
            parse_mode='HTML'
        )
        telegram_logger.log_message(user_id, "Help message sent")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        telegram_logger.log_command(user_id, "/settings")

        user_settings = self.user_manager.get_user_settings(user_id)
        keyboard = [
            [
                InlineKeyboardButton(
                    f"{'🔵' if user_settings.get('markdown_enabled', True) else '⚪'} Markdown Mode",
                    callback_data='toggle_markdown'
                )
            ],
            [
                InlineKeyboardButton(
                    f"{'🔵' if user_settings.get('code_suggestions', True) else '⚪'} Code Suggestions",
                    callback_data='toggle_code_suggestions'
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "⚙️ Bot Settings\nCustomize your interaction preferences:",
            reply_markup=reply_markup
        )
        telegram_logger.log_message(user_id, "Settings menu displayed")

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        telegram_logger.log_command(user_id, "/reset")
        
        self.user_manager.reset_user_data(user_id)
        await update.message.reply_text("Your conversation history has been reset.")
        telegram_logger.log_message(user_id, "User data reset")

    async def feedback_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        telegram_logger.log_command(user_id, "/feedback")

        await update.message.reply_text(
            "We value your feedback! Please send your comments or suggestions in the next message.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Cancel", callback_data='cancel_feedback')
            ]])
        )
        context.user_data['awaiting_feedback'] = True
        telegram_logger.log_message(user_id, "Feedback prompt sent")

    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if context.user_data.get('awaiting_feedback'):
            feedback = update.message.text
            telegram_logger.log_message(user_id, f"Feedback received: {feedback[:50]}...")
            await update.message.reply_text("Thank you for your feedback! We appreciate your input.")
            del context.user_data['awaiting_feedback']
        else:
            telegram_logger.log_message(user_id, "Unexpected feedback message")

    def get_handlers(self):
        """Return all command handlers"""
        return [
            ('start', self.start_command),
            ('help', self.help_command),
            ('settings', self.settings_command),
            ('reset', self.reset_command),
            ('feedback', self.feedback_command)
        ]

# Create handler instance
def create_command_handler(user_manager: UserDataManager = None):
    return CommandHandler(user_manager)

