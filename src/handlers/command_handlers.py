import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, CallbackContext
from services.user_data_manager import UserDataManager
from config import WELCOME_MESSAGE, HELP_MESSAGE

logger = logging.getLogger(__name__)
class CommandHandler:
    def __init__(self, user_manager: UserDataManager = None):
        self.user_manager = user_manager or UserDataManager()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        logger.info(f"Received /start command from user {user_id}")
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
        logger.info(f"Sent welcome message to user {user_id}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        logger.info(f"Received /help command from user {user_id}")
        await update.message.reply_text(
            HELP_MESSAGE,
            parse_mode='HTML'
        )
        logger.info(f"Sent help message to user {user_id}")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        logger.info(f"Received /settings command from user {user_id}")

        user_settings = self.user_manager.get_user_settings(user_id)
        keyboard = [
            [
                InlineKeyboardButton(
                    f"{'🔵' if user_settings['markdown_enabled'] else '⚪'} Markdown Mode",
                    callback_data='toggle_markdown'
                )
            ],
            [
                InlineKeyboardButton(
                    f"{'🔵' if user_settings['code_suggestions'] else '⚪'} Code Suggestions",
                    callback_data='toggle_code_suggestions'
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "⚙️ Bot Settings\nCustomize your interaction preferences:",
            reply_markup=reply_markup
        )
        logger.info(f"Sent settings menu to user {user_id}")

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        logger.info(f"Received /reset command from user {user_id}")

        self.user_manager.reset_user_data(user_id)
        await update.message.reply_text("Your data has been reset. All settings are back to default.")
        logger.info(f"Reset data for user {user_id}")

    async def feedback_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        logger.info(f"Received /feedback command from user {user_id}")

        await update.message.reply_text(
            "We value your feedback! Please send your comments or suggestions in the next message.",
            reply_markup=InlineKeyboardMarkup([[
                InlineKeyboardButton("Cancel", callback_data='cancel_feedback')
            ]])
        )
        context.user_data['awaiting_feedback'] = True
        logger.info(f"Waiting for feedback from user {user_id}")

    async def handle_feedback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if context.user_data.get('awaiting_feedback'):
            feedback = update.message.text
            logger.info(f"Received feedback from user {user_id}: {feedback[:50]}...")  # Log first 50 chars of feedback
            # Here you would typically store or send the feedback somewhere
            await update.message.reply_text("Thank you for your feedback! We appreciate your input.")
            del context.user_data['awaiting_feedback']
        else:
            logger.warning(f"Unexpected feedback message from user {user_id}")
# Create a single instance of CommandHandler
command_handler = CommandHandler()

# Export individual command functions
start_command = command_handler.start_command
help_command = command_handler.help_command
settings_command = command_handler.settings_command
reset_command = command_handler.reset_command
feedback_command = command_handler.feedback_command
handle_feedback = command_handler.handle_feedback
