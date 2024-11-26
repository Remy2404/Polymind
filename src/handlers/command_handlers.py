import os
import traceback
from typing import Optional, Dict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
    Application
)
from services.user_data_manager import UserDataManager
import logging
from services.gemini_api import GeminiAPI
from utils.telegramlog import telegram_logger

logger = logging.getLogger(__name__)

class CommandHandlers:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager):
        """
        Initialize the CommandHandler with Gemini API and User Data Manager.
        
        :param gemini_api: Instance of GeminiAPI for AI interactions
        :param user_data_manager: Instance of UserDataManager for user-related operations
        """
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.logger = logging.getLogger(__name__)
        self.telegram_logger = telegram_logger

    async def start_command(self, update: Update) -> None:
        """Handle the /start command"""
        if not update.effective_user:
            return
            
        user_id = update.effective_user.id
        welcome_message = (
            "👋 Welcome to GemBot! I'm your AI assistant powered by Gemini.\n\n"
            "I can help you with:\n"
            "🤖 General conversations\n"
            "📝 Code assistance\n"
            "voice to text conversion\n"
            "🖼️ Image analysis\n\n"

            "Feel free to start chatting or use /help to learn more!"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("Help 📚", callback_data='help_command'),
                InlineKeyboardButton("Settings ⚙️", callback_data='settings')
            ],
            [InlineKeyboardButton("Support Channel 📢", url='https://t.me/Gemini_AIAssistBot')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
        await self.user_data_manager.initialize_user(user_id)
        logger.info(f"New user started the bot: {user_id}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command"""
        if not update.message:
            return
            
        help_text = (
            "🤖 *Available Commands*\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/reset - Reset conversation history\n"
            "/settings - Configure bot settings\n\n"
            "/stats - Show bot statistics\n\n"
            "💡 *Features*\n"
            "• Send text messages for general conversation\n"
            "• Send images for analysis\n"
            "• Supports markdown formatting\n\n"
            "Need more help? Join our support channel!"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reset the conversation history when /reset is issued."""
        user_id = update.effective_user.id
        self.user_data_manager.clear_history(user_id)
        await update.message.reply_text("Conversation history has been reset!")

    async def settings(self, update: Update) -> None:
        """Handle the /settings command"""
        if not update.effective_user or not update.message:
            return
            
        user_id = update.effective_user.id
        settings = self.user_data_manager.get_user_settings(user_id)  # Remove 'await' here
        
        keyboard = [
            [
                InlineKeyboardButton(
                    f"{'🔵' if settings.get('markdown_enabled', True) else '⚪'} Markdown Mode",
                    callback_data='toggle_markdown'
                )
            ],
            [
                InlineKeyboardButton(
                    f"{'🔵' if settings.get('code_suggestions', True) else '⚪'} Code Suggestions",
                    callback_data='toggle_code_suggestions'
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚙️ *Bot Settings*\nCustomize your interaction preferences:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        self.telegram_logger.log_message(user_id, "Opened settings menu")
        
    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries from inline keyboards"""
        if not update.callback_query or not update.effective_user:
            return
            
        query = update.callback_query
        user_id = update.effective_user.id
        
        if query.data == 'help':
            await self.help(update, context)
        elif query.data == 'settings':
            await self.settings(update, context)
        elif query.data.startswith('toggle_'):
            setting = query.data.replace('toggle_', '')
            await self.user_data_manager.toggle_setting(user_id, setting)
            await self.settings(update, context)
        
        await query.answer()

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            user_id = update.effective_user.id
            stats = self.user_data_manager.get_user_statistics(user_id)
            if stats:
                await update.message.reply_text(
                    f"Here are your stats:\n"
                    f"• Total Messages Sent: {stats.get('total_messages', 0)}\n"
                    f"• Text Messages: {stats.get('text_messages', 0)}\n"
                    f"• Voice Messages: {stats.get('voice_messages', 0)}\n"
                    f"• Images Sent: {stats.get('images', 0)}"
                )
            else:
                await update.message.reply_text("No statistics available yet.")
        except Exception as e:
            self.logger.error(f"Error fetching user stats: {str(e)}")
            await self._error_handler(update, context)

    async def broadcast_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Broadcast a message to all users (admin only)."""
        try:
            admin_user_id = int(os.getenv('ADMIN_USER_ID', '0'))
            if update.effective_user.id != admin_user_id:
                await update.message.reply_text("You are not authorized to use this command.")
                return

            if not context.args:
                await update.message.reply_text("Please provide a message to broadcast.")
                return

            broadcast_message = ' '.join(context.args)
            all_users = self.user_data_manager.get_all_user_ids()
            successful_sends = 0
            for user_id in all_users:
                try:
                    await context.bot.send_message(chat_id=user_id, text=broadcast_message)
                    successful_sends += 1
                except Exception as e:
                    self.logger.error(f"Failed to send message to user {user_id}: {str(e)}")

            await update.message.reply_text(f"Broadcast message sent successfully to {successful_sends} users.")
        except Exception as e:
            self.logger.error(f"Error during broadcast: {str(e)}")
            await self._error_handler(update, context)

    async def test_api(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Test the Gemini API directly."""
        await update.message.reply_text("Testing Gemini API. Please wait...")
        
        try:
            response = await self.gemini_api.generate_response("Hello, can you give me a short joke?")
            if response:
                await update.message.reply_text(f"API Test Result:\n\n{response}")
            else:
                await update.message.reply_text("API test failed. No response received.")
        except Exception as e:
            await update.message.reply_text(f"API test failed with error: {str(e)}")


    def register_handlers(self, application: Application) -> None:
        """Register all command handlers with the application"""
        try:
            # Register command handlers
            application.add_handler(CommandHandler("test_api", self.test_api))
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("reset", self.reset_command))
            application.add_handler(CommandHandler("stats", self.stats_command))
            application.add_handler(CommandHandler("settings", self.settings))
            application.add_handler(CommandHandler("broadcast", self.broadcast_command))
            application.add_handler(CallbackQueryHandler(self.handle_callback_query))
            

            logger.info("Command handlers registered successfully")
        except Exception as e:
            logger.error(f"Failed to register command handlers: {str(e)}")
            raise
