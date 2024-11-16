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

logger = logging.getLogger(__name__)

class CommandHandlers:
    def __init__(self, user_data_manager: UserDataManager):
        self.user_data_manager = user_data_manager

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /start command"""
        if not update.effective_user:
            return
            
        user_id = update.effective_user.id
        welcome_message = (
            "👋 Welcome to GemBot! I'm your AI assistant powered by Gemini.\n\n"
            "I can help you with:\n"
            "🤖 General conversations\n"
            "📝 Code assistance\n"
            "🖼️ Image analysis\n\n"
            "Feel free to start chatting or use /help to learn more!"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("Help 📚", callback_data='help'),
                InlineKeyboardButton("Settings ⚙️", callback_data='settings')
            ],
            [InlineKeyboardButton("Support Channel 📢", url='https://t.me/Gemini_AIAssistBot')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
        await self.user_data_manager.initialize_user(user_id)
        logger.info(f"New user started the bot: {user_id}")

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command"""
        if not update.message:
            return
            
        help_text = (
            "🤖 *Available Commands*\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/reset - Reset conversation history\n"
            "/settings - Configure bot settings\n\n"
            "💡 *Features*\n"
            "• Send text messages for general conversation\n"
            "• Send images for analysis\n"
            "• Use /code for code-related questions\n"
            "• Supports markdown formatting\n\n"
            "Need more help? Join our support channel!"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /reset command"""
        if not update.effective_user:
            return
            
        user_id = update.effective_user.id
        await self.user_data_manager.reset_user_data(user_id)
        await update.message.reply_text("✨ Conversation history cleared! Let's start fresh.")
        logger.info(f"User {user_id} reset their conversation history")

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /settings command"""
        if not update.effective_user or not update.message:
            return
            
        user_id = update.effective_user.id
        settings = await self.user_data_manager.get_user_settings(user_id)
        
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

    def register_handlers(self, application: Application) -> None:
        """Register all command handlers with the application"""
        try:
            # Register command handlers
            application.add_handler(CommandHandler("start", self.start))
            application.add_handler(CommandHandler("help", self.help))
            application.add_handler(CommandHandler("reset", self.reset))
            application.add_handler(CommandHandler("settings", self.settings))
            
            # Register callback query handler
            application.add_handler(CallbackQueryHandler(self.handle_callback_query))

            logger.info("Command handlers registered successfully")
        except Exception as e:
            logger.error(f"Failed to register command handlers: {str(e)}")
            raise
