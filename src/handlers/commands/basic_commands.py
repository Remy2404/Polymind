import sys
import os
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import logging
class BasicCommands:
    def __init__(self, user_data_manager, telegram_logger):
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)
    async def start_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.effective_user:
            return
        user_id = update.effective_user.id
        welcome_message = (
            "👋 Welcome to DeepGem! I'm your AI assistant powered by Gemini-2.0-flash & Deepseek-R1 .\n\n"
            "I can help you with:\n"
            "🤖 General conversations\n"
            "📝 Code assistance\n"
            "🗣️ Voice to text conversion\n"
            "🖼️ Image generation and analysis\n"
            "📄 AI document generation\n"
            "📑 PDF analysis\n"
            "📊 Statistics tracking\n\n"
            "Available commands:\n"
            "/genimg - Generate images with Together AI\n"
            "/gendoc - Generate documents with Multi-Modal AI\n"
            "/export - Export conversation history first message convert it to docx files\n"
            "/reset - Reset conversation history\n"
            "/switchmodel - Switch between AI models\n\n"
            "Feel free to start chatting or use /help to learn more!"
        )
        keyboard = [
            [
                InlineKeyboardButton("Help 📚", callback_data="help"),
                InlineKeyboardButton("Settings ⚙️", callback_data="settings"),
            ],
            [InlineKeyboardButton("Support Channel 📢", url="https://t.me/GemBotAI")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        if update.effective_message:
            await update.effective_message.reply_text(
                welcome_message, reply_markup=reply_markup
            )
        await self.user_data_manager.initialize_user(user_id)
        self.logger.info(f"New user started the bot: {user_id}")
    async def help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        help_text = (
            "🤖 Available Commands\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/reset - Reset conversation history\n"
            "/settings - Configure bot settings\n"
            "/stats - Show bot statistics\n"
            "/genimg - Generate images with Together AI\n"
            "/switchmodel - Switch between AI models\n"
            "/export - Export conversation history\n\n"
            "💡 Features\n"
            "• General conversations with AI\n"
            "• Code assistance\n"
            "• Voice to text conversion\n"
            "• Image generation and analysis\n"
            "• Statistics tracking\n"
            "• Supports markdown formatting\n\n"
            "Need help? Join our support channel @GemBotAI!"
        )
        if update.effective_message:
            await update.effective_message.reply_text(help_text)
        if update.effective_user:
            self.telegram_logger.log_message(
                update.effective_user.id, "Help command requested"
            )
    async def reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if not update.effective_user:
            return
        user_id = update.effective_user.id
        personal_info = await self.user_data_manager.get_user_personal_info(user_id)
        self.user_data_manager.reset_conversation(user_id)
        if update.effective_message:
            if personal_info and "name" in personal_info:
                await update.effective_message.reply_text(
                    f"Conversation history has been reset, {personal_info['name']}! I'll still remember your personal details."
                )
            else:
                await update.effective_message.reply_text(
                    "Conversation history has been reset!"
                )
