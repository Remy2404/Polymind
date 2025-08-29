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
            "👋 **Welcome to DeepGem!** I'm your AI assistant powered by Gemini-2.0-flash & Deepseek-R1.\n\n"
            "**🎯 I can help you with:**\n"
            "🤖 General conversations\n"
            "📝 Code assistance\n"
            "🗣️ Voice to text conversion\n"
            "🖼️ Image generation and analysis\n"
            "📄 AI document generation\n"
            "📑 PDF analysis\n"
            "📊 Statistics tracking\n"
            "🔍 Real-time web search (NEW!)\n"
            "🧠 Advanced problem solving (NEW!)\n\n"
            "**🆕 Enhanced MCP Tools:**\n"
            "I now have access to powerful tools with full transparency:\n"
            "• Web search via Exa AI\n"
            "• Library documentation search\n"
            "• Sequential thinking for complex problems\n"
            "• Document analysis\n"
            "• All tool calls are now visible!\n\n"
            "**🚀 Quick Start:**\n"
            "/genimg - Generate images\n"
            "/search - Web search\n"
            "/gendoc - Generate documents\n"
            "/mcp_help - Learn about advanced tools\n\n"
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

        # Use effective_message for reply
        if update.effective_message:
            await update.effective_message.reply_text(
                welcome_message, reply_markup=reply_markup, parse_mode="Markdown"
            )

        await self.user_data_manager.initialize_user(user_id)
        self.logger.info(f"New user started the bot: {user_id}")

    async def help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        help_text = (
            "🤖 **Available Commands**\n\n"
            "**🎯 Basic Commands:**\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/reset - Reset conversation history\n"
            "/settings - Configure bot settings\n"
            "/stats - Show bot statistics\n\n"
            "**🎨 AI Features:**\n"
            "/genimg - Generate images with Together AI\n"
            "/switchmodel - Switch between AI models\n"
            "/export - Export conversation history\n"
            "/gendoc - Generate AI documents\n\n"
            "**� MCP Tools (Advanced):**\n"
            "/search <query> - Web search via Exa AI\n"
            "/Context7 <query> - Search library docs\n"
            "/sequentialthinking <problem> - Step-by-step problem solving\n"
            "/Docfork <url> - Analyze documents\n"
            "/mcp_status - Show MCP server status\n"
            "/mcp_help - Detailed MCP help\n\n"
            "**�💡 Features:**\n"
            "• General conversations with AI\n"
            "• Code assistance with multiple models\n"
            "• Voice to text conversion\n"
            "• Image generation and analysis\n"
            "• Real-time web search and research\n"
            "• Document analysis and generation\n"
            "• Statistics tracking\n"
            "• Enhanced tool call visibility\n"
            "• Supports markdown formatting\n\n"
            "**🆕 Enhanced MCP Integration:**\n"
            "All MCP tool calls now show detailed execution logs, including:\n"
            "• Tool invocation details\n"
            "• Execution timing\n"
            "• Success/failure status\n"
            "• Source attribution\n\n"
            "Need help? Join our support channel @GemBotAI!"
        )
        # Use effective_message for reply
        if update.effective_message:
            await update.effective_message.reply_text(help_text, parse_mode="Markdown")
        # Only log if effective_user exists
        if update.effective_user:
            self.telegram_logger.log_message(
                update.effective_user.id, "Help command requested"
            )

    async def reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        # Only proceed if effective_user exists
        if not update.effective_user:
            return
        user_id = update.effective_user.id

        # Get personal info before resetting
        personal_info = await self.user_data_manager.get_user_personal_info(user_id)

        # Reset conversation history
        self.user_data_manager.reset_conversation(user_id)

        # If there was personal information, confirm we remember it
        if update.effective_message:
            if personal_info and "name" in personal_info:
                await update.effective_message.reply_text(
                    f"Conversation history has been reset, {personal_info['name']}! I'll still remember your personal details."
                )
            else:
                await update.effective_message.reply_text(
                    "Conversation history has been reset!"
                )
