
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

        # Check if this is a confirmation response
        if context.user_data.get("awaiting_reset_confirmation"):
            # User confirmed the reset
            context.user_data["awaiting_reset_confirmation"] = False
            await self._perform_memory_cleanup(update, context, user_id)
            return

        # Show confirmation dialog
        keyboard = [
            [
                InlineKeyboardButton(
                    "✅ Yes, Reset Everything", callback_data="confirm_reset"
                ),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_reset"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        warning_message = (
            "⚠️ **Memory Reset Warning**\n\n"
            "This will permanently delete:\n"
            "• All conversation history\n"
            "• Chat context and memory\n"
            "• Cached responses\n\n"
            "Your personal information (name, preferences) will be preserved.\n\n"
            "Are you sure you want to reset your conversation memory?"
        )

        await update.effective_message.reply_text(
            warning_message, reply_markup=reply_markup, parse_mode="Markdown"
        )

        # Set flag to await confirmation
        context.user_data["awaiting_reset_confirmation"] = True

    async def _perform_memory_cleanup(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int
    ) -> None:
        """Perform the actual memory cleanup after confirmation."""
        # Get personal info before reset
        personal_info = await self.user_data_manager.get_user_personal_info(user_id)

        # Complete memory cleanup using all managers
        try:
            # 1. Reset conversation history in UserDataManager
            self.user_data_manager.reset_conversation(user_id)

            # 2. Clear memory from MemoryManager and ConversationManager
            # Import conversation_manager if not already available
            from src.services.memory_context.conversation_manager import (
                ConversationManager,
            )
            from src.services.memory_context.memory_manager import MemoryManager
            from src.services.memory_context.model_history_manager import (
                ModelHistoryManager,
            )

            # Initialize memory managers for cleanup
            memory_manager = MemoryManager(db=self.user_data_manager.db)
            model_history_manager = ModelHistoryManager(memory_manager)
            conversation_manager = ConversationManager(
                memory_manager, model_history_manager
            )

            # Clear conversation memory
            await conversation_manager.reset_conversation(user_id)

            # 3. Clear any cached data
            if (
                hasattr(self.user_data_manager, "user_data_cache")
                and user_id in self.user_data_manager.user_data_cache
            ):
                del self.user_data_manager.user_data_cache[user_id]

            success_message = "✅ Complete memory cleanup successful!"
            if personal_info and "name" in personal_info:
                await update.effective_message.reply_text(
                    f"{success_message}\n\nConversation history has been reset, {personal_info['name']}! I'll still remember your personal details."
                )
            else:
                await update.effective_message.reply_text(
                    f"{success_message}\n\nConversation history has been reset! I'll still remember your personal details."
                )

        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
            await update.effective_message.reply_text(
                "❌ There was an error resetting your conversation history. Please try again."
            )

    async def handle_reset_confirmation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, callback_data: str
    ) -> None:
        """Handle reset confirmation callback queries."""
        query = update.callback_query
        user_id = update.effective_user.id

        if callback_data == "confirm_reset":
            # User confirmed, perform cleanup
            await query.edit_message_text("🔄 Performing memory cleanup...")
            await self._perform_memory_cleanup(update, context, user_id)

        elif callback_data == "cancel_reset":
            # User cancelled
            await query.edit_message_text(
                "❌ Memory reset cancelled. Your conversation history is safe."
            )

        # Clear the confirmation flag
        context.user_data["awaiting_reset_confirmation"] = False
