from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler, Application ,CallbackContext
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from utils.telegramlog import telegram_logger
import logging

class CommandHandlers:
    def __init__(self, gemini_api: GeminiAPI, user_data_manager: UserDataManager):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.logger = logging.getLogger(__name__)
        self.telegram_logger = telegram_logger

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user:
            return
        
        user_id = update.effective_user.id
        welcome_message = (
            "👋 Welcome to GemBot! I'm your AI assistant powered by Gemini.\n\n"
            "I can help you with:\n"
            "🤖 General conversations\n"
            "📝 Code assistance\n"
            "🗣️ Voice to text conversion\n"
            "🖼️ Image analysis\n\n"
            "Feel free to start chatting or use /help to learn more!"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("Help 📚", callback_data='/help_command'),
                InlineKeyboardButton("Settings ⚙️", callback_data='/settings')
            ],
            [InlineKeyboardButton("Support Channel 📢", url='https://t.me/Gemini_AIAssistBot')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
        await self.user_data_manager.initialize_user(user_id)
        self.logger.info(f"New user started the bot: {user_id}")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        help_text = (
            "🤖 *Available Commands*\n\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/reset - Reset conversation history\n"
            "/settings - Configure bot settings\n"
            "/stats - Show bot statistics\n\n"
            "💡 *Features*\n"
            "• Send text messages for general conversation\n"
            "• Send images for analysis\n"
            "• Supports markdown formatting\n\n"
            "Need more help? Join our support channel!"
        )
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        self.user_data_manager.reset_conversation(user_id)
        await update.message.reply_text("Conversation history has been reset!")

    async def settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        settings = self.user_data_manager.get_user_settings(user_id)
        
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
    async def handle_stats(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        user_data = self.user_data_manager.get_user_data(user_id)
        stats = user_data.get('stats', {})
        
        stats_message = (
            "📊 Your Bot Usage Statistics:\n\n"
            f"📝 Text Messages: {stats.get('messages', 0)}\n"
            f"🎤 Voice Messages: {stats.get('voice_messages', 0)}\n"
            f"🖼 Images Processed: {stats.get('images', 0)}\n"
            f"📑 PDFs Analyzed: {stats.get('pdfs_processed', 0)}\n"
            f"Last Active: {stats.get('last_active', 'Never')}"
        )
        
        await update.message.reply_text(stats_message)
#Export Conversation History
    async def handle_export(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        user_data = self.user_data_manager.get_user_data(user_id)
        history = user_data.get('conversation_history', [])
        
        # Create formatted export
        export_text = "💬 Conversation History:\n\n"
        for msg in history:
            export_text += f"User: {msg['user']}\n"
            export_text += f"Bot: {msg['bot']}\n\n"
        
        # Send as file if too long
        if len(export_text) > 4000:
            with open(f'history_{user_id}.txt', 'w') as f:
                f.write(export_text)
            await update.message.reply_document(
                document=open(f'history_{user_id}.txt', 'rb'),
                filename='conversation_history.txt'
            )
        else:
            await update.message.reply_text(export_text)
    async def handle_preferences(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle user preferences command"""
        keyboard = [
            [InlineKeyboardButton("Language", callback_data="pref_language"),
            InlineKeyboardButton("Response Format", callback_data="pref_format")],
            [InlineKeyboardButton("Notifications", callback_data="pref_notifications"),
            InlineKeyboardButton("AI Model", callback_data="pref_model")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚙️ *User Preferences*\n\n"
            "Select a setting to modify:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )


    async def handle_callback_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()
        
        if query.data == 'help_command':
            await self.help_command(update, context)
        elif query.data == 'settings':
            await self.settings(update, context)
        elif query.data in ['toggle_markdown', 'toggle_code_suggestions']:
            # Implement toggle logic here
            await query.edit_message_text("Setting updated!")
    
    async def semantic_search_command(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        query = ' '.join(context.args)
        if not query:
            await update.message.reply_text("Please provide a search query. Usage: /search <query>")
            return
        results = await self.pdf_handler.semantic_search(user_id, query)
        response = "🔍 *Search Results:*\n" + "\n".join(results)
        await update.message.reply_text(response, parse_mode='Markdown')

    async def summary_command(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        length = context.args[0] if context.args else 'brief'
        if length not in ['brief', 'detailed']:
            await update.message.reply_text("Invalid summary length. Use 'brief' or 'detailed'.")
            return
        summary = await self.pdf_handler.generate_summary(user_id, length)
        await update.message.reply_text(f"📝 *{length.capitalize()} Summary:*\n{summary}", parse_mode='Markdown')
    
    async def keypoints_command(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        keypoints = await self.pdf_handler.extract_key_points(user_id)
        response = "📌 *Key Points:*\n" + "\n".join(keypoints)
        await update.message.reply_text(response, parse_mode='Markdown')

    def register_handlers(self, application: Application) -> None:
        try:
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("reset", self.reset_command))
            application.add_handler(CommandHandler("settings", self.settings))
            application.add_handler(CommandHandler("stats", self.handle_stats))
            application.add_handler(CommandHandler("export", self.handle_export))
            application.add_handler(CommandHandler("summary", self.summary_command))
            application.add_handler(CommandHandler("keypoints", self.keypoints_command))
            application.add_handler(CommandHandler("preferences", self.handle_preferences))
            application.add_handler(CommandHandler("search", self.semantic_search_command))
            application.add_handler(CallbackQueryHandler(self.handle_callback_query))
            
            self.logger.info("Command handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register command handlers: {str(e)}")
            raise
