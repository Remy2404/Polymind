import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.ext import CommandHandler, CallbackQueryHandler


logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE, user_data_manager):
    """Handle the /start command"""
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
    user_data_manager.initialize_user(user_id)
    logger.info(f"New user started the bot: {user_id}")

async def help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /help command"""
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

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE, user_data_manager):
    """Handle the /reset command"""
    user_id = update.effective_user.id
    user_data_manager.reset_user_data(user_id)
    await update.message.reply_text("✨ Conversation history cleared! Let's start fresh.")
    logger.info(f"User {user_id} reset their conversation history")

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE, user_data_manager):
    """Handle the /settings command"""
    user_id = update.effective_user.id
    settings = user_data_manager.get_user_settings(user_id)
    
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

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE, user_data_manager):
    """Handle callback queries from inline keyboards"""
    query = update.callback_query
    user_id = update.effective_user.id
    
    if query.data == 'help':
        await help(update, context)
    elif query.data == 'settings':
        await settings(update, context, user_data_manager)
    elif query.data.startswith('toggle_'):
        setting = query.data.replace('toggle_', '')
        user_data_manager.toggle_setting(user_id, setting)
        await settings(update, context, user_data_manager)
    
    await query.answer()

def register_handlers(application, user_data_manager):
    """Register all command handlers"""
    application.add_handler(CommandHandler("start", lambda u, c: start(u, c, user_data_manager)))
    application.add_handler(CommandHandler("help", help))
    application.add_handler(CommandHandler("reset", lambda u, c: reset(u, c, user_data_manager)))
    application.add_handler(CommandHandler("settings", lambda u, c: settings(u, c, user_data_manager)))
    application.add_handler(CallbackQueryHandler(lambda u, c: handle_callback_query(u, c, user_data_manager)))
