# src/handlers/command_handlers.py

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import ContextTypes, CommandHandler, CallbackQueryHandler, Application
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from utils.telegramlog import telegram_logger
import logging
import re
from io import BytesIO
import requests
from services.flux_lora_img import flux_lora_image_generator

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
                InlineKeyboardButton("Help 📚", callback_data='help_command'),
                InlineKeyboardButton("Settings ⚙️", callback_data='settings')
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
            "/generate_image <prompt> - Generate an image from text\n"
            "/export - Export conversation history\n"
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

    async def handle_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

    async def handle_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

        data = query.data

        if data == 'help_command':
            await self.help_command(update, context)
        elif data == 'settings':
            await self.settings(update, context)
        elif data in ['toggle_markdown', 'toggle_code_suggestions']:
            # Implement toggle logic here
            user_id = update.effective_user.id
            if data == 'toggle_markdown':
                current_value = self.user_data_manager.get_user_settings(user_id).get('markdown_enabled', True)
                self.user_data_manager.set_user_setting(user_id, 'markdown_enabled', not current_value)
                status = 'enabled' if not current_value else 'disabled'
                await query.edit_message_text(f"Markdown Mode has been {status}.")
            elif data == 'toggle_code_suggestions':
                current_value = self.user_data_manager.get_user_settings(user_id).get('code_suggestions', True)
                self.user_data_manager.set_user_setting(user_id, 'code_suggestions', not current_value)
                status = 'enabled' if not current_value else 'disabled'
                await query.edit_message_text(f"Code Suggestions have been {status}.")
        elif data.startswith('img_'):
            await self.handle_image_settings(update, context, data)
        elif data.startswith('pref_'):
            await self.handle_user_preferences(update, context, data)
        else:
            await query.edit_message_text("Unknown action.")

    async def generate_image_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        prompt = ' '.join(context.args)
        if not prompt:
            await update.message.reply_text("Please provide a prompt. Usage: /generate_image <your prompt>")
            return

        # Store prompt in user_data for later use
        context.user_data['image_prompt'] = prompt

        # Create inline keyboard with combined quality and generation settings
        keyboard = [
            [
                InlineKeyboardButton("Standard (256x256) - Quick Generation (20 steps)", callback_data="img_256_steps_20"),
                InlineKeyboardButton("Standard (256x256) - Detailed Generation (50 steps)", callback_data="img_256_steps_50")
            ],
            [
                InlineKeyboardButton("HD (512x512) - Quick Generation (20 steps)", callback_data="img_512_steps_20"),
                InlineKeyboardButton("HD (512x512) - Detailed Generation (50 steps)", callback_data="img_512_steps_50")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "Choose image quality and generation settings:",
            reply_markup=reply_markup
        )

    async def handle_image_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str) -> None:
        """
        Handle image generation settings based on user selection.

        Args:
            update (Update): Telegram update.
            context (ContextTypes.DEFAULT_TYPE): Context.
            data (str): Callback data.
        """
        # Example data format: "img_256_steps_20"
        match = re.match(r'img_(\d+)_steps_(\d+)', data)
        if not match:
            await update.callback_query.edit_message_text("Invalid selection. Please try again with /generate_image.")
            return

        width = int(match.group(1))
        height = int(match.group(1))  # Assuming square images
        steps = int(match.group(2))

        # Remove the inline buttons by editing the message text
        await update.callback_query.edit_message_text("🖌️ Generating image, please wait..")

        try:
            images = await flux_lora_image_generator.text_to_image(
                prompt=context.user_data.get('image_prompt', ''),
                num_images=1,
                num_inference_steps=steps,
                width=width,
                height=height
            )

            if not images:
                await update.callback_query.edit_message_text("Failed to generate image. Please try again.")
                return

            for idx, image in enumerate(images):
                with BytesIO() as output:
                    image.save(output, format="PNG")
                    output.seek(0)
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=InputFile(output),
                        caption=f"Generated image ({width}x{height}, {steps} steps)"
                    )

            # Optionally, delete the "Generating..." message
            await update.callback_query.delete_message()

        except Exception as e:
            self.logger.error(f"Error generating image: {e}")
            await update.callback_query.edit_message_text("Sorry, I couldn't generate the image. Please try again later.")

    async def handle_user_preferences(self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str) -> None:
        # Implement preference handlers
        # Placeholder for now
        await update.callback_query.edit_message_text("Preference settings not implemented yet.")

    def register_handlers(self, application: Application) -> None:
        try:
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("reset", self.reset_command))
            application.add_handler(CommandHandler("settings", self.settings))
            application.add_handler(CommandHandler("stats", self.handle_stats))
            application.add_handler(CommandHandler("export", self.handle_export))
            application.add_handler(CommandHandler("preferences", self.handle_preferences))
            application.add_handler(CommandHandler("generate_image", self.generate_image_command))
            application.add_handler(CallbackQueryHandler(self.handle_callback_query))
            
            self.logger.info("Command handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register command handlers: {e}")
            raise