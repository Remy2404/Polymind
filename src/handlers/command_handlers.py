import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from click import prompt
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    InlineQueryResultArticle,
    InputTextMessageContent,
)
from telegram.ext import (
    ContextTypes,
    CommandHandler,
    CallbackQueryHandler,
    Application,
    InlineQueryHandler,
)
from services.user_data_manager import UserDataManager
from services.gemini_api import GeminiAPI
from utils.telegramlog import TelegramLogger as telegram_logger
import logging
import re
from services.flux_lora_img import FluxLoraImageGenerator as flux_lora_image_generator
import time, asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cachetools import TTLCache
from typing import Optional
from telegram.constants import ChatAction
from services.text_to_video import text_to_video_generator
from PIL import Image
import io


@dataclass
class ImageRequest:
    prompt: str
    width: int
    height: int
    steps: int
    timestamp: float = field(default_factory=time.time)


class ImageGenerationHandler:
    def __init__(self):
        self.request_cache = TTLCache(maxsize=100, ttl=3600)
        self.request_limiter = {}
        self.processing_queue = asyncio.Queue()
        self.rate_limit_time = 30

    def is_rate_limited(self, user_id: int) -> bool:
        if user_id in self.request_limiter:
            last_request = self.request_limiter[user_id]
            if datetime.now() - last_request < timedelta(seconds=self.rate_limit_time):
                return True
        return False

    def update_rate_limit(self, user_id: int) -> None:
        self.request_limiter[user_id] = datetime.now()

    def get_cached_image(
        self, prompt: str, width: int, height: int, steps: int
    ) -> Optional[Image.Image]:
        cache_key = f"{prompt}_{width}_{height}_{steps}"
        return self.request_cache.get(cache_key)

    def cache_image(
        self, prompt: str, width: int, height: int, steps: int, image: Image.Image
    ) -> None:
        cache_key = f"{prompt}_{width}_{height}_{steps}"
        self.request_cache[cache_key] = image


class CommandHandlers:
    def __init__(
        self,
        gemini_api: GeminiAPI,
        user_data_manager: UserDataManager,
        telegram_logger: telegram_logger,
        flux_lora_image_generator: flux_lora_image_generator,
    ):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.flux_lora_image_generator = flux_lora_image_generator
        self.logger = logging.getLogger(__name__)
        self.telegram_logger = telegram_logger
        self.image_handler = ImageGenerationHandler()

    # --- AI Document Generation Methods --- START ---
    async def generate_ai_document_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate an AI-authored document in PDF or DOCX format"""
        user_id = update.effective_user.id
        self.telegram_logger.log_message(
            f"AI document generation requested by user {user_id}", user_id
        )

        # Check if a prompt was provided
        if not context.args:
            # Show document type selection if no arguments provided
            document_types = [
                [
                    InlineKeyboardButton(
                        "📝 Article", callback_data="aidoc_type_article"
                    ),
                    InlineKeyboardButton(
                        "📊 Report", callback_data="aidoc_type_report"
                    ),
                ],
                [
                    InlineKeyboardButton("📋 Guide", callback_data="aidoc_type_guide"),
                    InlineKeyboardButton(
                        "📑 Summary", callback_data="aidoc_type_summary"
                    ),
                ],
                [
                    InlineKeyboardButton("📚 Essay", callback_data="aidoc_type_essay"),
                    InlineKeyboardButton(
                        "📈 Analysis", callback_data="aidoc_type_analysis"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "💼 Proposal", callback_data="aidoc_type_proposal"
                    )
                ],
            ]
            reply_markup = InlineKeyboardMarkup(document_types)

            await update.message.reply_text(
                "🤖 *AI Document Generator*\n\n"
                "I can create professional documents on any topic. First, select the type of document you'd like me to create:",
                reply_markup=reply_markup,
                parse_mode="Markdown",
            )
            return

        # If prompt was provided, store it and ask for format selection
        prompt = " ".join(context.args)
        context.user_data["aidoc_prompt"] = prompt
        context.user_data["aidoc_type"] = "article"  # Default type

        # Ask for format
        await self._show_ai_document_format_selection(update, context)

    async def _show_ai_document_format_selection(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show document format selection buttons"""
        format_options = [
            [
                InlineKeyboardButton("📄 PDF Format", callback_data="aidoc_format_pdf"),
                InlineKeyboardButton(
                    "📝 DOCX Format", callback_data="aidoc_format_docx"
                ),
            ],
            [
                InlineKeyboardButton(
                    "🧠 Use Gemini", callback_data="aidoc_model_gemini"
                ),
                InlineKeyboardButton(
                    "🔍 Use DeepSeek", callback_data="aidoc_model_deepseek"
                ),
            ],
            [
                InlineKeyboardButton(
                    "🌀 Use Optimus Alpha", callback_data="aidoc_model_Optimus_Alpha"
                )
            ],
        ]
        reply_markup = InlineKeyboardMarkup(format_options)

        # Get current model and prompt from context
        model = context.user_data.get("aidoc_model", "gemini")
        doc_type = context.user_data.get("aidoc_type", "article")
        prompt = context.user_data.get("aidoc_prompt", "")

        # Ensure model name reflects the actual models available
        model_name = (
            "Gemini-2.5-Pro"
            if model == "gemini"
            else "DeepSeek 70B" if model == "deepseek" else "Optimus Alpha"
        )

        message = (
            f"🤖 *AI Document Generator*\n\n"
            f"I'll create a {doc_type} about:\n"
            f'_"{prompt}"_\n\n'
            f"Current model: *{model_name}*\n\n"
            f"Please select the output format and AI model:"
        )

        if update.callback_query:
            await update.callback_query.edit_message_text(
                message, reply_markup=reply_markup, parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                message, reply_markup=reply_markup, parse_mode="Markdown"
            )

    async def handle_ai_document_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        """Handle callbacks for AI document generation"""
        query = update.callback_query  # Get query object
        if not query:
            self.logger.warning(
                "handle_ai_document_callback received update without callback_query"
            )
            return
        await query.answer()  # Answer callback query early

        if data.startswith("aidoc_type_"):
            # Store document type preference
            doc_type = data.replace("aidoc_type_", "")
            context.user_data["aidoc_type"] = doc_type

            # Ask for document topic
            await query.edit_message_text(  # Use query object
                f"📝 You selected: *{doc_type.capitalize()}*\n\n"
                f"Now, please send me the topic or subject for your {doc_type}.",
                parse_mode="Markdown",
            )

            # Set state to await topic input
            context.user_data["awaiting_aidoc_topic"] = True

        elif data.startswith("aidoc_format_"):
            # Get the selected format
            output_format = data.replace("aidoc_format_", "")
            context.user_data["aidoc_format"] = output_format

            # Generate the document
            await self._generate_and_send_ai_document(update, context)  # Pass update

        elif data.startswith("aidoc_model_"):
            # Update the model selection
            new_model = data.replace("aidoc_model_", "")
            current_model = context.user_data.get("aidoc_model", "gemini")

            if new_model == current_model:
                # Model hasn't changed, just acknowledge
                model_name = (
                    "Gemini-2.5-Pro" if current_model == "gemini" else "DeepSeek 70B"
                )
                await query.answer(
                    f"{model_name} is already selected."
                )  # Use query.answer for brief notification
                return  # Stop further processing

            # Model has changed, update user data and the message
            context.user_data["aidoc_model"] = new_model
            # Update the message to show selected model
            await self._show_ai_document_format_selection(
                update, context
            )  # Pass update

    async def _generate_and_send_ai_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate AI document and send it to user"""
        query = update.callback_query  # Get query object for editing message
        if not query:
            self.logger.warning(
                "_generate_and_send_ai_document received update without callback_query"
            )
            # Attempt to send message directly if query is missing
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text="Starting document generation..."
            )
        else:
            # Show generating message using query
            await query.edit_message_text(
                f"🤖 *Generating your {context.user_data.get('aidoc_type', 'article')}*\n\n"
                f"I'm creating a {context.user_data.get('aidoc_format', 'pdf').upper()} document about:\n"
                f"_\"{context.user_data.get('aidoc_prompt', '')}\"_\n\n"
                f"This may take a minute or two. Please wait...",
                parse_mode="Markdown",
            )

        user_id = update.effective_user.id

        # Get all required parameters
        prompt = context.user_data.get("aidoc_prompt", "")
        doc_type = context.user_data.get("aidoc_type", "article")
        output_format = context.user_data.get("aidoc_format", "pdf")
        model = context.user_data.get("aidoc_model", "gemini")

        if not prompt:
            error_message = "Error: No document topic provided. Please try again with /gendoc command."
            if query:
                await query.edit_message_text(error_message)
            else:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id, text=error_message
                )
            return

        try:
            # Import AI document generator (ensure path is correct)
            from utils.document_ai_generator import AIDocumentGenerator

            ai_doc_generator = AIDocumentGenerator(self.gemini_api)

            # Generate document
            document_bytes, title = await ai_doc_generator.generate_ai_document(
                prompt=prompt,
                output_format=output_format,
                document_type=doc_type,
                model=model,
                max_tokens=4000,  # Consider making this configurable
            )

            # Send document to user
            with io.BytesIO(document_bytes) as doc_io:
                # Set filename based on format and title
                sanitized_title = (
                    "".join(c for c in title if c.isalnum() or c in " _-")
                    .strip()
                    .replace(" ", "_")
                )
                doc_io.name = f"{sanitized_title}_{datetime.now().strftime('%Y%m%d')}.{output_format}"

                # Format the caption to escape ALL Markdown special characters
                model_display_name = model.capitalize()
                if model == "quasar_alpha":
                    model_display_name = "Optimus Alpha"

                # Make sure to escape ALL special characters for MarkdownV2
                special_chars = [
                    "_",
                    "*",
                    "[",
                    "]",
                    "(",
                    ")",
                    "~",
                    "`",
                    ">",
                    "#",
                    "+",
                    "-",
                    "=",
                    "|",
                    "{",
                    "}",
                    ".",
                    "!",
                ]
                safe_title = title
                for char in special_chars:
                    safe_title = safe_title.replace(char, f"\\{char}")

                caption = (
                    f"📄 {safe_title}\n\nGenerated using {model_display_name} model"
                )

                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=doc_io,
                    filename=doc_io.name,
                    caption=caption,
                    parse_mode="MarkdownV2",
                )

            # Log success
            self.telegram_logger.log_message(
                f"AI document generated successfully in {output_format} format using {model} model",
                user_id,
            )

            # Clear the user data
            for key in [
                "aidoc_prompt",
                "aidoc_type",
                "aidoc_format",
                "aidoc_model",
                "awaiting_aidoc_topic",
            ]:  # Added awaiting_aidoc_topic
                if key in context.user_data:
                    del context.user_data[key]

        except Exception as e:
            self.logger.error(f"AI document generation error: {str(e)}")
            error_text = f"Sorry, there was an error generating your document: {str(e)}"
            # Try to edit the message first, then send a new one if it fails
            try:
                if query:
                    await query.edit_message_text(error_text)
                else:
                    # If query is not available, send a new message
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id, text=error_text
                    )
            except Exception as edit_error:
                self.logger.error(f"Failed to edit message or send error: {edit_error}")
                # Final fallback: send a generic error message
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Sorry, there was an error generating your document and sending the details.",
                )

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
            "🎬 Video generation\n"
            "📊 Statistics tracking\n\n"
            "Available commands:\n"
            "/generate_image - Create images from text\n"
            "/genvid - Generate videos from descriptions\n"
            "/genimg - Generate images with Together AI\n"
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

        if update.callback_query:
            await update.callback_query.message.reply_text(
                welcome_message, reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(welcome_message, reply_markup=reply_markup)

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
            "/generate_image - Create images from text\n"
            "/genvid - Generate videos from descriptions\n"
            "/genimg - Generate images with Together AI\n"
            "/switchmodel - Switch between AI models\n"
            "/export - Export conversation history\n\n"
            "💡 Features\n"
            "• General conversations with AI\n"
            "• Code assistance\n"
            "• Voice to text conversion\n"
            "• Image generation and analysis\n"
            "• Video generation\n"
            "• Statistics tracking\n"
            "• Supports markdown formatting\n\n"
            "Need help? Join our support channel @GemBotAI!"
        )
        if update.callback_query:
            await update.callback_query.message.reply_text(help_text)
        else:
            await update.message.reply_text(help_text)
            await update.callback_query.message.reply_text(help_text)
        self.telegram_logger.log_message(
            update.effective_user.id, "Help command requested"
        )

    async def reset_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_id = update.effective_user.id

        # Get personal info before resetting
        personal_info = await self.user_data_manager.get_user_personal_info(user_id)

        # Reset conversation history
        await self.user_data_manager.reset_conversation(user_id)

        # If there was personal information, confirm we remember it
        if personal_info and "name" in personal_info:
            await update.message.reply_text(
                f"Conversation history has been reset, {personal_info['name']}! I'll still remember your personal details."
            )
        else:
            await update.message.reply_text("Conversation history has been reset!")

    async def settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        try:
            user_id = update.effective_user.id
            self.telegram_logger.log_message(
                f"User {user_id} accessed settings", user_id
            )

            # Initialize user data if not already done
            await self.user_data_manager.initialize_user(user_id)

            # Get user settings with proper error handling and default values
            try:
                settings = await self.user_data_manager.get_user_settings(user_id)
                if not settings or not isinstance(settings, dict):
                    # If settings are not available or not a dictionary, use defaults
                    settings = {"markdown_enabled": True, "code_suggestions": True}
            except Exception as settings_error:
                self.logger.warning(
                    f"Error fetching settings for user {user_id}: {settings_error}"
                )
                settings = {"markdown_enabled": True, "code_suggestions": True}

            keyboard = [
                [
                    InlineKeyboardButton(
                        f"{'🔵' if settings.get('markdown_enabled', True) else '⚪'} Markdown Mode",
                        callback_data="toggle_markdown",
                    )
                ],
                [
                    InlineKeyboardButton(
                        f"{'🔵' if settings.get('code_suggestions', True) else '⚪'} Code Suggestions",
                        callback_data="toggle_code_suggestions",
                    )
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            settings_text = "⚙️ *Bot Settings*\nCustomize your interaction preferences:"

            # Different handling for callback_query vs direct command
            if update.callback_query:
                # This is called from a button click
                try:
                    await update.callback_query.message.reply_text(
                        settings_text, reply_markup=reply_markup, parse_mode="Markdown"
                    )
                except Exception as reply_error:
                    self.logger.error(f"Error replying to callback: {reply_error}")
                    # Try without parse_mode as fallback
                    await update.callback_query.message.reply_text(
                        settings_text.replace("*", ""), reply_markup=reply_markup
                    )
            else:
                # This is called directly from /settings command
                try:
                    await update.message.reply_text(
                        settings_text, reply_markup=reply_markup, parse_mode="Markdown"
                    )
                except Exception as reply_error:
                    self.logger.error(f"Error replying to message: {reply_error}")
                    # Try without parse_mode as fallback
                    await update.message.reply_text(
                        settings_text.replace("*", ""), reply_markup=reply_markup
                    )

            self.telegram_logger.log_message("Opened settings menu", user_id)

        except Exception as e:
            user_id = update.effective_user.id if update.effective_user else "Unknown"
            self.logger.error(f"Settings error for user {user_id}: {str(e)}")

            error_message = "An error occurred while processing your request. Please try again later."

            try:
                if update.callback_query:
                    await update.callback_query.message.reply_text(error_message)
                else:
                    await update.message.reply_text(error_message)
            except Exception as reply_error:
                self.logger.error(f"Failed to send error message: {reply_error}")

            self.telegram_logger.log_error(f"Settings error: {str(e)}", user_id)

    async def handle_stats(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_id = update.effective_user.id
        user_data = await self.user_data_manager.get_user_data(user_id)
        stats = user_data.get("stats", {})

        stats_message = (
            "📊 Your Bot Usage Statistics:\n\n"
            f"📝 Text Messages: {stats.get('messages', 0)}\n"
            f"🎤 Voice Messages: {stats.get('voice_messages', 0)}\n"
            f"🖼 Images Processed: {stats.get('images', 0)}\n"
            f"📑 PDFs Analyzed: {stats.get('pdfs_processed', 0)}\n"
            f"Last Active: {stats.get('last_active', 'Never')}"
        )

        await update.message.reply_text(stats_message)

    async def handle_export(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_id = update.effective_user.id
        user_data = self.user_data_manager.get_user_data(user_id)
        history = user_data.get("conversation_history", [])

        # Create formatted export
        export_text = "💬 Conversation History:\n\n"
        for msg in history:
            export_text += f"User: {msg['user']}\n"
            export_text += f"Bot: {msg['bot']}\n\n"

        # Send as file if too long
        if len(export_text) > 4000:
            with open(f"history_{user_id}.txt", "w") as f:
                f.write(export_text)
            await update.message.reply_document(
                document=open(f"history_{user_id}.txt", "rb"),
                filename="conversation_history.txt",
            )
        else:
            await update.message.reply_text(export_text)

    async def handle_preferences(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle user preferences command"""
        keyboard = [
            [
                InlineKeyboardButton("Language", callback_data="pref_language"),
                InlineKeyboardButton("Response Format", callback_data="pref_format"),
            ],
            [
                InlineKeyboardButton(
                    "Notifications", callback_data="pref_notifications"
                ),
                InlineKeyboardButton("AI Model", callback_data="pref_model"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "⚙️ *User Preferences*\n\n" "Select a setting to modify:",
            reply_markup=reply_markup,
            parse_mode="Markdown",
        )

    async def handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await query.answer()

        data = query.data
        self.logger.info(f"Handling callback query with data: {data}")

        if data == "help":  # Changed from 'help_command' to match button callback
            await self.help_command(update, context)
        elif data == "preferences":  # Add handler for preferences button
            await self.handle_preferences(update, context)
        elif data == "settings":
            await self.settings(update, context)
        elif data in ["toggle_markdown", "toggle_code_suggestions"]:
            # Implement toggle logic here
            user_id = update.effective_user.id
            if data == "toggle_markdown":
                current_value = await self.user_data_manager.get_user_settings(
                    user_id
                ).get("markdown_enabled", True)
                self.user_data_manager.set_user_setting(
                    user_id, "markdown_enabled", not current_value
                )
                status = "enabled" if not current_value else "disabled"
                await query.edit_message_text(f"Markdown Mode has been {status}.")
            elif data == "toggle_code_suggestions":
                current_value = await self.user_data_manager.get_user_settings(
                    user_id
                ).get("code_suggestions", True)
                await self.user_data_manager.set_user_setting(
                    user_id, "code_suggestions", not current_value
                )
                status = "enabled" if not current_value else "disabled"
                await query.edit_message_text(f"Code Suggestions have been {status}.")

        # Document export handlers
        elif data == "export_conversation":
            await self.handle_export_conversation(update, context)
        elif data == "export_custom":
            await query.edit_message_text(
                "Please send the text you want to convert to a document. You can include markdown formatting."
            )
            context.user_data["awaiting_doc_text"] = True
        elif data == "export_cancel":
            await query.edit_message_text("Document export cancelled.")
        elif data.startswith("export_format_"):
            document_format = data.replace("export_format_", "")
            await self.generate_document(update, context, document_format)

        # AI Document generation handlers
        elif data.startswith("aidoc_type_"):
            await self.handle_ai_document_callback(update, context, data)
        elif data.startswith("aidoc_format_"):
            await self.handle_ai_document_callback(update, context, data)
        elif data.startswith("aidoc_model_"):
            await self.handle_ai_document_callback(update, context, data)

        elif data.startswith("img_"):
            await self.handle_image_settings(update, context, data)
        elif data.startswith("pref_"):
            await self.handle_user_preferences(update, context, data)
        else:
            # Add logging to see which callback data is causing issues
            self.logger.warning(f"Unhandled callback data: {data}")
            await query.edit_message_text(f"Unknown action: {data}. Please try again.")

    async def generate_image_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_id = update.effective_user.id

        if self.image_handler.is_rate_limited(user_id):
            remaining_time = self.image_handler.rate_limit_time
            await update.message.reply_text(
                f"Please wait {remaining_time} seconds before generating another image."
            )
            return

        prompt = " ".join(context.args)
        if not prompt:
            await update.message.reply_text(
                "Please provide a prompt. Usage: /generate_image <your prompt>"
            )
            return

        if len(prompt) > 500:
            await update.message.reply_text(
                "Prompt is too long. Please limit to 500 characters."
            )
            return

        # Store the prompt in user_data for later use
        context.user_data["image_prompt"] = prompt

        # Create a preview message with confirmation buttons
        preview_message = (
            f"📝 Your image generation prompt:\n\n"
            f"'{prompt}'\n\n"
            f"Do you want to proceed with this prompt?"
        )

        keyboard = [
            [
                InlineKeyboardButton(
                    "✅ Confirm", callback_data="confirm_image_prompt"
                ),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_image_prompt"),
            ],
            [InlineKeyboardButton("✏️ Edit Prompt", callback_data="edit_image_prompt")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(preview_message, reply_markup=reply_markup)

    async def handle_image_prompt_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await query.answer()

        if query.data == "confirm_image_prompt":
            # Proceed with image generation
            await self.show_image_quality_options(update, context)
        elif query.data == "cancel_image_prompt":
            await query.edit_message_text("Image generation cancelled.")
        elif query.data == "edit_image_prompt":
            await query.edit_message_text(
                "Please send your updated prompt. You can cancel by sending /cancel."
            )
            context.user_data["awaiting_prompt_edit"] = True

    async def show_image_quality_options(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        keyboard = [
            [
                InlineKeyboardButton(
                    "📱 Standard - Quick (20 steps)", callback_data="img_256_steps_20"
                )
            ],
            [
                InlineKeyboardButton(
                    "📱 Standard - Detailed (50 steps)",
                    callback_data="img_256_steps_50",
                )
            ],
            [
                InlineKeyboardButton(
                    "🖥️ HD - Quick (20 steps)", callback_data="img_512_steps_20"
                )
            ],
            [
                InlineKeyboardButton(
                    "🖥️ HD - Detailed (50 steps)", callback_data="img_512_steps_50"
                )
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.callback_query.edit_message_text(
            "Choose image quality and generation settings:", reply_markup=reply_markup
        )

    async def handle_image_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await query.answer()

        start_time = time.time()
        data = query.data

        match = re.match(r"img_(\d+)_steps_(\d+)", data)
        if not match:
            await query.edit_message_text(
                "Invalid selection. Please try again with /generate_image."
            )
            return

        width = height = int(match.group(1))
        steps = int(match.group(2))
        prompt = context.user_data.get("image_prompt", "")

        if not prompt:
            await query.edit_message_text(
                "No image prompt found. Please use /generate_image command first."
            )
            return

        cached_image = self.image_handler.get_cached_image(prompt, width, height, steps)
        if cached_image:
            await self._send_image(
                update,
                context,
                cached_image,
                width,
                height,
                steps,
                "Retrieved from cache",
            )
            return

        progress_message = await query.edit_message_text(
            "🖌️ Generating image...\n\n" "⏳ Initializing..."
        )

        try:
            progress_task = asyncio.create_task(
                self._update_progress(progress_message, steps)
            )

            images = await self.flux_lora_image_generator.text_to_image(
                prompt=prompt,
                num_images=1,
                num_inference_steps=steps,
                width=width,
                height=height,
            )

            progress_task.cancel()

            if not images:
                await progress_message.edit_text(
                    "Failed to generate image. Please try again."
                )
                return

            self.image_handler.cache_image(prompt, width, height, steps, images[0])

            generation_time = time.time() - start_time
            await self._send_image(
                update,
                context,
                images[0],
                width,
                height,
                steps,
                f"Generated in {generation_time:.1f}s",
            )

            await progress_message.delete()

        except asyncio.CancelledError:
            await progress_message.edit_message_text("Image generation was cancelled.")
        except Exception as e:
            self.logger.error(f"Error generating image: {e}")
            await progress_message.edit_message_text(
                "Sorry, I couldn't generate the image. Please try again later."
            )

    async def _update_progress(self, message, total_steps: int) -> None:
        try:
            for step in range(1, total_steps + 1):
                await asyncio.sleep(0.5)
                progress = step / total_steps * 100
                await message.edit_text(
                    f"🖌️ Generating image...\n\n"
                    f"Progress: {progress:.0f}%\n"
                    f"Step {step}/{total_steps}"
                )
        except asyncio.CancelledError:
            pass

    async def _send_image(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        image: Image.Image,
        width: int,
        height: int,
        steps: int,
        status: str,
    ) -> None:
        with io.BytesIO() as output:
            if image.mode != "RGB":
                image = image.convert("RGB")

            image.save(output, format="JPEG", optimize=True)
            output.seek(0)

            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=InputFile(output),
                caption=f"Generated image ({width}x{height}, {steps} steps)\n{status}",
            )

    async def handle_user_preferences(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        # Implement preference handlers
        # Placeholder for now
        await update.callback_query.edit_message_text(
            "Preference settings not implemented yet."
        )

    async def generate_image_advanced(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /imagen3 command for advanced image generation."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message("Imagen 3 image generation requested", user_id)

        if not context.args:
            await update.message.reply_text(
                "Please provide a description for the image you want to generate.\n"
                "Example: `/imagen3 a surreal landscape with floating islands and waterfalls`",
                parse_mode="Markdown",
            )
            return

        # Join all arguments to form the prompt
        prompt = " ".join(context.args)

        # Send a status message
        status_message = await update.message.reply_text(
            "Generating image with AI... This may take a moment."
        )

        try:
            # Use the correct method name: text_to_image instead of generate_images
            images = await self.flux_lora_image_generator.text_to_image(
                prompt=prompt,
                num_images=1,
                num_inference_steps=30,  # Higher quality setting
                width=768,
                height=768,
                guidance_scale=7.5,
            )

            if images and len(images) > 0:
                # Delete the status message
                await status_message.delete()

                # Convert PIL Image to bytes
                with io.BytesIO() as output:
                    images[0].save(output, format="PNG")
                    output.seek(0)
                    image_bytes = output.getvalue()

                # Send the generated image
                await update.message.reply_photo(
                    photo=image_bytes,
                    caption=f"Generated image based on: '{prompt}'",
                    parse_mode="Markdown",
                )

                # Update user stats
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, image_generation=True)
            else:
                await status_message.edit_text(
                    "Sorry, I couldn't generate that image. Please try a different description or try again later."
                )
        except Exception as e:
            self.logger.error(f"Image generation error: {str(e)}")
            await status_message.edit_text(
                "Sorry, there was an error generating your image. Please try a different description."
            )

    async def show_document_history(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show the user's document processing history."""
        user_id = update.effective_user.id

        if (
            "document_history" not in context.user_data
            or not context.user_data["document_history"]
        ):
            await update.message.reply_text("You haven't processed any documents yet.")
            return

        history_text = "Your document history:\n\n"

        for idx, doc in enumerate(context.user_data["document_history"]):
            timestamp = datetime.datetime.fromisoformat(doc["timestamp"])
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")

            history_text += f"{idx+1}. {doc['file_name']} ({formatted_time})\n"
            history_text += f"   Prompt: {doc['prompt']}\n\n"

        await update.message.reply_text(history_text)

    async def generate_video_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /generate_video command for text-to-video generation."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message("Video generation requested", user_id)

        if not context.args:
            await update.message.reply_text(
                "Please provide a description for the video you want to generate.\n"
                "Example: `/generate_video an astronaut dancing on the moon, detailed, 4k`",
                parse_mode="Markdown",
            )
            return

        # Join all arguments to form the prompt
        prompt = " ".join(context.args)

        # Send a status message
        status_message = await update.message.reply_text(
            "🎬 Generating video from your description... This may take several minutes."
        )

        # Send typing action to indicate processing
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_VIDEO
        )

        try:
            # Generate the video
            video_bytes = await text_to_video_generator.generate_video(
                prompt=prompt,
                num_frames=24,  # reasonable default
                height=256,
                width=256,
                num_inference_steps=30,
            )

            # Delete status message
            await status_message.delete()

            if video_bytes:
                # Send the video
                with io.BytesIO(video_bytes) as video_io:
                    video_io.name = "generated_video.mp4"
                    await update.message.reply_video(
                        video=video_io,
                        caption=f"🎬 Generated video based on: '{prompt}'",
                        supports_streaming=True,
                    )

                # Update user stats if available
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, videos_generated=1)
            else:
                await update.message.reply_text(
                    "❌ Sorry, I couldn't generate the video. Please try a different description or try again later."
                )
        except Exception as e:
            self.logger.error(f"Video generation error: {str(e)}")
            await status_message.edit_text(
                "❌ Sorry, there was an error generating your video. The system might be busy or the request too complex."
            )

    # Add this method to your CommandHandlers class
    async def handle_inline_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle inline queries for video generation with @botname."""
        query = update.inline_query.query

        if not query:
            return

        results = [
            InlineQueryResultArticle(
                id=f"video_{hash(query)}",
                title="Generate a video",
                description=f"Create a video of: {query}",
                input_message_content=InputTextMessageContent(
                    f"🎬 Generating video: '{query}'\n\n(This may take several minutes...)"
                ),
                thumb_url="https://img.icons8.com/color/452/video.png",  # Optional video icon thumbnail
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "Cancel", callback_data=f"cancel_video_{hash(query)}"
                            )
                        ]
                    ]
                ),
            )
        ]

        await update.inline_query.answer(results, cache_time=300)

    # Add this method to handle what happens after a user selects the inline result
    async def handle_chosen_inline_result(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the result chosen from an inline query."""
        result_id = update.chosen_inline_result.result_id
        query = update.chosen_inline_result.query
        from_user = update.chosen_inline_result.from_user
        inline_message_id = update.chosen_inline_result.inline_message_id

        if result_id.startswith("video_"):
            # Start the video generation process
            try:
                # Generate video
                video_bytes = await text_to_video_generator.generate_video(
                    prompt=query,
                    num_frames=24,
                    height=256,
                    width=256,
                    num_inference_steps=30,
                )

                if video_bytes:
                    # We can't directly edit an inline message with a video, so we need to
                    # use a callback to notify the user the video is ready
                    await context.bot.edit_message_text(
                        text=f"✅ Video generated! Check your bot chat to view it.",
                        inline_message_id=inline_message_id,
                    )

                    # Send the video directly to the user in a private chat
                    with io.BytesIO(video_bytes) as video_io:
                        video_io.name = "generated_video.mp4"
                        await context.bot.send_video(
                            chat_id=from_user.id,
                            video=video_io,
                            caption=f"🎬 Generated video based on: '{query}'",
                            supports_streaming=True,
                        )

                    # Update user stats
                    if self.user_data_manager:
                        self.user_data_manager.update_stats(
                            from_user.id, videos_generated=1
                        )
                else:
                    await context.bot.edit_message_text(
                        text=f"❌ Failed to generate video for '{query}'.",
                        inline_message_id=inline_message_id,
                    )
            except Exception as e:
                self.logger.error(f"Inline video generation error: {str(e)}")
                await context.bot.edit_message_text(
                    text=f"❌ Error generating video: {str(e)}",
                    inline_message_id=inline_message_id,
                )

    # Add this to the CommandHandlers class in command_handlers.py

    async def generate_together_image(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /genimg command for image generation using Together AI."""
        user_id = update.effective_user.id
        self.telegram_logger.log_message(
            "Together AI image generation requested", user_id
        )

        if not context.args:
            await update.message.reply_text(
                "Please provide a description for the image you want to generate.\n"
                "Example: `/genimg a sunset over a calm lake with mountains in the background`",
                parse_mode="Markdown",
            )
            return

        # Join all arguments to form the prompt
        prompt = " ".join(context.args)

        # Send a status message
        status_message = await update.message.reply_text(
            "🎨 Generating image with Together AI... This may take a moment."
        )

        # Send typing action to indicate processing
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO
        )

        try:
            # Import the generator
            from services.together_ai_img import together_ai_image_generator

            # Generate the image
            image = await together_ai_image_generator.generate_image(
                prompt=prompt, num_steps=4, width=1024, height=1024
            )

            if image:
                # Delete the status message
                await status_message.delete()

                # Send the generated image
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    output.seek(0)

                    await update.message.reply_photo(
                        photo=output,
                        caption=f"🖼️ Generated image based on: '{prompt}'",
                        parse_mode="Markdown",
                    )

                # Update user stats if available
                if self.user_data_manager:
                    self.user_data_manager.update_stats(user_id, image_generation=True)
            else:
                await status_message.edit_text(
                    "❌ Sorry, I couldn't generate the image. Please try a different description or try again later."
                )
        except Exception as e:
            self.telegram_logger.log_error(f"Image generation error: {str(e)}", user_id)
            await status_message.edit_text(
                "❌ An error occurred while generating your image. Please try again later."
            )

    async def switch_model_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /switchmodel command to let users select their preferred LLM."""
        user_id = update.effective_user.id

        # Get the model registry from the application
        model_registry = None
        user_model_manager = None

        if hasattr(context.application, "bot_data"):
            model_registry = context.application.bot_data.get("model_registry")
            user_model_manager = context.application.bot_data.get("user_model_manager")

        # If we found the model registry, use it to build dynamic buttons
        if model_registry:
            # Get all available models
            available_models = model_registry.get_all_models()

            # Get current model from UserModelManager if available, otherwise fallback
            if user_model_manager:
                current_model = user_model_manager.get_user_model(user_id)
                current_model_config = model_registry.get_model_config(current_model)
                current_model_name = (
                    current_model_config.display_name
                    if current_model_config
                    else "Unknown"
                )
            else:
                # Fallback to user_data_manager preferences
                current_model = await self.user_data_manager.get_user_preference(
                    user_id, "preferred_model", default="gemini"
                )
                # Map model code to display name using registry if possible
                model_config = model_registry.get_model_config(current_model)
                current_model_name = (
                    model_config.display_name if model_config else "Unknown"
                )

            # Build dynamic keyboard from available models
            keyboard = []
            row = []

            # Group models in rows of 2
            for i, (model_id, model_config) in enumerate(available_models.items()):
                # Create button with emoji if available
                button_text = (
                    f"{model_config.indicator_emoji} {model_config.display_name}"
                )
                button = InlineKeyboardButton(
                    button_text, callback_data=f"model_{model_id}"
                )

                row.append(button)

                # Add row after every 2 buttons or at the end
                if (i + 1) % 2 == 0 or i == len(available_models) - 1:
                    keyboard.append(row)
                    row = []

        else:  # Fallback to hardcoded models if registry not available
            keyboard = [
                [
                    InlineKeyboardButton(
                        "gemini-2.5-Pro", callback_data="model_gemini"
                    ),
                    InlineKeyboardButton(
                        "DeepSeek 70B", callback_data="model_deepseek"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🌀 Optimus Alpha", callback_data="model_optimus_alpha"
                    ),
                    InlineKeyboardButton(
                        "🧑‍💻 DeepCoder", callback_data="model_deepcoder"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🦙 Llama-4 Maverick", callback_data="model_llama4_maverick"
                    )
                ],
            ]

            # Get current model using old method
            current_model = await self.user_data_manager.get_user_preference(
                user_id, "preferred_model", default="gemini"
            )

            # Map model code to display name
            model_names = {
                "gemini": "Gemini-2.5-Pro",
                "deepseek": "DeepSeek 70B",
                "optimus_alpha": "🌀 Optimus Alpha",  # Match button text/callback
                "deepcoder": "🧑‍💻 DeepCoder",  # Match button text/callback
                "llama4_maverick": "🦙 Llama-4 Maverick",  # Match button text/callback
            }
            # Use the model ID from preferences to get the display name
            current_model_name = model_names.get(
                current_model, "Unknown"
            )  # Default to Unknown if not found

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"🔄 Your current model is: *{current_model_name}*\n\n"
            "Choose the AI model you'd like to use for chat:",
            reply_markup=reply_markup,
            parse_mode="Markdown",
        )

    async def handle_model_selection(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle model selection callbacks."""
        query = update.callback_query
        user_id = query.from_user.id

        await query.answer()

        selected_model_from_callback = query.data.replace("model_", "")

        # --- Fix for model ID inconsistency ---
        # Convert the callback format (e.g., optimus_alpha) to the format expected by the factory/registry (e.g., optimus-alpha)
        if selected_model_from_callback == "optimus_alpha":
            selected_model_id_for_backend = "optimus-alpha"
        elif selected_model_from_callback == "llama4_maverick":
            # Ensure consistency if backend expects hyphenated version
            # Based on factory, it expects 'llama4_maverick' (underscore), so no change needed here usually.
            # However, let's keep the variable distinct for clarity.
            selected_model_id_for_backend = "llama4_maverick"
        else:
            # For other models like gemini, deepseek, deepcoder, the callback format matches the expected ID
            selected_model_id_for_backend = selected_model_from_callback
        # --- End of fix ---

        # Get model registry and user model manager if available
        model_registry = None
        user_model_manager = None

        if hasattr(context.application, "bot_data"):
            model_registry = context.application.bot_data.get("model_registry")
            user_model_manager = context.application.bot_data.get("user_model_manager")

        # Use the new UserModelManager if available
        model_switched = False
        model_name = (
            selected_model_id_for_backend  # Use the corrected ID for display fallback
        )

        if user_model_manager and model_registry:
            # Set model using UserModelManager with the CORRECTED ID
            model_switched = user_model_manager.set_user_model(
                user_id, selected_model_id_for_backend
            )

            # Get display name from model config using the CORRECTED ID
            model_config = model_registry.get_model_config(
                selected_model_id_for_backend
            )
            if model_config:
                model_name = model_config.display_name

            # Log the model change
            self.logger.info(
                f"User {user_id} switched to model: {selected_model_id_for_backend} ({model_name}) using UserModelManager"
            )
        else:
            # Fallback to legacy method
            model_switched = True
            # Get display name from hardcoded mapping using the ORIGINAL callback value for lookup
            fallback_model_names = {
                "gemini": "Gemini-2.5-Pro",
                "deepseek": "DeepSeek 70B",
                "optimus_alpha": "🌀 Optimus Alpha",  # Use underscore version from callback for lookup
                "deepcoder": "🧑‍💻 DeepCoder",
                "llama4_maverick": "🦙 Llama-4 Maverick",
            }
            model_name = fallback_model_names.get(
                selected_model_from_callback, selected_model_from_callback.capitalize()
            )

            # Save the CORRECTED ID to user preferences if using legacy
            await self.user_data_manager.set_user_preference(
                user_id, "preferred_model", selected_model_id_for_backend
            )

            # Log the model change
            self.logger.info(
                f"User {user_id} switched to model: {selected_model_id_for_backend} (display: {model_name}) using legacy method"
            )

        # Set flags in context.user_data using the CORRECTED ID
        context.user_data["just_switched_model"] = True
        context.user_data["switched_to_model"] = selected_model_id_for_backend
        context.user_data["model_switch_counter"] = 0  # Reset counter

        # Create a more descriptive success message
        if model_switched:
            # Get model features from registry if available
            features_text = ""
            if model_registry:
                model_config = model_registry.get_model_config(
                    selected_model_id_for_backend
                )
                if model_config:
                    capabilities = [cap.value for cap in model_config.capabilities]
                    features_text = f"\n\nFeatures: {', '.join(capabilities)}"

            await query.edit_message_text(
                f"✅ Model switched successfully!\n\nYou're now using *{model_name}*{features_text}\n\n"
                f"You can change this anytime with /switchmodel",
                parse_mode="Markdown",
            )
        else:
            await query.edit_message_text(
                f"❌ Error switching model. The selected model may not be available.",
                parse_mode="Markdown",
            )

    async def export_to_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate a professionally formatted PDF or DOCX document from text"""
        user_id = update.effective_user.id
        self.telegram_logger.log_message(
            f"Document export requested by user {user_id}", user_id
        )

        # Get format preference
        format_options = [
            [
                InlineKeyboardButton(
                    "📄 PDF Format", callback_data="export_format_pdf"
                ),
                InlineKeyboardButton(
                    "📝 DOCX Format", callback_data="export_format_docx"
                ),
            ]
        ]
        format_markup = InlineKeyboardMarkup(format_options)

        # Check if a prompt was provided
        if context.args:
            # Save the content to be converted
            context.user_data["doc_export_text"] = " ".join(context.args)
            await update.message.reply_text(
                "Please select the document format you want to export to:",
                reply_markup=format_markup,
            )
        else:
            # No content provided - offer to export conversation history
            export_options = [
                [
                    InlineKeyboardButton(
                        "📜 Export Conversation", callback_data="export_conversation"
                    ),
                    InlineKeyboardButton(
                        "✏️ Provide Custom Text", callback_data="export_custom"
                    ),
                ],
                [InlineKeyboardButton("❌ Cancel", callback_data="export_cancel")],
            ]
            export_markup = InlineKeyboardMarkup(export_options)

            await update.message.reply_text(
                "What would you like to export to a document?\n\n"
                "You can export your conversation history or provide custom text.",
                reply_markup=export_markup,
            )

    async def handle_export_conversation(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Export conversation history to a document"""
        user_id = update.effective_user.id

        # Get conversation history
        user_data = await self.user_data_manager.get_user_data(user_id)
        history = user_data.get("conversation_history", [])

        if not history:
            await update.callback_query.edit_message_text(
                "You don't have any conversation history to export."
            )
            return

        # Format the conversation for document export
        formatted_content = "# Conversation History\n\n"

        for i, msg in enumerate(history):
            # Add a section break for readability between exchanges
            if i > 0:
                formatted_content += "\n---\n\n"

            # Add the user message with markdown formatting
            formatted_content += f"## User\n\n{msg['user']}\n\n"

            # Add the bot response with markdown formatting
            formatted_content += f"## Assistant\n\n{msg['bot']}\n\n"

        # Store formatted content
        context.user_data["doc_export_text"] = formatted_content

        # Ask for format preference
        format_options = [
            [
                InlineKeyboardButton(
                    "📄 PDF Format", callback_data="export_format_pdf"
                ),
                InlineKeyboardButton(
                    "📝 DOCX Format", callback_data="export_format_docx"
                ),
            ]
        ]
        format_markup = InlineKeyboardMarkup(format_options)

        await update.callback_query.edit_message_text(
            "Please select the document format you want to export to:",
            reply_markup=format_markup,
        )

    async def generate_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, document_format: str
    ) -> None:
        """Generate and send document in the specified format"""
        user_id = update.effective_user.id

        if "doc_export_text" not in context.user_data:
            await update.callback_query.edit_message_text(
                "No content found for document generation. Please try again."
            )
            return

        content = context.user_data["doc_export_text"]

        # Send processing message
        await update.callback_query.edit_message_text(
            f"Generating {document_format.upper()} document... This may take a moment."
        )

    async def try_generate_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, document_format: str
    ) -> None:
        try:
            # Import AI document generator
            from utils.document_ai_generator import AIDocumentGenerator

            ai_doc_generator = AIDocumentGenerator(self.gemini_api)

            # Get user ID
            user_id = update.effective_user.id

            # Get content from user data
            content = context.user_data.get("doc_export_text", "")

            # Generate document
            document_bytes, title = await ai_doc_generator.generate_ai_document(
                prompt=content,
                output_format=document_format,
                document_type=document_format,
                model="gemini",
                max_tokens=4000,
            )

            # Send document to user
            with io.BytesIO(document_bytes) as doc_io:
                # Set filename based on format and title
                sanitized_title = (
                    "".join(c for c in title if c.isalnum() or c in " _-")
                    .strip()
                    .replace(" ", "_")
                )
                doc_io.name = f"{sanitized_title}_{datetime.now().strftime('%Y%m%d')}.{document_format}"

                # Format the caption to escape Markdown special characters
                model = context.user_data.get(
                    "aidoc_model", "gemini"
                )  # Default to gemini if not specified
                model_display_name = model.capitalize()
                if model == "quasar_alpha":
                    model_display_name = "Quasar Alpha"

                safe_title = (
                    title.replace("*", "\\*")
                    .replace("_", "\\_")
                    .replace("`", "\\`")
                    .replace("[", "\\[")
                    .replace("]", "\\]")
                )
                caption = (
                    f"📄 {safe_title}\n\nGenerated using {model_display_name} model"
                )

                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=doc_io,
                    filename=doc_io.name,
                    caption=caption,
                    parse_mode="MarkdownV2",
                )

            # Log success
            self.telegram_logger.log_message(
                f"AI document generated successfully in {document_format} format using Gemini model",
                user_id,
            )

            # Clear the user data
            for key in ["doc_export_text", "aidoc_type", "aidoc_format", "aidoc_model"]:
                if key in context.user_data:
                    del context.user_data[key]

        except Exception as e:
            self.logger.error(f"AI document generation error: {str(e)}")
            error_text = f"Sorry, there was an error generating your document: {str(e)}"
            # Try to edit the message first, then send a new one if it fails
            try:
                if update.callback_query:
                    await update.callback_query.edit_message_text(error_text)
                else:
                    # If callback query is not available, send a new message
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id, text=error_text
                    )
            except Exception as error:
                self.logger.error(f"Failed to edit message or send error: {error}")
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Sorry, there was an error generating your document and sending the details.",
                )

    def register_handlers(self, application: Application, cache=None) -> None:
        try:
            # Command handlers
            application.add_handler(CommandHandler("start", self.start_command))
            application.add_handler(CommandHandler("help", self.help_command))
            application.add_handler(CommandHandler("reset", self.reset_command))
            application.add_handler(CommandHandler("settings", self.settings))
            application.add_handler(CommandHandler("stats", self.handle_stats))
            application.add_handler(CommandHandler("export", self.handle_export))
            application.add_handler(
                CommandHandler("preferences", self.handle_preferences)
            )
            application.add_handler(
                CommandHandler("generate_image", self.generate_image_command)
            )
            application.add_handler(
                CommandHandler("imagen3", self.generate_image_advanced)
            )
            application.add_handler(
                CommandHandler("genvid", self.generate_video_command)
            )
            application.add_handler(
                CommandHandler("genimg", self.generate_together_image)
            )
            application.add_handler(
                CommandHandler("switchmodel", self.switch_model_command)
            )
            # Add document export command
            application.add_handler(
                CommandHandler("exportdoc", self.export_to_document)
            )
            # Add AI document generation command
            application.add_handler(
                CommandHandler("gendoc", self.generate_ai_document_command)
            )

            # Specific callback handlers FIRST
            application.add_handler(
                CallbackQueryHandler(self.handle_model_selection, pattern="^model_")
            )
            application.add_handler(
                CallbackQueryHandler(
                    self.handle_image_prompt_callback,
                    pattern="^(confirm|cancel|edit)_image_prompt$",
                )
            )
            application.add_handler(
                CallbackQueryHandler(
                    self.handle_image_settings, pattern="^img_.+_steps_.+$"
                )
            )

            # Save cache for use in command handlers if needed
            self.response_cache = cache

            # General callback handler LAST
            application.add_handler(CallbackQueryHandler(self.handle_callback_query))

            self.logger.info("Command handlers registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register command handlers: {e}")
            raise
