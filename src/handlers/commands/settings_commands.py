"""
Settings and user preference command handlers.
Contains settings, stats, export, preferences commands.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import logging


class SettingsCommands:
    def __init__(self, user_data_manager, telegram_logger):
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)

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
            reply_markup=reply_markup,            parse_mode="Markdown",
        )

    async def handle_user_preferences(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        """Handle preference handlers"""
        # Placeholder for now
        await update.callback_query.edit_message_text(
            "Preference settings not implemented yet."
        )

    async def handle_toggle_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        """Handle toggle callbacks for settings"""
        query = update.callback_query
        user_id = update.effective_user.id
        
        if data == "toggle_markdown":
            settings = await self.user_data_manager.get_user_settings(user_id)
            current_value = settings.get("markdown_enabled", True)
            await self.user_data_manager.set_user_setting(
                user_id, "markdown_enabled", not current_value
            )
            status = "enabled" if not current_value else "disabled"
            await query.edit_message_text(f"Markdown Mode has been {status}.")
        elif data == "toggle_code_suggestions":
            settings = await self.user_data_manager.get_user_settings(user_id)
            current_value = settings.get("code_suggestions", True)
            await self.user_data_manager.set_user_setting(
                user_id, "code_suggestions", not current_value
            )
            status = "enabled" if not current_value else "disabled"
            await query.edit_message_text(f"Code Suggestions have been {status}.")
        """Handle toggle callbacks for settings"""
        query = update.callback_query
        user_id = update.effective_user.id
        
        if data == "toggle_markdown":
            settings = await self.user_data_manager.get_user_settings(user_id)
            current_value = settings.get("markdown_enabled", True)
            await self.user_data_manager.set_user_setting(
                user_id, "markdown_enabled", not current_value
            )
            status = "enabled" if not current_value else "disabled"
            await query.edit_message_text(f"Markdown Mode has been {status}.")
        elif data == "toggle_code_suggestions":
            settings = await self.user_data_manager.get_user_settings(user_id)
            current_value = settings.get("code_suggestions", True)
            await self.user_data_manager.set_user_setting(
                user_id, "code_suggestions", not current_value
            )
            status = "enabled" if not current_value else "disabled"
            await query.edit_message_text(f"Code Suggestions have been {status}.")

    async def handle_toggle_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, data: str
    ) -> None:
        """Handle toggle callbacks for settings"""
        query = update.callback_query
        user_id = update.effective_user.id
        
        if data == "toggle_markdown":
            settings = await self.user_data_manager.get_user_settings(user_id)
            current_value = settings.get("markdown_enabled", True)
            await self.user_data_manager.set_user_setting(
                user_id, "markdown_enabled", not current_value
            )
            status = "enabled" if not current_value else "disabled"
            await query.edit_message_text(f"Markdown Mode has been {status}.")
        elif data == "toggle_code_suggestions":
            settings = await self.user_data_manager.get_user_settings(user_id)
            current_value = settings.get("code_suggestions", True)
            await self.user_data_manager.set_user_setting(
                user_id, "code_suggestions", not current_value
            )
            status = "enabled" if not current_value else "disabled"
            await query.edit_message_text(f"Code Suggestions have been {status}.")
