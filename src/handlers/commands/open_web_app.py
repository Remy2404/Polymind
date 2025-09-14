"""
Web App opening command handlers.
Contains commands to open and manage Telegram Mini Web App integration.
"""

import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import ContextTypes


class OpenWebAppCommands:
    """Handles Web App opening and integration commands"""

    def __init__(self, user_data_manager, telegram_logger):
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)
        self.web_app_url = os.getenv("WEBAPP_URL_PRODUCTION") or os.getenv(
            "WEBAPP_URL_DEV", ""
        )
        if self.web_app_url:
            self.logger.info(f"Web App URL: {self.web_app_url}")
        else:
            self.logger.warning("No Web App URL configured")

    def _validate_web_app_url(self, url: str) -> bool:
        """Allow only HTTPS URLs"""
        return bool(url and url.startswith("https://"))

    async def open_web_app_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /webapp command to open the Telegram Mini Web App"""
        user_id = update.effective_user.id
        self.telegram_logger.log_message("Web App opening requested", user_id)
        try:
            web_app_url = self.web_app_url
            if not self._validate_web_app_url(web_app_url):
                await update.message.reply_text(
                    "❌ Web App is not available at the moment."
                )
                return
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🚀 Open Web App", web_app=WebAppInfo(url=web_app_url)
                    )
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "🌟 **Open Web App**", reply_markup=reply_markup, parse_mode="Markdown"
            )
            self.logger.info(
                f"Web app opened for user {user_id} with URL: {web_app_url}"
            )
        except Exception as e:
            self.logger.error(f"Error opening web app for user {user_id}: {str(e)}")
            await update.message.reply_text(
                "❌ Sorry, there was an error opening the web app."
            )
