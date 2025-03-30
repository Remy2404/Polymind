import logging
from telegram import Update
from telegram.ext import ContextTypes
from utils.telegramlog import telegram_logger

logger = logging.getLogger(__name__)


class LanguageManager:
    """Manages language preferences for users"""

    def __init__(self):
        self.supported_languages = {
            "en": "English",
            "km": "Khmer (ភាសាខ្មែរ)",
            "kh": "Khmer (ភាសាខ្មែរ)",  # Alternative code for Khmer
            "ru": "Russian (Русский)",
            "fr": "French (Français)",
            "es": "Spanish (Español)",
            "de": "German (Deutsch)",
            "ja": "Japanese (日本語)",
            "zh": "Chinese (中文)",
            "th": "Thai (ไทย)",
            "vi": "Vietnamese (Tiếng Việt)",
        }

    async def set_language(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set the preferred language for a user"""
        user_id = update.effective_user.id

        # Extract language code from command
        message_parts = update.message.text.split()
        if len(message_parts) < 2:
            # Show available languages if none specified
            language_list = "\n".join(
                f"• {code} - {name}" for code, name in self.supported_languages.items()
            )
            await update.message.reply_text(
                f"Please specify a language code:\n\n{language_list}\n\nExample: /language en"
            )
            return

        language_code = message_parts[1].lower()

        # Normalize some common alternative codes
        if language_code == "kh":
            language_code = "km"  # Standardize to ISO code

        # Check if language is supported
        if language_code not in self.supported_languages:
            await update.message.reply_text(
                f"Sorry, language '{language_code}' is not supported. Available languages:\n\n"
                + "\n".join(
                    f"• {code} - {name}"
                    for code, name in self.supported_languages.items()
                )
            )
            return

        # Save user's language preference
        if hasattr(context.user_data, "get") and callable(context.user_data.get):
            if "preferences" not in context.user_data:
                context.user_data["preferences"] = {}
            context.user_data["preferences"]["language"] = language_code

        # If user_data_manager is available in context (added during handler setup)
        if (
            hasattr(context.application, "user_data_manager")
            and context.application.user_data_manager
        ):
            await context.application.user_data_manager.set_user_preference(
                user_id, "preferred_language", language_code
            )

        # Log the language change
        try:
            telegram_logger.log_message(
                f"User set language to {language_code}", user_id
            )
        except Exception as e:
            logger.error(f"Error logging language change: {e}")

        # Respond with confirmation
        language_name = self.supported_languages[language_code]

        # Bilingual responses for better user experience
        if language_code == "km":
            await update.message.reply_text(
                f"ភាសា​របស់​អ្នក​ត្រូវ​បាន​កំណត់​ទៅ​ជា {language_name}។\n"
                f"Your language has been set to {language_name}."
            )
        else:
            await update.message.reply_text(
                f"Your language has been set to {language_name}."
            )
