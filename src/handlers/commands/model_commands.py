import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from services.model_handlers.simple_api_manager import (
    SuperSimpleAPIManager,
    APIProvider,
)
import logging


class ModelCommands:
    def __init__(
        self,
        user_data_manager,
        telegram_logger,
        deepseek_api=None,
        openrouter_api=None,
        gemini_api=None,
    ):
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)

        # 🚀 Initialize the super simple API manager
        self.api_manager = SuperSimpleAPIManager(
            gemini_api=gemini_api,
            deepseek_api=deepseek_api,
            openrouter_api=openrouter_api,
        )

    async def switch_model_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """🔄 Handle model switching with a nice UI"""
        user_id = update.effective_user.id

        # Get current model
        current_model = await self.user_data_manager.get_user_preference(
            user_id, "preferred_model", default="gemini"
        )

        current_config = self.api_manager.get_model_config(current_model)
        current_name = current_config.display_name if current_config else "Unknown"

        # Build keyboard with all available models
        keyboard = []
        row = []

        all_models = self.api_manager.get_all_models()

        # Sort by provider for better organization
        providers_order = [
            APIProvider.GEMINI,
            APIProvider.DEEPSEEK,
            APIProvider.OPENROUTER,
        ]
        sorted_models = []

        for provider in providers_order:
            provider_models = [
                (k, v) for k, v in all_models.items() if v.provider == provider
            ]
            sorted_models.extend(provider_models)

        # Create buttons (2 per row)
        for i, (model_id, model_config) in enumerate(sorted_models):
            button_text = f"{model_config.emoji} {model_config.display_name}"
            button = InlineKeyboardButton(
                button_text, callback_data=f"model_{model_id}"
            )
            row.append(button)

            # New row every 2 buttons
            if (i + 1) % 2 == 0 or i == len(sorted_models) - 1:
                keyboard.append(row)
                row = []

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"🔄 Current model: *{current_name}*\n\n" "Choose your AI model:",
            reply_markup=reply_markup,
            parse_mode="Markdown",
        )

    async def handle_model_selection(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """✅ Handle when user selects a model"""
        query = update.callback_query
        user_id = query.from_user.id
        await query.answer()

        # Extract model ID
        selected_model = query.data.replace("model_", "")
        model_config = self.api_manager.get_model_config(selected_model)

        if not model_config:
            await query.edit_message_text("❌ Invalid model selection.")
            return

        # Save user preference
        await self.user_data_manager.set_user_preference(
            user_id, "preferred_model", selected_model
        )

        self.logger.info(f"User {user_id} switched to: {selected_model}")

        # Confirmation message
        await query.edit_message_text(
            f"✅ Switched to *{model_config.display_name}*\n\n"
            f"{model_config.description}\n\n"
            "Ready to chat! 🚀",
            parse_mode="Markdown",
        )

    async def list_models_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """📋 Show all available models"""

        # Group by provider
        providers_data = {
            APIProvider.GEMINI: {"title": "*🧠 Gemini Models:*", "models": []},
            APIProvider.DEEPSEEK: {"title": "*🔮 DeepSeek Models:*", "models": []},
            APIProvider.OPENROUTER: {
                "title": "*🌐 OpenRouter Models (Free):*",
                "models": [],
            },
        }

        # Organize models by provider
        for model_id, config in self.api_manager.get_all_models().items():
            model_line = f"• {config.emoji} *{config.display_name}*"
            if config.description:
                model_line += f" - {config.description}"

            if config.provider in providers_data:
                providers_data[config.provider]["models"].append(model_line)

        # Build message
        message_parts = ["🤖 *Available AI Models*\n"]

        for provider_info in providers_data.values():
            if provider_info["models"]:
                message_parts.append(provider_info["title"])
                message_parts.extend(provider_info["models"])
                message_parts.append("")

        message_parts.append("💡 Use /switchmodel to change your model.")

        await update.message.reply_text("\n".join(message_parts), parse_mode="Markdown")
