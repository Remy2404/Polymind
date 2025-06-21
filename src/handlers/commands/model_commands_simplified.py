"""
Simplified Model switching command handlers.
Uses the unified API management system.
"""

import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from services.model_handlers.simple_api_manager import SuperSimpleAPIManager, APIProvider
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
        self.api_manager = SuperSimpleAPIManager(
            gemini_api=gemini_api,
            deepseek_api=deepseek_api,
            openrouter_api=openrouter_api,
        )

    async def switch_model_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /switchmodel command to let users select their preferred LLM."""
        user_id = update.effective_user.id

        # Get current model preference
        current_model = await self.user_data_manager.get_user_preference(
            user_id, "preferred_model", default="gemini"
        )

        # Get current model display name
        current_config = self.api_manager.get_model_config(current_model)
        current_model_name = (
            current_config.display_name if current_config else "Unknown"
        )

        # Build dynamic keyboard from available models
        keyboard = []
        row = []

        # Get all models grouped by provider
        all_models = self.api_manager.get_all_models()

        # Sort models by provider for better organization
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

        # Group models in rows of 2
        for i, (model_id, model_config) in enumerate(sorted_models):
            # Create button with emoji and name
            button_text = f"{model_config.emoji} {model_config.display_name}"
            button = InlineKeyboardButton(
                button_text, callback_data=f"model_{model_id}"
            )

            row.append(button)

            # Add row after every 2 buttons or at the end
            if (i + 1) % 2 == 0 or i == len(sorted_models) - 1:
                keyboard.append(row)
                row = []

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

        # Extract model ID from callback data
        selected_model = query.data.replace("model_", "")

        # Get model configuration
        model_config = self.api_manager.get_model_config(selected_model)
        if not model_config:
            await query.edit_message_text(
                "❌ Invalid model selection. Please try again.",
            )
            return

        # Save user preference
        await self.user_data_manager.set_user_preference(
            user_id, "preferred_model", selected_model
        )

        # Log the model change
        self.logger.info(f"User {user_id} switched to model: {selected_model}")

        # Send confirmation message
        await query.edit_message_text(
            f"✅ Model changed to *{model_config.display_name}*\n\n"
            f"{model_config.description if model_config.description else 'AI assistant ready to help!'}\n\n"
            "You can now start chatting with your new AI model! 🚀",
            parse_mode="Markdown",
        )

    async def list_models_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /listmodels command to show all available models."""

        # Group models by provider
        providers_data = {
            APIProvider.GEMINI: {"title": "*📱 Gemini Models:*", "models": []},
            APIProvider.DEEPSEEK: {"title": "*🧠 DeepSeek Models:*", "models": []},
            APIProvider.OPENROUTER: {
                "title": "*🌐 OpenRouter Models (Free):*",
                "models": [],            },
        }
        
        # Group models by provider
        for model_id, config in self.api_manager.get_all_models().items():
            model_line = f"• {config.emoji} *{config.display_name}*"
            if config.description:
                model_line += f" - {config.description}"

            if config.provider in providers_data:
                providers_data[config.provider]["models"].append(model_line)
            else:
                # Handle unknown providers gracefully
                self.logger.warning(f"Unknown provider for model {model_id}: {config.provider}")

        # Build message
        message_parts = ["🤖 *Available AI Models*\n"]

        for provider_info in providers_data.values():
            if provider_info["models"]:
                message_parts.append(provider_info["title"])
                message_parts.extend(provider_info["models"])
                message_parts.append("")

        message_parts.append("💡 Use /switchmodel to change your active model.")

        await update.message.reply_text("\n".join(message_parts), parse_mode="Markdown")

    async def current_model_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle the /currentmodel command to show current model info."""
        user_id = update.effective_user.id

        # Get current model preference
        current_model = await self.user_data_manager.get_user_preference(
            user_id, "preferred_model", default="gemini"
        )

        # Get current model display name and info
        current_config = self.api_manager.get_model_config(current_model)
        
        if current_config:
            message = (
                f"ℹ️ **Current Model Information**\n\n"
                f"**Name:** {current_config.emoji} {current_config.display_name}\n"
                f"**Provider:** {current_config.provider.value.title()}\n"
            )
            
            if current_config.description:
                message += f"**Description:** {current_config.description}\n"
            
            if hasattr(current_config, 'openrouter_key') and current_config.openrouter_key:
                message += f"**OpenRouter Key:** `{current_config.openrouter_key}`\n"
            
            message += f"\n✅ This model is currently active and ready to use!"
        else:
            message = f"❌ Current model '{current_model}' not found in configuration."

        await update.message.reply_text(message, parse_mode="Markdown")
