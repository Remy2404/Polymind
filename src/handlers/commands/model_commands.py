"""
Model switching and listing commands for the Telegram bot.
Provides categorized model lists and easy switching between AI models.
"""

import os
import sys
import logging
from typing import List
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from src.services.model_handlers.simple_api_manager import (
    SuperSimpleAPIManager,
)
from src.services.user_data_manager import UserDataManager
from src.handlers.support_tool_call import ToolCallSupportDetector
from src.handlers.commands.callback_data_mapper import callback_mapper

logger = logging.getLogger(__name__)


class ModelCommands:
    def __init__(
        self, api_manager: SuperSimpleAPIManager, user_data_manager: UserDataManager
    ):
        self.api_manager = api_manager
        self.user_data_manager = user_data_manager
        self.tool_call_detector = ToolCallSupportDetector(api_manager)

    async def switchmodel_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """🔄 Main switchmodel command - shows category selection"""
        user_id = update.effective_user.id
        current_model = await self.user_data_manager.get_user_preference(
            user_id, "preferred_model", default="gemini"
        )
        current_config = self.api_manager.get_model_config(current_model)
        current_name = (
            getattr(current_config, "display_name", current_model)
            if current_config
            else current_model
        )
        categories = self.api_manager.get_models_by_category()
        keyboard = []
        category_buttons = []
        for category_id, category_info in categories.items():
            model_count = len(category_info["models"])
            button_text = (
                f"{category_info['emoji']} {category_info['name']} ({model_count})"
            )
            button = InlineKeyboardButton(
                button_text, callback_data=f"category_{category_id}"
            )
            category_buttons.append(button)
            if len(category_buttons) == 2:
                keyboard.append(category_buttons)
                category_buttons = []
        if category_buttons:
            keyboard.append(category_buttons)
        tool_call_count = 0
        try:
            tool_call_stats = self.tool_call_detector.get_tool_call_statistics()
            tool_call_count = tool_call_stats.get("tool_call_models", 0)
        except Exception:
            pass
        button_text = f"🛠️ Tool-Calling Models ({tool_call_count})"
        keyboard.append(
            [
                InlineKeyboardButton(
                    "📊 All Models (A-Z)", callback_data="category_all"
                ),
                InlineKeyboardButton("ℹ️ Current Model", callback_data="current_model"),
                InlineKeyboardButton(button_text, callback_data="category_tool_call"),
            ]
        )
        reply_markup = InlineKeyboardMarkup(keyboard)
        message = (
            f"🤖 **Model Selection Center**\n\n"
            f"Current Model: **{current_name}** {getattr(current_config, 'emoji', getattr(current_config, 'indicator_emoji', '🤖')) if current_config else '🤖'}\n\n"
            f"📂 Choose a category to browse models:\n"
            f"• Select any category to see available models\n"
            f"• All OpenRouter models are **completely free** 🆓\n"
            f"• Switch instantly between any model ⚡"
        )
        await update.message.reply_text(
            message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def handle_category_selection(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle when user selects a model category"""
        query = update.callback_query
        await query.answer()
        category_id = query.data.replace("category_", "")
        if category_id == "all":
            await self._show_all_models(query)
        elif category_id == "current":
            await self._show_current_model(query)
        elif category_id == "tool_call":
            await self._show_tool_call_models(query)
        else:
            await self._show_category_models(query, category_id)

    async def _show_category_models(self, query, category_id: str) -> None:
        """Show models in a specific category"""
        categories = self.api_manager.get_models_by_category()
        category_info = categories.get(category_id)
        if not category_info:
            await query.edit_message_text("❌ Category not found.")
            return
        models = category_info["models"]
        keyboard = []
        for model_id, config in models.items():
            emoji = getattr(config, "emoji", getattr(config, "indicator_emoji", "🤖"))
            display_name = getattr(config, "display_name", model_id)
            display_text = f"{emoji} {display_name}"
            if config.openrouter_key:
                display_text += f"\n({config.openrouter_key})"
            button = InlineKeyboardButton(
                display_text,
                callback_data=callback_mapper.get_callback_data(model_id, "model"),
            )
            keyboard.append([button])
        keyboard.append(
            [
                InlineKeyboardButton(
                    "⬅️ Back to Categories", callback_data="back_to_categories"
                )
            ]
        )
        reply_markup = InlineKeyboardMarkup(keyboard)
        message = (
            f"{category_info['emoji']} **{category_info['name']}**\n\n"
            f"📋 Available models in this category:\n"
            f"• Click any model to switch instantly\n"
            f"• Total models: **{len(models)}**\n\n"
            f"🔄 Select a model to switch:"
        )
        await query.edit_message_text(
            message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def _show_all_models(self, query) -> None:
        """Show all models alphabetically"""
        all_models = self.api_manager.get_all_models()
        sorted_models = sorted(
            all_models.items(),
            key=lambda x: getattr(x[1], "display_name", x[0]) if x[1] else x[0],
        )
        keyboard = []
        for model_id, config in sorted_models:
            emoji = getattr(config, "emoji", getattr(config, "indicator_emoji", "🤖"))
            display_name = getattr(config, "display_name", model_id)
            display_text = f"{emoji} {display_name}"
            if config.openrouter_key:
                display_text += f"\n({config.openrouter_key})"
            button = InlineKeyboardButton(
                display_text,
                callback_data=callback_mapper.get_callback_data(model_id, "model"),
            )
            keyboard.append([button])
        keyboard.append(
            [
                InlineKeyboardButton(
                    "⬅️ Back to Categories", callback_data="back_to_categories"
                )
            ]
        )
        reply_markup = InlineKeyboardMarkup(keyboard)
        message = (
            f"📊 **All Available Models** (A-Z)\n\n"
            f"📋 Complete list of {len(all_models)} models:\n"
            f"• Sorted alphabetically for easy browsing\n"
            f"• Click any model to switch instantly\n\n"
            f"🔄 Select a model:"
        )
        await query.edit_message_text(
            message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def _show_tool_call_models(self, query) -> None:
        """Show only models that support tool calling"""
        tool_call_models = self.tool_call_detector.get_tool_call_supported_models()
        if not tool_call_models:
            await query.edit_message_text(
                "❌ No tool-calling models found in current configuration.",
                reply_markup=InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "⬅️ Back to Categories",
                                callback_data="back_to_categories",
                            )
                        ]
                    ]
                ),
            )
            return
        sorted_models = sorted(
            tool_call_models.items(),
            key=lambda x: getattr(x[1], "display_name", x[0]) if x[1] else x[0],
        )
        keyboard = []
        for model_id, config in sorted_models:
            emoji = getattr(config, "emoji", getattr(config, "indicator_emoji", "🤖"))
            display_name = getattr(config, "display_name", model_id)
            display_text = f"{emoji} {display_name}"
            if config.openrouter_key:
                display_text += f"\n({config.openrouter_key})"
            button = InlineKeyboardButton(
                display_text,
                callback_data=callback_mapper.get_callback_data(model_id, "model"),
            )
            keyboard.append([button])
        keyboard.append(
            [
                InlineKeyboardButton(
                    "⬅️ Back to Categories", callback_data="back_to_categories"
                )
            ]
        )
        reply_markup = InlineKeyboardMarkup(keyboard)
        stats = self.tool_call_detector.get_tool_call_statistics()
        message = (
            f"🛠️ **Tool-Calling Models Only**\n\n"
            f"📋 Models with tool/function calling support:\n"
            f"• **{stats['tool_call_models']}** out of **{stats['total_models']}** models\n"
            f"• **{stats['percentage']}%** of all available models\n"
            f"• Click any model to switch instantly\n\n"
            f"🔄 Select a tool-calling model:"
        )
        await query.edit_message_text(
            message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def _show_current_model(self, query) -> None:
        """Show current model information"""
        user_id = query.from_user.id
        current_model = await self.user_data_manager.get_user_preference(
            user_id, "preferred_model", default="gemini"
        )
        current_config = self.api_manager.get_model_config(current_model)
        if current_config:
            emoji = getattr(
                current_config,
                "emoji",
                getattr(current_config, "indicator_emoji", "🤖"),
            )
            display_name = getattr(current_config, "display_name", current_model)
            message = (
                f"ℹ️ **Current Model Information**\n\n"
                f"**Name:** {emoji} {display_name}\n"
                f"**Provider:** {current_config.provider.value.title()}\n"
                f"**Description:** {current_config.description}\n"
            )
            if current_config.openrouter_key:
                message += f"**OpenRouter Key:** `{current_config.openrouter_key}`\n"
            message += "\n✅ This model is currently active and ready to use!"
        else:
            message = f"❌ Current model '{current_model}' not found in configuration."
        keyboard = [
            [
                InlineKeyboardButton(
                    "⬅️ Back to Categories", callback_data="back_to_categories"
                )
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def handle_model_selection(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle when user selects a specific model"""
        query = update.callback_query
        user_id = query.from_user.id
        await query.answer()
        selected_model = callback_mapper.get_model_id(query.data)
        if not selected_model:
            selected_model = query.data.replace("model_", "")
        model_config = self.api_manager.get_model_config(selected_model)
        if not model_config:
            await query.edit_message_text("❌ Invalid model selection.")
            return
        await self.user_data_manager.set_user_preference(
            user_id, "preferred_model", selected_model
        )
        logger.info(f"User {user_id} switched to: {selected_model}")
        emoji = getattr(
            model_config, "emoji", getattr(model_config, "indicator_emoji", "🤖")
        )
        display_name = getattr(model_config, "display_name", selected_model)
        message = (
            f"✅ **Successfully switched to:**\n\n"
            f"**{emoji} {display_name}**\n\n"
            f"📝 {model_config.description}\n"
        )
        if model_config.openrouter_key:
            message += f"\n🔗 **OpenRouter Model:** `{model_config.openrouter_key}`\n"
        message += "\n🚀 **Ready to chat!** Send me any message to test the new model."
        keyboard = [
            [
                InlineKeyboardButton(
                    "🔄 Switch Again", callback_data="back_to_categories"
                )
            ],
            [InlineKeyboardButton("ℹ️ Model Info", callback_data="current_model")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def handle_back_to_categories(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle back to categories button"""
        query = update.callback_query
        await query.answer()
        user_id = query.from_user.id
        current_model = await self.user_data_manager.get_user_preference(
            user_id, "preferred_model", default="gemini"
        )
        current_config = self.api_manager.get_model_config(current_model)
        current_name = (
            getattr(current_config, "display_name", current_model)
            if current_config
            else current_model
        )
        categories = self.api_manager.get_models_by_category()
        keyboard = []
        category_buttons = []
        for category_id, category_info in categories.items():
            model_count = len(category_info["models"])
            button_text = (
                f"{category_info['emoji']} {category_info['name']} ({model_count})"
            )
            button = InlineKeyboardButton(
                button_text, callback_data=f"category_{category_id}"
            )
            category_buttons.append(button)
            if len(category_buttons) == 2:
                keyboard.append(category_buttons)
                category_buttons = []
        if category_buttons:
            keyboard.append(category_buttons)
        keyboard.append(
            [
                InlineKeyboardButton(
                    "📊 All Models (A-Z)", callback_data="category_all"
                ),
                InlineKeyboardButton("ℹ️ Current Model", callback_data="current_model"),
            ]
        )
        tool_call_count = 0
        try:
            tool_call_stats = self.tool_call_detector.get_tool_call_statistics()
            tool_call_count = tool_call_stats.get("tool_call_models", 0)
        except Exception:
            pass
        button_text = f"🛠️ Tool-Calling Models ({tool_call_count})"
        keyboard.append(
            [InlineKeyboardButton(button_text, callback_data="category_tool_call")]
        )
        reply_markup = InlineKeyboardMarkup(keyboard)
        message = (
            f"🤖 **Model Selection Center**\n\n"
            f"Current Model: **{current_name}** {getattr(current_config, 'emoji', getattr(current_config, 'indicator_emoji', '🤖')) if current_config else '🤖'}\n\n"
            f"📂 Choose a category to browse models:\n"
            f"• Select any category to see available models\n"
            f"• All OpenRouter models are **completely free** 🆓\n"
            f"• Switch instantly between any model ⚡"
        )
        await query.edit_message_text(
            message, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def current_model_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show current model information (standalone command)"""
        user_id = update.effective_user.id
        current_model = await self.user_data_manager.get_user_preference(
            user_id, "preferred_model", default="gemini"
        )
        current_config = self.api_manager.get_model_config(current_model)
        if current_config:
            emoji = getattr(
                current_config,
                "emoji",
                getattr(current_config, "indicator_emoji", "🤖"),
            )
            display_name = getattr(current_config, "display_name", current_model)
            message = (
                f"ℹ️ **Current Active Model**\n\n"
                f"**{emoji} {display_name}**\n\n"
                f"**Provider:** {current_config.provider.value.title()}\n"
                f"**Description:** {current_config.description}\n"
            )
            if current_config.openrouter_key:
                message += f"**OpenRouter Key:** `{current_config.openrouter_key}`\n"
            message += "\n✅ This model is ready to use! Send any message to chat."
        else:
            message = f"❌ Current model '{current_model}' not found in configuration."
        await update.message.reply_text(message, parse_mode="Markdown")

    async def list_models_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List all available models by category (text-only)"""
        categories = self.api_manager.get_models_by_category()
        message_parts = ["🤖 **Available AI Models by Category**\n"]
        for category_id, category_info in categories.items():
            message_parts.append(
                f"\n{category_info['emoji']} **{category_info['name']}:**"
            )
            for model_id, config in category_info["models"].items():
                emoji = getattr(
                    config, "emoji", getattr(config, "indicator_emoji", "🤖")
                )
                display_name = getattr(config, "display_name", model_id)
                model_line = f"• {emoji} {display_name}"
                if config and config.openrouter_key:
                    model_line += f" (`{config.openrouter_key}`)"
                message_parts.append(model_line)
        message_parts.append("\n💡 Use `/switchmodel` for interactive model selection.")
        message_parts.append("💡 Use `/currentmodel` to see your active model.")
        await update.message.reply_text("\n".join(message_parts), parse_mode="Markdown")


def get_model_command_handlers(model_commands: ModelCommands) -> List[tuple]:
    """Get list of callback handlers for model commands"""
    return [
        ("category_", model_commands.handle_category_selection),
        ("model_", model_commands.handle_model_selection),
        ("back_to_categories", model_commands.handle_back_to_categories),
        (
            "current_model",
            model_commands.handle_category_selection,
        ),
    ]
