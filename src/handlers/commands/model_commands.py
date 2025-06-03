"""
Model switching command handlers.
Contains model switching and model selection functionality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import logging


class ModelCommands:
    def __init__(self, user_data_manager, telegram_logger, deepseek_api=None, openrouter_api=None):
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.deepseek_api = deepseek_api
        self.openrouter_api = openrouter_api
        self.logger = logging.getLogger(__name__)

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
                [                    InlineKeyboardButton(
                        "Gemini-2.0-Flash", callback_data="model_gemini"
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
            )            # Map model code to display name
            model_names = {
                "gemini": "Gemini-2.0-Flash",
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
            selected_model_id_for_backend = "llama4-maverick"
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
            fallback_model_names = {
                "gemini": "Gemini-2.0-Flash",
                "deepseek": "DeepSeek 70B",
                "optimus_alpha": "🌀 Optimus Alpha",
                "deepcoder": "🧑‍💻 DeepCoder",
                "llama4_maverick": "🦙 Llama-4 Maverick",
            }
            model_name = fallback_model_names.get(
                selected_model_from_callback, selected_model_from_callback.capitalize()
            ) 
            await self.user_data_manager.set_user_preference(
                user_id, "preferred_model", selected_model_id_for_backend
            )
            
            await self.user_data_manager.update_user_settings_async(
                user_id, {"active_model": selected_model_id_for_backend}
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
