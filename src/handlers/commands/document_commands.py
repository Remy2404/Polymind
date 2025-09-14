"""
AI Document generation command handlers.
Contains AI document generation, format selection, and callback handling.
"""
import sys
import os
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from datetime import datetime
import logging
import io
from src.services.model_handlers.model_configs import ModelConfigurations
class DocumentCommands:
    def __init__(
        self, gemini_api, user_data_manager, telegram_logger, api_manager=None
    ):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)
        if api_manager:
            self.api_manager = api_manager
        else:
            from src.services.model_handlers.simple_api_manager import (
                SuperSimpleAPIManager,
            )
            self.api_manager = SuperSimpleAPIManager(gemini_api=gemini_api)
    async def generate_ai_document_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate an AI-authored document in PDF or DOCX format"""
        user_id = update.effective_user.id
        self.telegram_logger.log_message(
            f"AI document generation requested by user {user_id}", user_id
        )
        if not context.args:
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
        prompt = " ".join(context.args)
        context.user_data["aidoc_prompt"] = prompt
        context.user_data["aidoc_type"] = "article"
        await self._show_ai_document_format_selection(update, context)
    async def _show_ai_document_format_selection(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show document format selection buttons"""
        all_models = ModelConfigurations.get_all_models()
        model_buttons = []
        models_per_row = 2
        current_row = []
        preferred_models = [
            "gemini",
            "llama-3.2-3b-instruct",
            "qwen3-235b",
            "dolphin-mistral-24b-venice-edition",
            "qwerky-72b",
            "deephermes-3-llama-8b",
            "dolphin3-r1-mistral-24b",
            "openrouter-horizon-beta",
        ]
        for model_id in preferred_models:
            if model_id in all_models:
                model_config = all_models[model_id]
                button_text = (
                    f"{model_config.indicator_emoji} {model_config.display_name}"
                )
                current_row.append(
                    InlineKeyboardButton(
                        button_text, callback_data=f"aidoc_model_{model_id}"
                    )
                )
                if len(current_row) == models_per_row:
                    model_buttons.append(current_row)
                    current_row = []
        if current_row:
            model_buttons.append(current_row)
        format_options = [
            [
                InlineKeyboardButton("📄 PDF Format", callback_data="aidoc_format_pdf"),
                InlineKeyboardButton(
                    "📝 DOCX Format", callback_data="aidoc_format_docx"
                ),
            ]
        ]
        all_buttons = model_buttons + format_options
        reply_markup = InlineKeyboardMarkup(all_buttons)
        model = context.user_data.get("aidoc_model", "gemini")
        doc_type = context.user_data.get("aidoc_type", "article")
        prompt = context.user_data.get("aidoc_prompt", "")
        model_config = all_models.get(model)
        model_name = model_config.display_name if model_config else "Gemini 2.0 Flash"
        message = (
            f"🤖 *AI Document Generator*\n\n"
            f"I'll create a {doc_type} about:\n"
            f'_"{prompt}"_\n\n'
            f"Current model: *{model_name}*\n\n"
            f"Please select the AI model and output format:"
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
        query = update.callback_query
        if not query:
            self.logger.warning(
                "handle_ai_document_callback received update without callback_query"
            )
            return
        await query.answer()
        if data.startswith("aidoc_type_"):
            doc_type = data.replace("aidoc_type_", "")
            context.user_data["aidoc_type"] = doc_type
            await query.edit_message_text(
                f"📝 You selected: *{doc_type.capitalize()}*\n\n"
                f"Now, please send me the topic or subject for your {doc_type}.",
                parse_mode="Markdown",
            )
            context.user_data["awaiting_aidoc_topic"] = True
        elif data.startswith("aidoc_format_"):
            output_format = data.replace("aidoc_format_", "")
            context.user_data["aidoc_format"] = output_format
            await self._generate_and_send_ai_document(update, context)
        elif data.startswith("aidoc_model_"):
            new_model = data.replace("aidoc_model_", "")
            current_model = context.user_data.get("aidoc_model", "gemini")
            if new_model == current_model:
                all_models = ModelConfigurations.get_all_models()
                model_config = all_models.get(current_model)
                model_name = (
                    model_config.display_name if model_config else "Selected Model"
                )
                await query.answer(f"{model_name} is already selected.")
                return
            context.user_data["aidoc_model"] = new_model
            await self._show_ai_document_format_selection(update, context)
    async def _generate_and_send_ai_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate AI document and send it to user"""
        query = update.callback_query
        if not query:
            self.logger.warning(
                "_generate_and_send_ai_document received update without callback_query"
            )
            await context.bot.send_message(
                chat_id=update.effective_chat.id, text="Starting document generation..."
            )
        else:
            await query.edit_message_text(
                f"🤖 *Generating your {context.user_data.get('aidoc_type', 'article')}*\n\n"
                f"I'm creating a {context.user_data.get('aidoc_format', 'pdf').upper()} document about:\n"
                f'_"{context.user_data.get("aidoc_prompt", "")}"_\n\n'
                f"This may take a minute or two. Please wait...",
                parse_mode="Markdown",
            )
        user_id = update.effective_user.id
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
            from src.utils.docgen.document_ai_generator import AIDocumentGenerator
            ai_doc_generator = AIDocumentGenerator(api_manager=self.api_manager)
            document_bytes, title = await ai_doc_generator.generate_ai_document(
                prompt=prompt,
                output_format=output_format,
                document_type=doc_type,
                model=model,
                max_tokens=4000,
            )
            with io.BytesIO(document_bytes) as doc_io:
                sanitized_title = (
                    "".join(c for c in title if c.isalnum() or c in " _-")
                    .strip()
                    .replace(" ", "_")
                )
                doc_io.name = f"{sanitized_title}_{datetime.now().strftime('%Y%m%d')}.{output_format}"
                all_models = ModelConfigurations.get_all_models()
                model_config = all_models.get(model)
                model_display_name = (
                    model_config.display_name if model_config else model.capitalize()
                )
                caption = f"📄 {title}\n\nGenerated using {model_display_name} model"
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=doc_io,
                    filename=doc_io.name,
                    caption=caption,
                    parse_mode=None,
                )
            self.telegram_logger.log_message(
                f"AI document generated successfully in {output_format} format using {model} model",
                user_id,
            )
            for key in [
                "aidoc_prompt",
                "aidoc_type",
                "aidoc_format",
                "aidoc_model",
                "awaiting_aidoc_topic",
            ]:
                if key in context.user_data:
                    del context.user_data[key]
        except Exception as e:
            self.logger.error(f"AI document generation error: {str(e)}")
            error_text = f"Sorry, there was an error generating your document: {str(e)}"
            try:
                if query:
                    await query.edit_message_text(error_text)
                else:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id, text=error_text
                    )
            except Exception as edit_error:
                self.logger.error(f"Failed to edit message or send error: {edit_error}")
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Sorry, there was an error generating your document and sending the details.",
                )
