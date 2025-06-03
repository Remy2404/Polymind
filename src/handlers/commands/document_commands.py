"""
AI Document generation command handlers.
Contains AI document generation, format selection, and callback handling.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from datetime import datetime
import logging
import io


class DocumentCommands:
    def __init__(self, gemini_api, user_data_manager, telegram_logger):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)

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
        prompt = context.user_data.get("aidoc_prompt", "")        # Ensure model name reflects the actual models available
        model_name = (
            "Gemini-2.0-Flash"
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
        query = update.callback_query
        if not query:
            self.logger.warning(
                "handle_ai_document_callback received update without callback_query"
            )
            return
        await query.answer()

        if data.startswith("aidoc_type_"):
            # Store document type preference
            doc_type = data.replace("aidoc_type_", "")
            context.user_data["aidoc_type"] = doc_type

            # Ask for document topic
            await query.edit_message_text(
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
            await self._generate_and_send_ai_document(update, context)

        elif data.startswith("aidoc_model_"):
            # Update the model selection
            new_model = data.replace("aidoc_model_", "")
            current_model = context.user_data.get("aidoc_model", "gemini")

            if new_model == current_model:
                model_name = (
                    "Gemini-2.0-flash" if current_model == "gemini" else "DeepSeek 70B"
                )
                await query.answer(f"{model_name} is already selected.")
                return

            # Model has changed, update user data and the message
            context.user_data["aidoc_model"] = new_model
            # Update the message to show selected model
            await self._show_ai_document_format_selection(update, context)

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

        try:            # Import AI document generator (ensure path is correct)
            from utils.docgen.document_ai_generator import AIDocumentGenerator

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
