"""
Export and document conversion command handlers.
Contains export conversation, document export, and format conversion functionality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from datetime import datetime
import logging
import io


class ExportCommands:
    def __init__(self, gemini_api, user_data_manager, telegram_logger):
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)

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
        try:            # Import AI document generator
            from utils.docgen.document_ai_generator import AIDocumentGenerator

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
