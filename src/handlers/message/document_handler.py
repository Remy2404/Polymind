"""
Document message handling.
"""
import os
import io
import traceback
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from src.handlers.text_handlers import TextHandler
from src.utils.bot_username_helper import BotUsernameHelper


class DocumentMessageHandlerMixin:
    """Mixin for document message handling."""

    async def _handle_document_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming document messages using unified conversation system."""
        user_id = update.effective_user.id
        self.logger.info(f"Processing document for user: {user_id}")
        self.telegram_logger.log_message("Received document message", user_id)

        try:
            if not update.message or not update.message.document:
                await update.message.reply_text("Sorry, I couldn't process this document.")
                return

            document = update.message.document
            if document.file_size and document.file_size > 50 * 1024 * 1024:
                await update.message.reply_text("Sorry, this document is too large (max 50MB).")
                return

            # Check model support
            supports_documents, model_id, error_message = await self.check_model_supports_media(user_id, "document")
            if not supports_documents:
                await update.message.reply_text(error_message, parse_mode="Markdown")
                return

            # Route through TextHandler
            text_handler = TextHandler(
                self.gemini_api,
                self.user_data_manager,
                openrouter_api=self.openrouter_api,
                deepseek_api=self.deepseek_api,
            )

            has_attached_media, media_files, media_type = await text_handler._extract_media_files(update, context)

            if has_attached_media and media_files:
                processing_message = await update.message.reply_text(
                    f"Processing your document: {document.file_name}. Please wait..."
                )
                await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

                try:
                    prompt = update.message.caption or f"Please analyze this {document.file_name} file."
                    await text_handler._handle_media_analysis(
                        update, context, processing_message, media_files, media_type, prompt, user_id, model_id
                    )
                    self.telegram_logger.log_message(
                        f"Document processed successfully: {document.file_name}", user_id
                    )
                    try:
                        await self.user_data_manager.update_stats(user_id, document=True)
                    except Exception:
                        pass

                except Exception as e:
                    self.logger.error(f"Document processing error: {str(e)}")
                    try:
                        await processing_message.edit_text(
                            "Sorry, there was an error processing your document. Please try again."
                        )
                    except Exception:
                        await update.message.reply_text(
                            "Sorry, there was an error processing your document. Please try again."
                        )
            else:
                await update.message.reply_text("Sorry, I couldn't extract the document data.")

        except Exception as e:
            self.logger.error(f"Error in document message handler: {str(e)}")
            if "RATE_LIMIT_EXCEEDED" in str(e).upper():
                await update.message.reply_text("The service is experiencing high demand. Please try again later.")
            else:
                await self._error_handler(update, context)

    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle document with full processing (legacy method)."""
        user_id = update.effective_user.id
        conversation_id = f"user_{user_id}"
        self.telegram_logger.log_message("Processing document", user_id)

        try:
            if update.effective_chat.type in ["group", "supergroup"]:
                caption = update.message.caption or ""
                caption_entities = []
                if update.message.caption_entities:
                    caption_entities = [
                        {"type": e.type, "offset": e.offset, "length": e.length}
                        for e in update.message.caption_entities
                    ]
                if not BotUsernameHelper.is_bot_mentioned(caption, context, entities=caption_entities):
                    return
                caption = BotUsernameHelper.remove_bot_mention(caption, context)
            else:
                caption = update.message.caption or "Please analyze this document."

            document = update.message.document
            file_name = document.file_name
            file_id = document.file_id
            file_extension = os.path.splitext(file_name)[1][1:] if "." in file_name else ""

            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            status_message = await update.message.reply_text(
                f"Processing your {file_extension.upper()} document..."
            )

            document_file = await context.bot.get_file(file_id)
            file_content = await document_file.download_as_bytearray()
            document_file_obj = io.BytesIO(file_content)

            prompt = caption or f"Please analyze this {file_extension.upper()} file."

            if file_extension.lower() == "pdf":
                response = await self.document_processor.process_document_enhanced(
                    file=document_file_obj, file_extension=file_extension, prompt=prompt
                )
            else:
                response = await self.document_processor.process_document_from_file(
                    file=document_file_obj, file_extension=file_extension, prompt=prompt
                )

            try:
                await status_message.delete()
            except Exception:
                pass

            if response:
                response_text = response.get("result", "Document processed successfully.")
                document_id = response.get("document_id", "Unknown")
                formatted_response = (
                    f"**Document Analysis Completed**\n\n{response_text}\n\n**Document ID:** {document_id}"
                )
                await self.response_formatter.safe_send_message(
                    update.message, formatted_response, disable_web_page_preview=True
                )
            else:
                await update.message.reply_text("Sorry, I couldn't analyze the document. Please try again.")

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            await update.message.reply_text(f"Sorry, I couldn't process your document. Error: {str(e)[:100]}...")

    async def handle_awaiting_doc_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Process text when awaiting document content."""
        if context.user_data.get("awaiting_doc_text"):
            context.user_data["awaiting_doc_text"] = False
            content = update.message.text
            context.user_data["doc_export_text"] = content

            format_options = [
                [
                    InlineKeyboardButton("PDF Format", callback_data="export_format_pdf"),
                    InlineKeyboardButton("DOCX Format", callback_data="export_format_docx"),
                ]
            ]
            await update.message.reply_text(
                f"Text received ({len(content)} characters). Select the export format:",
                reply_markup=InlineKeyboardMarkup(format_options),
            )
            return True
        return False

    async def handle_awaiting_doc_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Process image when awaiting document content."""
        if context.user_data.get("awaiting_doc_image"):
            context.user_data["awaiting_doc_image"] = False
            image_file = update.message.photo[-1].get_file()
            context.user_data["doc_export_image"] = image_file

            format_options = [[InlineKeyboardButton("PDF Format", callback_data="export_format_pdf")]]
            await update.message.reply_text(
                "Select the document format:", reply_markup=InlineKeyboardMarkup(format_options)
            )
            return True
        return False
