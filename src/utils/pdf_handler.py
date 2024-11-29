import io
from PyPDF2 import PdfReader
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)
from utils.telegramlog import telegram_logger
from services.gemini_api import GeminiAPI
from handlers.text_handlers import TextHandler

class PDFHandler:
    def __init__(self, gemini_api: GeminiAPI, text_handler: TextHandler):
        self.gemini_api = gemini_api
        self.text_handler = text_handler
        self.pdf_content = {}
        self.conversation_history = {}

    def extract_text_from_pdf(self, file_content: io.BytesIO) -> str:
        pdf_reader = PdfReader(file_content)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    async def handle_pdf_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if update.message.document and update.message.document.mime_type == 'application/pdf':
            try:
                file = await context.bot.get_file(update.message.document.file_id)
                file_bytes = await file.download_as_bytearray()
                file_content = io.BytesIO(file_bytes)
                extracted_text = self.extract_text_from_pdf(file_content)
                self.pdf_content[user_id] = extracted_text
                self.conversation_history[user_id] = []

                keyboard = [
                    [InlineKeyboardButton("Reset Conversation", callback_data="reset_conversation")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)

                await update.message.reply_text(
                    "📄 PDF uploaded successfully! You can now ask questions about it.",
                    reply_markup=reply_markup
                )
                telegram_logger.log_message(f"PDF uploaded and processed for user {user_id}", user_id)
            except Exception as e:
                await update.message.reply_text("❌ Failed to process the PDF. Please try again.")
                telegram_logger.error(f"Error processing PDF for user {user_id}: {e}")
        else:
            await update.message.reply_text("⚠️ Please upload a valid PDF file.")
            telegram_logger.log_message(f"User {user_id} tried to upload an invalid PDF file", user_id)

    async def handle_pdf_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id

        if user_id not in self.pdf_content:
            await update.message.reply_text(
                "⚠️ You don't have any PDF uploaded. Please upload a PDF first."
            )
            return

        content = self.pdf_content[user_id]
        content_size = len(content)
        summary = await self.get_pdf_summary(user_id)

        info_message = (
            f"📄 **PDF Information:**\n"
            f"- **Content Size:** {content_size} characters\n"
            f"- **Summary:** {summary[:1000]}..."
        )

        keyboard = [
            [InlineKeyboardButton("Reset Conversation", callback_data="reset_conversation")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            info_message,
            parse_mode='MarkdownV2',
            reply_markup=reply_markup
        )
        telegram_logger.log_message(f"Provided PDF info to user {user_id}", user_id)

    async def get_pdf_summary(self, user_id: int) -> str:
        content = self.pdf_content.get(user_id, "")
        if not content:
            return "No PDF content available."

        prompt = f"Provide a brief summary of the following PDF content:\n\n{content[:4000]}"

        try:
            summary = await self.gemini_api.generate_response(prompt)
            return summary
        except Exception as e:
            error_message = f"Error generating PDF summary: {str(e)}"
            telegram_logger.error(error_message)
            return "Unable to generate summary due to an error."

    async def ask_pdf_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if user_id not in self.pdf_content:
            await update.message.reply_text(
                "⚠️ You don't have any PDF content stored. Please upload a PDF first."
            )
            return

        question = update.message.text
        try:
            answer = await self.text_handler.answer_question(self.pdf_content[user_id], question)
            self.conversation_history[user_id].append({"question": question, "answer": answer})
            await update.message.reply_text(answer)
            telegram_logger.log_message(f"Answered question for user {user_id}: {question}", user_id)
        except Exception as e:
            await update.message.reply_text("❌ An error occurred while processing your question.")
            telegram_logger.error(f"Error answering question for user {user_id}: {e}")

    async def handle_reset_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        user_id = query.from_user.id
        await query.answer()

        if user_id in self.conversation_history:
            self.conversation_history[user_id] = []
            await query.edit_message_text("🧹 Conversation has been reset.")
            telegram_logger.log_message(f"Conversation reset for user {user_id}", user_id)
        else:
            await query.edit_message_text("ℹ️ No active conversation to reset.")

    def get_handlers(self):
        return [
            MessageHandler(filters.Document.PDF, self.handle_pdf_upload),
            CommandHandler("pdf_info", self.handle_pdf_info),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.ask_pdf_question),
            CallbackQueryHandler(self.handle_reset_conversation, pattern="^reset_conversation$"),
        ]