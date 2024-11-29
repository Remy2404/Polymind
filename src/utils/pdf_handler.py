import io
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from telegram.ext import (
    ContextTypes,
    ConversationHandler,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)
from utils.telegramlog import telegram_logger
from services.gemini_api import GeminiAPI
from handlers.text_handlers import TextHandler

PDF_CONVERSATION = range(1)

class PDFHandler:
    def __init__(self, gemini_api: GeminiAPI, text_handler: TextHandler):
        self.gemini_api = gemini_api
        self.text_handler = text_handler
        self.pdf_content = {}
        self.conversation_history = {}

    def extract_text_from_pdf(self, file_content: io.BytesIO) -> str:
        output = io.StringIO()
        extract_text_to_fp(file_content, output, laparams=LAParams(), output_type='text', codec=None)
        return output.getvalue()

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

                # Send confirmation with Reset button
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

    async def ask_pdf_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        message: Message = update.message

        if user_id not in self.pdf_content:
            await message.reply_text("⚠️ You don't have any PDF uploaded. Please upload a PDF first.")
            return

        # Optional: Check if the message is a reply to the PDF upload
        if message.reply_to_message and message.reply_to_message.document:
            question = message.text
            await message.reply_text("🔍 Processing your question...")
            try:
                answer = await self.text_handler.answer_question(self.pdf_content[user_id], question)
                await message.reply_text(answer)
                telegram_logger.log_message(f"Answered question for user {user_id}: {question}", user_id)
            except Exception as e:
                await message.reply_text("❌ An error occurred while processing your question.")
                telegram_logger.error(f"Error answering question for user {user_id}: {e}")
        else:
            await message.reply_text("ℹ️ Please reply to the PDF message with your question.")

    async def handle_reset_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        user_id = query.from_user.id
        await query.answer()

        # Reset the conversation history
        if user_id in self.conversation_history:
            self.conversation_history[user_id] = []
            await query.edit_message_text("🧹 Conversation has been reset.")
            telegram_logger.log_message(f"Conversation reset for user {user_id}", user_id)
        else:
            await query.edit_message_text("ℹ️ No active conversation to reset.")

    def get_handlers(self):
        return [
            MessageHandler(filters.Document.PDF, self.handle_pdf_upload),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.ask_pdf_question),
            CallbackQueryHandler(self.handle_reset_conversation, pattern="^reset_conversation$"),
        ]