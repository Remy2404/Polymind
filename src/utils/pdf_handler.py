# utils/pdf_handler.py

import io
import asyncio
from typing import List
from PyPDF2 import PdfReader
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ContextTypes,
    MessageHandler,
    filters,
)
from utils.telegramlog import telegram_logger
from services.gemini_api import GeminiAPI
from handlers.text_handlers import TextHandler
import pytesseract
from PIL import Image
import logging
import html
import re
""" from telegram.utils.helpers import escape_markdown """



# Configure logging for PDFHandler
logger = logging.getLogger(__name__)


class PDFHandler:
    def __init__(self, text_handler: TextHandler = None, telegram_logger=None):
        """
        Initialize the PDFHandler.

        Args:
            text_handler (TextHandler, optional): Instance of TextHandler. Defaults to None.
            telegram_logger (TelegramLogger, optional): Instance of TelegramLogger. Defaults to global telegram_logger.
        """
        self.gemini_api = GeminiAPI(vision_model=None, rate_limiter=None)
        self.text_handler = text_handler
        self.telegram_logger = telegram_logger if telegram_logger else telegram_logger
        self.pdf_content = {}  # Stores PDF content per user
        self.conversation_history = {}

    async def handle_pdf(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_id = update.effective_user.id
        if update.message.document and update.message.document.mime_type == 'application/pdf':
            try:
                file = await context.bot.get_file(update.message.document.file_id)
                file_content = io.BytesIO(await file.download_as_bytearray())
                extracted_text = self.extract_text_from_pdf(file_content, user_id)
    
                self.pdf_content[user_id] = extracted_text
                self.conversation_history[user_id] = []
    
                keyboard = [
                    [InlineKeyboardButton("Reset Conversation", callback_data="reset_conversation")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
    
                # Use html.escape instead of escape_markdown
                success_message = html.escape("📄 PDF uploaded successfully! You can now ask questions about it.")
    
                await update.message.reply_text(
                    success_message,
                    reply_markup=reply_markup,
                    parse_mode='HTML'
                )
            except Exception as e:
                error_message = f"Error processing PDF: {str(e)}"
                self.telegram_logger.log_error(f"Error processing PDF for user {user_id}: {error_message}")
                
                # Use html.escape instead of escape_markdown
                safe_error_message = html.escape("Error processing PDF: Please ensure the PDF is not corrupted and try again.")
                
                await update.message.reply_text(safe_error_message, parse_mode='HTML')
        else:
            await update.message.reply_text("Please send a valid PDF document.")

    def extract_text_from_pdf(self, file_content: io.BytesIO, user_id: int = None) -> str:
        """Extract text from a PDF file, using OCR for scanned images."""
        try:
            reader = PdfReader(file_content)
            text = ""
            for page_number, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text()
                if extracted:
                    text += extracted
                else:
                    # Attempt OCR if no text is found
                    self.telegram_logger.log_error(f"No text found on page {page_number}. Attempting OCR.", user_id)
                    # Extract images from the page for OCR
                    images = page.images
                    if images:
                        for image_index, img in enumerate(images, start=1):
                            try:
                                image_data = img.data  # Assuming img.data contains image bytes
                                image = Image.open(io.BytesIO(image_data))
                                ocr_text = pytesseract.image_to_string(image)
                                text += ocr_text + "\n"
                            except Exception as ocr_e:
                                self.telegram_logger.log_error(f"OCR failed on page {page_number}, image {image_index}: {str(ocr_e)}", user_id)
                    else:
                        self.telegram_logger.log_error(f"No images found on page {page_number} for OCR.", user_id)
            return text
        except Exception as e:
            self.telegram_logger.log_error(f"Text extraction error: {str(e)}", user_id)
            return ""

    async def process_caption_with_pdf(self, pdf_text: str, caption: str) -> str:
        """Process the caption as a question or instruction related to the uploaded PDF."""
        try:
            prompt = f"{caption}\n\n{pdf_text[:4000]}"  # Limit the context if necessary
            response = await self.gemini_api.generate_response(prompt)
            if not response:
                raise ValueError("Gemini API returned an empty response.")
            return response
        except Exception as e:
            self.telegram_logger.log_error(f"Error processing caption: {str(e)}", None)
            return "❌ An error occurred while processing your caption."

    async def ask_pdf_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle questions related to the uploaded PDF."""
        user_id = update.effective_user.id
        if user_id not in self.pdf_content:
            await update.message.reply_text(
                "⚠️ You don't have any PDF content stored. Please upload a PDF first."
            )
            return

        question = update.message.text
        try:
            answer = await self.text_handler.answer_question(self.pdf_content[user_id]["content"], question)
            self.conversation_history[user_id].append({"question": question, "answer": answer})
            # Escape answer before sending
            escaped_answer = self.escape_markdown_v2(answer)
            await update.message.reply_text(escaped_answer, parse_mode="MarkdownV2")
            self.telegram_logger.log_message(f"Answered question for user {user_id}: {question}", user_id)
        except Exception as e:
            await update.message.reply_text("❌ An error occurred while processing your question.")
            self.telegram_logger.log_error(f"Error answering question for user {user_id}: {e}", user_id)

    def get_handlers(self):
        """Register all necessary handlers for PDF processing."""
        return [
            MessageHandler(filters.Document.PDF, self.handle_pdf),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.ask_pdf_question),
        ]

    @staticmethod
    def escape_markdown_v2(text: str) -> str:
        """Escapes special characters for Telegram MarkdownV2."""
        escape_chars = r'_*\[\]()~`>#+-=|{}.!'
        escaped_text = ""
        for char in text:
            if char in escape_chars:
                escaped_text += f'\\{char}'
            else:
                escaped_text += char
        return escaped_text