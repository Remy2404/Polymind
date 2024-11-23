import io
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from telegram import Update
from telegram.ext import ContextTypes, ConversationHandler, CommandHandler, MessageHandler, filters
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

    async def start_pdf_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id = update.effective_user.id
        if user_id in self.pdf_content:
            await update.message.reply_text("You can now ask questions about the PDF you uploaded. Type /end_pdf when you're done.")
            self.conversation_history[user_id] = []
            return PDF_CONVERSATION
        else:
            await update.message.reply_text("Please upload a PDF first before starting a conversation about it.")
            return ConversationHandler.END

    async def handle_pdf_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id = update.effective_user.id
        question = update.message.text

        if user_id not in self.pdf_content:
            await update.message.reply_text("I'm sorry, but I don't have any PDF content stored for you. Please upload a PDF first.")
            return ConversationHandler.END

        pdf_text = self.pdf_content[user_id]
        conversation_history = self.conversation_history[user_id]

        # Prepare the prompt with conversation history
        prompt = f"Based on the following PDF content and our conversation history, answer this question: {question}\n\n"
        prompt += f"PDF content: {pdf_text[:4000]}\n\n"
        prompt += "Conversation history:\n"
        for entry in conversation_history[-5:]:  # Include last 5 exchanges
            prompt += f"{entry['role']}: {entry['content']}\n"

        response = await self.gemini_api.generate_response(prompt)

        # Format the response with markdown
        formatted_response = await self.text_handler.format_telegram_markdown(response)

        await update.message.reply_text(formatted_response, parse_mode='MarkdownV2')

        # Update conversation history
        conversation_history.append({"role": "Human", "content": question})
        conversation_history.append({"role": "AI", "content": response})

        return PDF_CONVERSATION

    async def end_pdf_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user_id = update.effective_user.id
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
        await update.message.reply_text("PDF conversation ended. You can start a new one with /start_pdf_conversation")
        return ConversationHandler.END

    def get_conversation_handler(self):
        return ConversationHandler(
            entry_points=[CommandHandler("start_pdf_conversation", self.start_pdf_conversation)],
            states={
                PDF_CONVERSATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_pdf_question)]
            },
            fallbacks=[CommandHandler("end_pdf", self.end_pdf_conversation)]
        )


    async def handle_pdf(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle PDF documents."""
        user_id = update.effective_user.id
        telegram_logger.log_message(f"Processing PDF for user {user_id}", user_id)
    
        try:
            # Get the file
            file = await context.bot.get_file(update.message.document.file_id)
    
            # Download the file
            pdf_bytes = await file.download_as_bytearray()
    
            # Extract text from PDF
            output_string = io.StringIO()
            with io.BytesIO(pdf_bytes) as pdf_file:
                extract_text_to_fp(pdf_file, output_string, laparams=LAParams())
    
            pdf_text = output_string.getvalue()
    
            if not pdf_text.strip():
                raise ValueError("The PDF appears to be empty or unreadable.")
    
            # Store the PDF content for this user
            self.pdf_content[user_id] = pdf_text
    
            # Set the state to indicate that a PDF has been uploaded
            context.user_data['pdf_uploaded'] = True
    
            # Generate a summary using Gemini API
            prompt = f"Summarize the following PDF content in a concise manner:\n\n{pdf_text[:4000]}"  # Limit to first 4000 characters
            summary = await self.gemini_api.generate_response(prompt)
    
            if not summary.strip():
                raise ValueError("Failed to generate a summary from the PDF content.")
    
            # Format the summary with markdown
            formatted_summary = await self.text_handler.format_telegram_markdown(summary)
    
            await update.message.reply_text(
                f"PDF processed successfully\\. Here's a summary:\n\n{formatted_summary}\n\nYou can now ask questions about this PDF\\.",
                parse_mode='MarkdownV2'
            )
    
        except ValueError as ve:
            error_message = f"Error processing PDF: {str(ve)}"
            telegram_logger.log_error(error_message, user_id)
            await update.message.reply_text(f"An error occurred: {str(ve)}\\. Please try a different PDF\\.", parse_mode='MarkdownV2')
        except Exception as e:
            error_message = f"Unexpected error processing PDF: {str(e)}"
            telegram_logger.log_error(error_message, user_id)
            await update.message.reply_text("An unexpected error occurred while processing your PDF\\. Please try again or contact support\\.", parse_mode='MarkdownV2')

    async def handle_pdf_followup(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle follow-up questions about the PDF content."""
        user_id = update.effective_user.id
        question = update.message.text
    
        if user_id not in self.pdf_content:
            await update.message.reply_text(
                "I'm sorry, but I don't have any PDF content stored for you\\. "
                "Please upload a PDF first using the /upload\\_pdf command\\.",
                parse_mode='MarkdownV2'
            )
            return
    
        pdf_text = self.pdf_content[user_id]
    
        # Prepare the prompt with the question and PDF content
        prompt = (
            f"Based on the following PDF content, answer this question: {question}\n\n"
            f"PDF content (first 4000 characters): {pdf_text[:4000]}"
        )
    
        try:
            # Generate response using Gemini API
            response = await self.gemini_api.generate_response(prompt)
    
            # Format the response with markdown
            formatted_response = await self.text_handler.format_telegram_markdown(response)
    
            # Send the response to the user
            await update.message.reply_text(formatted_response, parse_mode='MarkdownV2')
    
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            telegram_logger.log_error(error_message, user_id)
            await update.message.reply_text(
                "I'm sorry, but I encountered an error while processing your question\\. "
                "Please try again or contact support if the problem persists\\.",
                parse_mode='MarkdownV2'
            )
    
        # Log the interaction
        telegram_logger.log_message(f"PDF question handled for user {user_id}", user_id)

    async def clear_pdf_content(self, user_id: int):
        """Clear the stored PDF content for a user."""
        if user_id in self.pdf_content:
            del self.pdf_content[user_id]
            telegram_logger.log_message(f"PDF content cleared for user {user_id}", user_id)
        else:
            telegram_logger.log_message(f"No PDF content to clear for user {user_id}", user_id)

    async def get_pdf_summary(self, user_id: int) -> str:
        """Get a summary of the stored PDF content for a user."""
        if user_id not in self.pdf_content:
            return "No PDF content available."
        
        pdf_text = self.pdf_content[user_id]
        prompt = f"Provide a brief summary of the following PDF content:\n\n{pdf_text[:4000]}"
        
        try:
            summary = await self.gemini_api.generate_response(prompt)
            return summary
        except Exception as e:
            error_message = f"Error generating PDF summary: {str(e)}"
            telegram_logger.log_error(error_message, user_id)
            return "Unable to generate summary due to an error."

    def get_pdf_content_size(self, user_id: int) -> int:
        """Get the size of the stored PDF content for a user."""
        if user_id in self.pdf_content:
            return len(self.pdf_content[user_id])
        return 0

    async def handle_pdf_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle requests for information about the stored PDF."""
        user_id = update.effective_user.id
        
        if user_id not in self.pdf_content:
            await update.message.reply_text(
                "You don't have any PDF content stored\\. "
                "Please upload a PDF first using the /upload\\_pdf command\\.",
                parse_mode='MarkdownV2'
            )
            return
        
        content_size = self.get_pdf_content_size(user_id)
        summary = await self.get_pdf_summary(user_id)
        
        info_message = (
            f"PDF Information:\n"
            f"- Content size: {content_size} characters\n"
            f"- Summary: {summary[:1000]}..."  # Truncate summary if it's too long
        )
        
        formatted_info = await self.text_handler.format_telegram_markdown(info_message)
        await update.message.reply_text(formatted_info, parse_mode='MarkdownV2')