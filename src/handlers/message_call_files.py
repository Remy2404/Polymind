# src/handlers/message_call_files.py

import io
from telegram.ext import MessageHandler, filters , ContextTypes
from telegram import Update
from utils.fileHandler import FileHandler
from services.user_data_manager import UserDataManager

class MessageCallFiles:
    def __init__(self, filehandler: FileHandler, user_data_manager: UserDataManager):
        self.file_handler = filehandler
        self.user_data_manager = user_data_manager

    async def process_pdf(self, filecontent: io.BytesIO, user_id: int) -> str:
        return await self.file_handler.handle_pdf(filecontent, user_id)
    
    async def process_docx(self, filecontent: io.BytesIO, user_id: int) -> str:
        return await self.file_handler.handle_docx(filecontent, user_id)
    
    async def process_zip(self, filecontent: io.BytesIO, user_id: int) -> str:
        return await self.file_handler.handle_zip(filecontent, user_id)
    
    async def process_code(self, filecontent: io.BytesIO, user_id: int, language: str) -> str:
        return await self.file_handler.handle_code(filecontent, user_id, language)
    
    async def process_additional_file_types(self, filecontent: io.BytesIO, user_id: int, file_type: str) -> str:
        return await self.file_handler.handle_additional_file_types(filecontent, user_id, file_type)
    
    # Example for handling Java files
    async def process_java(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            document = update.message.document
            file = await context.bot.get_file(document.file_id)
            file_content = io.BytesIO()
            await file.download_to_drive(file_content)
            user_id = update.effective_user.id
            result = await self.file_handler.handle_code(file_content, user_id, language='java')
            await update.message.reply_text(result)
        except Exception as e:
            await update.message.reply_text("Error processing Java file")
            raise

    def register_handlers(self, application):
        """Register file processing handlers with the application."""
        # Fixed: Using correct MimeType attribute
        java_handler = MessageHandler(
            filters.Document.MimeType(['application/java', 'text/x-java-source']),
            self.process_java
        )
        application.add_handler(java_handler)
        
        # Add more handlers for different file types as needed