# src/utils/file_handler.py

import io
import zipfile
import os
import tempfile
import aiofiles
from pdfminer.high_level import extract_text
from docx import Document

# External services and utilities
from services.gemini_api import GeminiAPI
from utils.telegramlog import TelegramLogger
from services.user_data_manager import UserDataManager

# Caching
from aiocache import cached, Cache

class FileHandler:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_ZIP_SIZE = 50 * 1024 * 1024   # 50 MB
    MAX_CODE_SIZE = 5 * 1024 * 1024   # 5 MB

    def __init__(self, telegram_logger: TelegramLogger, gemini_api: GeminiAPI, user_data_manager: UserDataManager):
        self.telegram_logger = telegram_logger
        self.gemini_api = gemini_api
        self.user_data_manager = user_data_manager

    @cached(ttl=3600, cache=Cache.MEMORY)
    async def analyze_text_cached(self, text: str) -> str:
        return await self.gemini_api.analyze_text(text)

    async def handle_pdf(self, file_content: io.BytesIO, user_id: int) -> str:
        try:
            if self._validate_file_size(file_content, self.MAX_FILE_SIZE):
                text = extract_text(file_content)
                preprocessed_text = self._preprocess_text(text)
                analysis = await self.analyze_text_cached(preprocessed_text)
                await self._update_user_stats(user_id, file_type='pdf')
                return f"📄 **PDF Analysis**\n\n{analysis}"
            else:
                return "❌ PDF file is too large. Maximum allowed size is 10MB."
        except Exception as e:
            self.telegram_logger.log_error(f"PDF processing error: {str(e)}", user_id)
            return "❌ Error processing PDF file."

    async def handle_docx(self, file_content: io.BytesIO, user_id: int) -> str:
        try:
            if self._validate_file_size(file_content, self.MAX_FILE_SIZE):
                # Since aiofiles does not support file-like objects, write to a temporary file first
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(file_content.read())
                    tmp_file_path = tmp_file.name

                async with aiofiles.open(tmp_file_path, 'rb') as f:
                    file_bytes = await f.read()

                document = Document(io.BytesIO(file_bytes))
                text = "\n".join([para.text for para in document.paragraphs])
                preprocessed_text = self._preprocess_text(text)
                analysis = await self.analyze_text_cached(preprocessed_text)
                await self._update_user_stats(user_id, file_type='docx')

                # Clean up the temporary file
                os.remove(tmp_file_path)

                return f"📄 **DOCX Analysis**\n\n{analysis}"
            else:
                return "❌ DOCX file is too large. Maximum allowed size is 10MB."
        except Exception as e:
            self.telegram_logger.log_error(f"DOCX processing error: {str(e)}", user_id)
            return "❌ Error processing DOCX file."

    async def handle_zip(self, file_content: io.BytesIO, user_id: int) -> str:
        try:
            if self._validate_file_size(file_content, self.MAX_ZIP_SIZE):
                with tempfile.TemporaryDirectory(prefix=f'user_{user_id}_') as extract_path:
                    with zipfile.ZipFile(file_content) as zip_ref:
                        self._safe_extract(zip_ref, extract_path)
                    # Further processing of extracted files can be implemented here

                await self._update_user_stats(user_id, file_type='zip')
                return "✅ ZIP file extracted and processed successfully."
            else:
                return "❌ ZIP file is too large. Maximum allowed size is 50MB."
        except zipfile.BadZipFile:
            self.telegram_logger.log_error("Invalid ZIP file format.", user_id)
            return "❌ Invalid ZIP file."
        except Exception as e:
            self.telegram_logger.log_error(f"ZIP processing error: {str(e)}", user_id)
            return "❌ Error processing ZIP file."

    async def handle_code(self, file_content: io.BytesIO, user_id: int, language: str) -> str:
        try:
            if self._validate_file_size(file_content, self.MAX_CODE_SIZE):
                # Since aiofiles does not support file-like objects, write to a temporary file first
                with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as tmp_file:
                    tmp_file.write(file_content.read().decode('utf-8'))
                    tmp_file_path = tmp_file.name

                async with aiofiles.open(tmp_file_path, 'r', encoding='utf-8') as f:
                    code = await f.read()

                # Clean up the temporary file
                os.remove(tmp_file_path)

                preprocessed_code = self._preprocess_code(code)
                analysis = await self.gemini_api.analyze_code(preprocessed_code, language=language)
                await self._update_user_stats(user_id, file_type='code')
                return f"💻 **{language.capitalize()} Code Analysis**\n\n{analysis}"
            else:
                return "❌ Code file is too large. Maximum allowed size is 5MB."
        except UnicodeDecodeError:
            self.telegram_logger.log_error("Unable to decode code file. Ensure it's a UTF-8 encoded text file.", user_id)
            return "❌ Unable to decode code file. Please ensure it's a UTF-8 encoded text file."
        except Exception as e:
            self.telegram_logger.log_error(f"Code processing error: {str(e)}", user_id)
            return "❌ Error processing code file."

    async def handle_additional_file_types(self, file_content: io.BytesIO, user_id: int, file_type: str) -> str:
        """
        Extend this method to handle additional file types as needed.
        """
        if file_type == 'java':
            return await self.handle_code(file_content, user_id, language='java')
        # Add more file type handlers here
        else:
            return "❌ Unsupported file type."

    def _validate_file_size(self, file_content: io.BytesIO, max_size: int) -> bool:
        file_content.seek(0, os.SEEK_END)
        size = file_content.tell()
        file_content.seek(0)
        return size <= max_size

    def _safe_extract(self, zip_ref: zipfile.ZipFile, extract_path: str) -> None:
        """
        Safely extract ZIP files to prevent path traversal attacks.
        """
        for member in zip_ref.namelist():
            member_path = os.path.join(extract_path, member)
            if not os.path.realpath(member_path).startswith(os.path.realpath(extract_path)):
                raise zipfile.BadZipFile("Attempted Path Traversal in ZIP file")
        zip_ref.extractall(extract_path)

    def _preprocess_text(self, text: str) -> str:
        # Implement any text preprocessing steps here
        return text.strip()

    def _preprocess_code(self, code: str) -> str:
        # Implement any code preprocessing steps here
        return code.strip()

    async def _update_user_stats(self, user_id: int, file_type: str) -> None:
        await self.user_data_manager.update_stats(user_id, file_type=file_type)