import logging
import google.generativeai as genai
import io
import httpx
from typing import Optional, List, Dict, BinaryIO
import asyncio
from utils.telegramlog import telegram_logger
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class DocumentProcessor:
    """Handles document processing using the Gemini API."""

    # Supported MIME types for document processing
    SUPPORTED_MIME_TYPES = {
        'pdf': 'application/pdf',
        'js': ['application/x-javascript', 'text/javascript'],
        'py': ['application/x-python', 'text/x-python'],
        'txt': 'text/plain',
        'html': 'text/html',
        'css': 'text/css',
        'md': 'text/markdown',
        'csv': 'text/csv',
        'xml': 'text/xml',
        'rtf': 'text/rtf'
    }

    # Code file extensions that can be converted to text
    CODE_EXTENSIONS = {
        'py', 'js', 'java', 'cpp', 'c', 'cs', 'php', 'rb', 'go',
        'swift', 'kt', 'rs', 'ts', 'html', 'css', 'sql', 'sh',
        'yaml', 'json', 'xml', 'md'
    }

    def __init__(self):
        """Initialize the DocumentProcessor with Gemini API configuration."""
        self.logger = logging.getLogger(__name__)

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found or empty")

        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-1.5-pro")

        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 4096,
        }

    async def upload_file(self, file: BinaryIO, mime_type: str) -> str:
        """Asynchronously upload a file to the Gemini API."""
        try:
            loop = asyncio.get_event_loop()
            uploaded_file = await loop.run_in_executor(
                None,
                lambda: genai.upload_file(file, mime_type=mime_type)
            )
            return uploaded_file
        except Exception as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            raise

    def get_mime_type(self, file_extension: str) -> Optional[str]:
        """Get the MIME type for a given file extension."""
        mime_type = self.SUPPORTED_MIME_TYPES.get(file_extension.lower().strip('.'))
        if isinstance(mime_type, list):
            return mime_type[0]  # Return first MIME type if multiple are available
        return mime_type

    async def process_document_from_url(self, document_url: str, prompt: str) -> str:
        """Process a document from a URL using the Gemini API."""
        try:
            # Get file extension from URL
            file_extension = document_url.split('.')[-1]
            mime_type = self.get_mime_type(file_extension)
            if not mime_type:
                raise ValueError(f"Unsupported file type: {file_extension}")

            # Retrieve document from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(document_url)
                response.raise_for_status()
                doc_data = io.BytesIO(response.content)

            # Upload document using File API with mime_type
            uploaded_file = await self.upload_file(
                file=doc_data,
                mime_type=mime_type
            )

            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                [uploaded_file, prompt],
                generation_config=self.generation_config
            )

            return response.text
        except Exception as e:
            self.logger.error(f"Error processing document from URL: {str(e)}")
            raise

    def _convert_code_to_text(self, file: BinaryIO, file_extension: str) -> tuple[io.BytesIO, str]:
        """Convert code file to text format."""
        try:
            content = file.read().decode('utf-8')
            
            # Add file extension as a header
            header = f"// File type: {file_extension}\n\n"
            formatted_content = header + content
            
            # Convert back to BytesIO with text/plain mime type
            text_file = io.BytesIO(formatted_content.encode('utf-8'))
            return text_file, 'text/plain'
        except Exception as e:
            self.logger.error(f"Error converting code to text: {str(e)}")
            raise

    async def process_document_from_file(self, file: BinaryIO, file_extension: str, prompt: str) -> str:
        """Process a document file using the Gemini API."""
        try:
            file_extension = file_extension.lower().strip('.')
            
            # Handle code files by converting them to text
            if file_extension in self.CODE_EXTENSIONS:
                text_file, mime_type = self._convert_code_to_text(file, file_extension)
            else:
                mime_type = self.get_mime_type(file_extension)
                if not mime_type:
                    supported_formats = (
                        list(self.SUPPORTED_MIME_TYPES.keys()) + 
                        list(self.CODE_EXTENSIONS)
                    )
                    raise ValueError(
                        f"Unsupported file type: {file_extension}. "
                        f"Supported formats are: {', '.join(supported_formats)}"
                    )
                text_file = file

            # Upload file
            uploaded_file = await self.upload_file(file=text_file, mime_type=mime_type)

            # Generate content
            response = await asyncio.to_thread(
                self.model.generate_content,
                [uploaded_file, {"text": prompt}],
                generation_config=self.generation_config
            )

            # Clean up the uploaded file
            try:
                await asyncio.to_thread(
                    genai.delete_file, uploaded_file
                )
            except Exception as e:
                self.logger.warning(f"Failed to delete uploaded file: {e}")

            if hasattr(response, 'text') and response.text:
                return response.text
            elif isinstance(response, str) and response:
                return response
            return "Sorry, I couldn't process the document content."

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise

    async def process_multiple_documents(self, documents: List[Dict], prompt: str) -> str:
        """Process multiple documents simultaneously."""
        try:
            uploaded_docs = []

            for doc in documents:
                if 'url' in doc:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(doc['url'])
                        response.raise_for_status()
                        doc_data = io.BytesIO(response.content)
                else:
                    doc_data = doc['file']

                mime_type = self.get_mime_type(doc['extension'])
                if not mime_type:
                    raise ValueError(f"Unsupported file type: {doc['extension']}")

                uploaded_doc = await self.upload_file(
                    file=doc_data,
                    mime_type=mime_type
                )
                uploaded_docs.append(uploaded_doc)

            contents = uploaded_docs + [prompt]

            response = await asyncio.to_thread(
                self.model.generate_content,
                contents,
                generation_config=self.generation_config
            )

            if hasattr(response, 'text') and response.text:
                return response.text
            elif isinstance(response, str) and response:
                return response
            return "Sorry, I couldn't process the documents."

        except Exception as e:
            self.logger.error(f"Error processing multiple documents: {str(e)}")
            raise

    async def delete_processed_file(self, file_name: str) -> bool:
        """Delete a processed file from Gemini API storage."""
        try:
            await asyncio.to_thread(genai.delete_file, file_name)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting file {file_name}: {str(e)}")
            return False

    async def list_processed_files(self) -> List[str]:
        """List all files currently stored in Gemini API storage."""
        try:
            files = await asyncio.to_thread(genai.list_files)
            return [f.name for f in files]
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            return []