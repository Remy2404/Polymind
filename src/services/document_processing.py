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
        # Update the SUPPORTED_MIME_TYPES dictionary
    SUPPORTED_MIME_TYPES = {
        'pdf': 'application/pdf',
        'js': 'text/plain',  # Changed to text/plain
        'py': 'text/plain',  # Changed to text/plain
        'txt': 'text/plain',
        'html': 'text/html',
        'css': 'text/css',
        'md': 'text/markdown',
        'csv': 'text/csv',
        'xml': 'text/xml',
        'rtf': 'text/rtf',
        # Add defaults for common programming languages
        'java': 'text/plain',
        'cpp': 'text/plain',
        'c': 'text/plain',
        'cs': 'text/plain',
        'php': 'text/plain',
        'rb': 'text/plain',
        'go': 'text/plain',
        'swift': 'text/plain',
        'kt': 'text/plain',
        'rs': 'text/plain',
        'ts': 'text/plain',
        'sql': 'text/plain',
        'sh': 'text/plain',
        'yaml': 'text/plain',
        'json': 'text/plain',
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

    def get_mime_type(self, file_extension: str) -> str:
        """Get the MIME type for a given file extension with fallback to text/plain for code files."""
        file_extension = file_extension.lower().strip('.')
        
        # Check if it's a recognized code file extension
        if file_extension in self.CODE_EXTENSIONS:
            return 'text/plain'
        
        # Try to get from supported mime types
        mime_type = self.SUPPORTED_MIME_TYPES.get(file_extension)
        if isinstance(mime_type, list):
            return mime_type[0]  # Return first MIME type if multiple are available
        
        # If no specific mapping and it's text-like, default to text/plain
        if file_extension in ['txt', 'log', 'ini', 'cfg', 'conf', 'properties']:
            return 'text/plain'
        
        return mime_type or 'text/plain'  # Default to text/plain if all else fails

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

    async def process_document_from_file(self, file, file_extension: str, prompt: str) -> str:
        """Process a document from a file object."""
        try:
            # Convert BytesIO to bytes if needed
            if hasattr(file, 'read'):
                # If file is a file-like object (like BytesIO), read its content
                file_content = file.read()
                if isinstance(file_content, bytearray):
                    file_content = bytes(file_content)
            elif isinstance(file, bytearray):
                # If file is directly a bytearray, convert to bytes
                file_content = bytes(file)
            else:
                # Otherwise use as is
                file_content = file
                
            # Fixed method name: _get_mime_type → get_mime_type
            mime_type = self.get_mime_type(file_extension)
            
            # Now continue with document processing
            file_io = io.BytesIO(file_content)
            file_io.seek(0)  # Reset file pointer to beginning
            
            # Upload file to Gemini API
            uploaded_file = await self.upload_file(
                file=file_io,
                mime_type=mime_type
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                [uploaded_file, prompt],
                generation_config=self.generation_config
            )
            
            # Clean up
            try:
                await asyncio.to_thread(genai.delete_file, uploaded_file)
            except Exception as e:
                self.logger.warning(f"Failed to delete uploaded file: {e}")
            
            if hasattr(response, 'text') and response.text:
                return response.text
            elif isinstance(response, str):
                return response
            return "Sorry, I couldn't process the document properly."
                
        except Exception as e:
            logging.error(f"Error processing document: {str(e)}")
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
        
    async def process_document_enhanced(self, file, file_extension: str, prompt: str) -> str:
        """
        Process documents with enhanced capabilities, including code analysis.
        """
        try:
            # Convert BytesIO to bytes if needed
            if hasattr(file, 'read'):
                file_content = file.read()
                if isinstance(file_content, bytearray):
                    file_content = bytes(file_content)
            elif isinstance(file, bytearray):
                file_content = bytes(file)
            else:
                file_content = file
            
            file_extension = file_extension.lower().strip('.')
            
            # Enhanced code file handling
            if file_extension in self.CODE_EXTENSIONS:
                mime_type = 'text/plain'
                
                # Add a header to identify the programming language
                header = f"# Code file: {file_extension.upper()}\n\n"
                file_content = header.encode('utf-8') + file_content
                
                # Create a prompt that guides the model to analyze the code
                enhanced_prompt = self._get_language_specific_prompt(file_extension, f"""
                This is a {file_extension.upper()} code file. Please analyze it with the following focus:
                - Understand the structure and functionality
                - Identify key components, classes, and functions
                - Highlight any potential issues or improvements
                - Suggest optimizations or best practices
                
                User's instructions: {prompt}
                """)
                
                file_io = io.BytesIO(file_content)
                file_io.seek(0)
                
                uploaded_file = await self.upload_file(file=file_io, mime_type=mime_type)
                
                # Generate response
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    [uploaded_file, {"text": enhanced_prompt}],
                    generation_config=self.generation_config
                )
                
                # Clean up
                try:
                    await asyncio.to_thread(genai.delete_file, uploaded_file)
                except Exception as e:
                    self.logger.warning(f"Failed to delete uploaded file: {e}")
                
                if hasattr(response, 'text') and response.text:
                    return response.text
                return f"Sorry, I couldn't analyze the {file_extension.upper()} code properly."
                
            # Enhanced PDF handling (existing code)
            elif file_extension == 'pdf':
                mime_type = 'application/pdf'
                
                # Create a prompt that guides the model to leverage its enhanced capabilities
                enhanced_prompt = f"""
                Process this PDF document with the following focus:
                - Preserve and understand the document layout 
                - Analyze any charts, tables, or diagrams in the document
                - Extract structured information where relevant
                
                User's instructions: {prompt}
                """
                
                # Upload file to Gemini
                uploaded_file = await self.upload_file(file=file_content, mime_type=mime_type)
                
                # Use Gemini 1.5 Pro for best document processing results
                pro_model = genai.GenerativeModel("gemini-1.5-pro")
                
                # Generate response with enhanced configuration
                response = await asyncio.to_thread(
                    pro_model.generate_content,
                    [uploaded_file, {"text": enhanced_prompt}],
                    generation_config={
                        "temperature": 0.2,  # Lower for more factual responses
                        "top_p": 0.95,
                        "top_k": 64,
                        "max_output_tokens": 8192,  # Allow longer responses for document analysis
                        "response_mime_type": "text/plain"
                    }
                )
                
                # Clean up the uploaded file
                try:
                    await asyncio.to_thread(genai.delete_file, uploaded_file)
                except Exception as e:
                    self.logger.warning(f"Failed to delete uploaded file: {e}")
                
                if hasattr(response, 'text') and response.text:
                    return response.text
                else:
                    return "Sorry, I couldn't analyze the document properly."
                    
            # Use existing implementation for other file types
            else:
                return await self.process_document_from_file(file, file_extension, prompt)
                
        except Exception as e:
            self.logger.error(f"Error in enhanced document processing: {str(e)}")
            raise ValueError(f"Document processing failed: {str(e)}")

    # Add this method to provide language-specific guidance
    def _get_language_specific_prompt(self, file_extension: str, base_prompt: str) -> str:
        """Get language-specific analysis prompts based on file extension."""
        language_prompts = {
            'py': """
                Python-specific analysis:
                - Check for PEP 8 compliance
                - Identify use of common libraries (NumPy, Pandas, Flask, etc.)
                - Look for Pythonic patterns and anti-patterns
            """,
            'js': """
                JavaScript-specific analysis:
                - Identify the framework if any (React, Vue, Angular, etc.)
                - Check for modern JS features and patterns
                - Look for potential performance issues
            """,
            'java': """
                Java-specific analysis:
                - Identify design patterns used
                - Check for proper exception handling
                - Review class structure and inheritance
            """,
            'cpp': """
                C++ specific analysis:
                - Check for memory management issues
                - Look for efficient use of STL
                - Identify potential performance bottlenecks
            """
        }
        
        language_specific = language_prompts.get(file_extension.lower(), "")
        if language_specific:
            return f"{base_prompt}\n\n{language_specific}"
        return base_prompt