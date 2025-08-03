"""
Modern Document Processor for Gemini 2.0 Flash
Handles document processing with multimodal capabilities
Integrated with the new GeminiAPI
"""

import logging
import io
from typing import Optional, List, Dict, Any, Union

from services.gemini_api import (
    GeminiAPI,
    ProcessingResult,
    create_document_input,
)


class DocumentProcessor:
    """
    Enhanced document processor using Gemini 2.0 Flash
    Supports multimodal document analysis with images, text, and structured data
    """

    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)

        # Supported document types
        self.supported_extensions = {
            # Documents
            "pdf",
            "doc",
            "docx",
            "ppt",
            "pptx",
            "xls",
            "xlsx",
            # Text formats
            "txt",
            "csv",
            "html",
            "md",
            "json",
            "xml",
            "rtf",
            # Code files
            "py",
            "js",
            "ts",
            "java",
            "cpp",
            "c",
            "cs",
            "php",
            "rb",
            "go",
            "rs",
            "sql",
            "sh",
            "yaml",
            "yml",
        }

    async def process_document(
        self,
        file_data: Union[bytes, io.BytesIO],
        filename: str,
        prompt: str = "Analyze this document and provide a comprehensive summary.",
        context: Optional[List[Dict]] = None,
    ) -> ProcessingResult:
        """
        Process a single document with Gemini 2.0 Flash

        Args:
            file_data: Document data as bytes or BytesIO
            filename: Original filename with extension
            prompt: Analysis prompt
            context: Conversation context

        Returns:
            ProcessingResult with analysis
        """
        try:
            # Create document input
            doc_input = create_document_input(file_data, filename)

            # Enhanced prompt for document analysis
            enhanced_prompt = f"""
            Analyze this document comprehensively:
            
            1. **Document Overview**: Identify the document type, structure, and main purpose
            2. **Key Content**: Extract and summarize the most important information
            3. **Entities & Topics**: Identify people, organizations, places, dates, and main topics
            4. **Structure Analysis**: Describe tables, lists, sections, and formatting
            5. **Technical Details**: For code files, analyze functionality, dependencies, and structure
            
            User Request: {prompt}
            
            Provide a well-structured response with clear sections and bullet points where appropriate.
            """

            # Process with Gemini
            result = await self.gemini_api.process_multimodal_input(
                text_prompt=enhanced_prompt, media_inputs=[doc_input], context=context
            )

            if result.success:
                self.logger.info(f"Successfully processed document: {filename}")
            else:
                self.logger.error(
                    f"Failed to process document {filename}: {result.error}"
                )

            return result

        except Exception as e:
            error_msg = f"Document processing failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(success=False, error=error_msg)

    async def process_multiple_documents(
        self,
        files: List[Dict[str, Any]],
        prompt: str = "Analyze these documents and provide a comparative summary.",
        context: Optional[List[Dict]] = None,
    ) -> ProcessingResult:
        """
        Process multiple documents in a single request

        Args:
            files: List of file dictionaries with 'data', 'filename' keys
            prompt: Analysis prompt
            context: Conversation context

        Returns:
            ProcessingResult with combined analysis
        """
        try:
            media_inputs = []
            file_names = []

            for file_info in files:
                if "data" not in file_info or "filename" not in file_info:
                    continue

                doc_input = create_document_input(
                    file_info["data"], file_info["filename"]
                )
                media_inputs.append(doc_input)
                file_names.append(file_info["filename"])

            if not media_inputs:
                return ProcessingResult(
                    success=False, error="No valid documents provided"
                )

            # Enhanced prompt for multiple documents
            enhanced_prompt = f"""
            Analyze these {len(media_inputs)} documents:
            Files: {', '.join(file_names)}
            
            Provide:
            1. **Individual Summaries**: Brief summary of each document
            2. **Comparative Analysis**: How do these documents relate to each other?
            3. **Common Themes**: What topics, entities, or concepts appear across documents?
            4. **Key Differences**: What makes each document unique?
            5. **Synthesis**: Combined insights from all documents
            
            User Request: {prompt}
            
            Structure your response clearly with headings and organize information logically.
            """

            # Process with Gemini
            result = await self.gemini_api.process_multimodal_input(
                text_prompt=enhanced_prompt, media_inputs=media_inputs, context=context
            )

            if result.success:
                result.metadata = {
                    **result.metadata,
                    "processed_files": file_names,
                    "file_count": len(file_names),
                }
                self.logger.info(f"Successfully processed {len(file_names)} documents")
            else:
                self.logger.error(
                    f"Failed to process multiple documents: {result.error}"
                )

            return result

        except Exception as e:
            error_msg = f"Multiple document processing failed: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(success=False, error=error_msg)

    async def extract_document_entities(
        self, file_data: Union[bytes, io.BytesIO], filename: str
    ) -> ProcessingResult:
        """
        Extract entities and structured data from a document

        Args:
            file_data: Document data
            filename: Original filename

        Returns:
            ProcessingResult with extracted entities
        """
        try:
            doc_input = create_document_input(file_data, filename)

            entity_prompt = """
            Extract structured information from this document:
            
            1. **People**: Names of individuals mentioned
            2. **Organizations**: Companies, institutions, groups
            3. **Locations**: Places, addresses, geographic references
            4. **Dates & Times**: Important dates, deadlines, time periods
            5. **Topics & Keywords**: Main subjects and technical terms
            6. **Numbers & Metrics**: Important statistics, amounts, measurements
            7. **Technologies**: Software, tools, platforms mentioned
            8. **Contacts**: Email addresses, phone numbers, URLs
            
            Format your response as structured data with clear categories.
            If no entities are found in a category, indicate "None found".
            """

            result = await self.gemini_api.process_multimodal_input(
                text_prompt=entity_prompt, media_inputs=[doc_input]
            )

            if result.success:
                self.logger.info(f"Successfully extracted entities from: {filename}")

            return result

        except Exception as e:
            error_msg = f"Entity extraction failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(success=False, error=error_msg)

    async def code_analysis(
        self,
        file_data: Union[bytes, io.BytesIO],
        filename: str,
        analysis_type: str = "comprehensive",
    ) -> ProcessingResult:
        """
        Specialized code analysis

        Args:
            file_data: Code file data
            filename: Original filename
            analysis_type: Type of analysis (comprehensive, security, performance, structure)

        Returns:
            ProcessingResult with code analysis
        """
        try:
            # Check if it's a code file
            extension = filename.split(".")[-1].lower() if "." in filename else ""
            code_extensions = {
                "py",
                "js",
                "ts",
                "java",
                "cpp",
                "c",
                "cs",
                "php",
                "rb",
                "go",
                "rs",
            }

            if extension not in code_extensions:
                return ProcessingResult(
                    success=False,
                    error=f"File {filename} is not recognized as a code file",
                )

            doc_input = create_document_input(file_data, filename)

            analysis_prompts = {
                "comprehensive": """
                Provide a comprehensive code analysis:
                
                1. **Code Overview**: What does this code do? Main functionality and purpose
                2. **Structure Analysis**: Classes, functions, modules, and their relationships
                3. **Dependencies**: Imports, libraries, and external dependencies
                4. **Code Quality**: Best practices, potential improvements, code style
                5. **Security**: Potential security issues or vulnerabilities
                6. **Performance**: Performance considerations and optimizations
                7. **Documentation**: Comments, docstrings, and code readability
                8. **Testing**: Test coverage and testing approach
                """,
                "security": """
                Focus on security analysis of this code:
                
                1. **Security Vulnerabilities**: Potential security issues
                2. **Input Validation**: How user input is handled
                3. **Authentication & Authorization**: Security mechanisms
                4. **Data Protection**: Sensitive data handling
                5. **Dependencies**: Security of external libraries
                6. **Best Practices**: Security best practices compliance
                """,
                "performance": """
                Analyze performance aspects of this code:
                
                1. **Performance Bottlenecks**: Potential slow operations
                2. **Algorithms**: Efficiency of algorithms used
                3. **Resource Usage**: Memory and CPU considerations
                4. **Scalability**: How well code scales with data/users
                5. **Optimization Opportunities**: Suggestions for improvement
                """,
                "structure": """
                Analyze the structure and architecture of this code:
                
                1. **Architecture**: Overall code organization and patterns
                2. **Components**: Main classes, functions, and modules
                3. **Dependencies**: Internal and external dependencies
                4. **Design Patterns**: Design patterns used
                5. **Maintainability**: How easy is it to maintain and extend
                """,
            }

            prompt = analysis_prompts.get(
                analysis_type, analysis_prompts["comprehensive"]
            )

            result = await self.gemini_api.process_multimodal_input(
                text_prompt=prompt, media_inputs=[doc_input]
            )

            if result.success:
                result.metadata = {
                    **result.metadata,
                    "analysis_type": analysis_type,
                    "file_extension": extension,
                    "language": self._detect_language(extension),
                }
                self.logger.info(f"Successfully analyzed code file: {filename}")

            return result

        except Exception as e:
            error_msg = f"Code analysis failed for {filename}: {str(e)}"
            self.logger.error(error_msg)
            return ProcessingResult(success=False, error=error_msg)

    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension"""
        language_map = {
            "py": "Python",
            "js": "JavaScript",
            "ts": "TypeScript",
            "java": "Java",
            "cpp": "C++",
            "c": "C",
            "cs": "C#",
            "php": "PHP",
            "rb": "Ruby",
            "go": "Go",
            "rs": "Rust",
            "sql": "SQL",
            "sh": "Shell Script",
            "yaml": "YAML",
            "yml": "YAML",
            "json": "JSON",
            "html": "HTML",
            "css": "CSS",
        }
        return language_map.get(extension.lower(), "Unknown")

    def is_supported_document(self, filename: str) -> bool:
        """Check if document type is supported"""
        if not filename or "." not in filename:
            return False
        extension = filename.split(".")[-1].lower()
        return extension in self.supported_extensions

    def get_document_info(self, filename: str) -> Dict[str, str]:
        """Get information about document type"""
        if not filename or "." not in filename:
            return {"type": "unknown", "category": "unknown"}

        extension = filename.split(".")[-1].lower()

        categories = {
            "document": {"pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf"},
            "text": {"txt", "csv", "html", "md", "json", "xml"},
            "code": {
                "py",
                "js",
                "ts",
                "java",
                "cpp",
                "c",
                "cs",
                "php",
                "rb",
                "go",
                "rs",
                "sql",
                "sh",
                "yaml",
                "yml",
            },
        }

        for category, extensions in categories.items():
            if extension in extensions:
                return {
                    "type": extension,
                    "category": category,
                    "language": (
                        self._detect_language(extension) if category == "code" else None
                    ),
                }

        return {"type": extension, "category": "unknown"}

    async def process_document_enhanced(
        self,
        file: Union[bytes, io.BytesIO],
        file_extension: str,
        prompt: str = "Analyze this document and provide a comprehensive summary.",
    ) -> Dict[str, Any]:
        """
        Enhanced document processing for PDFs and complex documents.

        Args:
            file: Document data as bytes or BytesIO
            file_extension: File extension (e.g., 'pdf', 'docx')
            prompt: Analysis prompt

        Returns:
            Dictionary with processing results
        """
        try:
            # Create a filename from the extension
            filename = f"document.{file_extension}"

            # Use the existing process_document method
            result = await self.process_document(
                file_data=file, filename=filename, prompt=prompt
            )

            # Generate a meaningful document ID
            import datetime
            import hashlib

            # Create document ID based on timestamp and file content hash
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create a hash of the file content for uniqueness
            if isinstance(file, io.BytesIO):
                file.seek(0)  # Reset position to start
                file_content = file.read()
                file.seek(0)  # Reset again for any future reads
            else:
                file_content = file if isinstance(file, bytes) else str(file).encode()

            content_hash = hashlib.md5(file_content).hexdigest()[:8]
            document_id = f"{file_extension}_{timestamp}_{content_hash}"

            # Convert ProcessingResult to dictionary format expected by message handlers
            return {
                "result": (
                    result.content if result.success else f"Error: {result.error}"
                ),
                "document_id": document_id,
                "success": result.success,
                "metadata": result.metadata if hasattr(result, "metadata") else {},
            }

        except Exception as e:
            self.logger.error(f"Error in process_document_enhanced: {e}")
            return {
                "result": f"Error processing document: {str(e)}",
                "document_id": "error",
                "success": False,
                "metadata": {},
            }

    async def process_document_from_file(
        self,
        file: Union[bytes, io.BytesIO],
        file_extension: str,
        prompt: str = "Analyze this document and provide a comprehensive summary.",
    ) -> Dict[str, Any]:
        """
        Process documents from file data (non-PDF documents).

        Args:
            file: Document data as bytes or BytesIO
            file_extension: File extension (e.g., 'txt', 'docx', 'json')
            prompt: Analysis prompt

        Returns:
            Dictionary with processing results
        """
        try:
            # Create a filename from the extension
            filename = f"document.{file_extension}"

            # Use the existing process_document method
            result = await self.process_document(
                file_data=file, filename=filename, prompt=prompt
            )

            # Generate a meaningful document ID (same logic as enhanced method)
            import datetime
            import hashlib

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            if isinstance(file, io.BytesIO):
                file.seek(0)
                file_content = file.read()
                file.seek(0)
            else:
                file_content = file if isinstance(file, bytes) else str(file).encode()

            content_hash = hashlib.md5(file_content).hexdigest()[:8]
            document_id = f"{file_extension}_{timestamp}_{content_hash}"

            # Convert ProcessingResult to dictionary format expected by message handlers
            return {
                "result": (
                    result.content if result.success else f"Error: {result.error}"
                ),
                "document_id": document_id,
                "success": result.success,
                "metadata": result.metadata if hasattr(result, "metadata") else {},
            }

        except Exception as e:
            self.logger.error(f"Error in process_document_from_file: {e}")
            return {
                "result": f"Error processing document: {str(e)}",
                "document_id": "error",
                "success": False,
                "metadata": {},
            }


# Utility functions for document processing
async def quick_document_analysis(
    gemini_api: GeminiAPI,
    file_data: Union[bytes, io.BytesIO],
    filename: str,
    prompt: str = "Analyze this document",
) -> str:
    """Quick document analysis helper"""
    processor = DocumentProcessor(gemini_api)
    result = await processor.process_document(file_data, filename, prompt)
    return result.content if result.success else f"Error: {result.error}"


async def extract_document_text(
    gemini_api: GeminiAPI, file_data: Union[bytes, io.BytesIO], filename: str
) -> str:
    """Extract plain text from document"""
    processor = DocumentProcessor(gemini_api)
    result = await processor.process_document(
        file_data,
        filename,
        "Extract all text content from this document. Return only the text without analysis.",
    )
    return result.content if result.success else f"Error: {result.error}"


# Utility functions for document processing
async def quick_document_analysis(
    gemini_api: GeminiAPI,
    file_data: Union[bytes, io.BytesIO],
    filename: str,
    prompt: str = "Analyze this document",
) -> str:
    """Quick document analysis helper"""
    processor = DocumentProcessor(gemini_api)
    result = await processor.process_document(file_data, filename, prompt)
    return result.content if result.success else f"Error: {result.error}"


async def extract_document_text(
    gemini_api: GeminiAPI, file_data: Union[bytes, io.BytesIO], filename: str
) -> str:
    """Extract plain text from document"""
    processor = DocumentProcessor(gemini_api)
    result = await processor.process_document(
        file_data,
        filename,
        "Extract all text content from this document. Return only the text without analysis.",
    )
    return result.content if result.success else f"Error: {result.error}"
