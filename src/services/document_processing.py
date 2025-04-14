import logging
import google.generativeai as genai
import io
import httpx
from typing import Optional, List, Dict, BinaryIO, Union, Any
import asyncio
import os
from dotenv import load_dotenv
from telegram import Bot
from services.knowledge_graph import KnowledgeGraph
import mimetypes
import re
import json

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class DocumentProcessor:
    """Handles document processing using the Gemini API."""

    # Simplified and consolidated MIME types dictionary
    MIME_TYPE_MAPPING = {
        # Document formats
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "ppt": "application/vnd.ms-powerpoint",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        # Text formats
        "txt": "text/plain",
        "csv": "text/csv",
        "html": "text/html",
        "htm": "text/html",
        "md": "text/markdown",
        "rtf": "text/rtf",
        "xml": "text/xml",
        # Code formats (all treated as text/plain)
        "py": "text/plain",
        "js": "text/plain",
        "java": "text/plain",
        "cpp": "text/plain",
        "c": "text/plain",
        "cs": "text/plain",
        "php": "text/plain",
        "rb": "text/plain",
        "go": "text/plain",
        "swift": "text/plain",
        "kt": "text/plain",
        "rs": "text/plain",
        "ts": "text/plain",
        "sql": "text/plain",
        "sh": "text/plain",
        "yaml": "text/plain",
        "json": "application/json",
        # Image formats
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "svg": "image/svg+xml",
    }

    # Code file extensions for special handling
    CODE_EXTENSIONS = {
        "py",
        "js",
        "java",
        "cpp",
        "c",
        "cs",
        "php",
        "rb",
        "go",
        "swift",
        "kt",
        "rs",
        "ts",
        "html",
        "css",
        "sql",
        "sh",
        "yaml",
        "json",
        "xml",
        "md",
    }

    def __init__(self, bot: Bot, knowledge_graph: Optional[KnowledgeGraph] = None):
        """Initialize the DocumentProcessor with Gemini API configuration."""
        self.bot = bot
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger(__name__)

        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found or empty")

        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_output_tokens": 4096,
        }

    async def upload_file(self, file: BinaryIO, mime_type: str) -> Dict[str, Any]:
        """Asynchronously upload a file to the Gemini API."""
        try:
            file_data = file.read()
            return {"mime_type": mime_type, "data": file_data}
        except Exception as e:
            self.logger.error(f"Error uploading file: {str(e)}")
            raise

    def get_mime_type(self, file_extension: str) -> str:
        """Get the MIME type for a given file extension with fallback mechanisms."""
        extension = file_extension.lower()

        # Check our mapping first
        if extension in self.MIME_TYPE_MAPPING:
            return self.MIME_TYPE_MAPPING[extension]

        # Try using mimetypes module as fallback
        mime_type = mimetypes.guess_type(f"file.{extension}")[0]
        if mime_type:
            return mime_type

        # Default fallback
        return "application/octet-stream"

    async def process_document_from_url(self, document_url: str, prompt: str) -> str:
        """Process a document from a URL using the Gemini API."""
        try:
            # Get file extension from URL
            file_extension = document_url.split(".")[-1]
            mime_type = self.get_mime_type(file_extension)

            # Retrieve document from URL
            async with httpx.AsyncClient() as client:
                response = await client.get(document_url)
                response.raise_for_status()
                doc_data = io.BytesIO(response.content)

            # Upload document using File API with mime_type
            uploaded_file = await self.upload_file(file=doc_data, mime_type=mime_type)

            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                [uploaded_file, prompt],
                generation_config=self.generation_config,
            )

            return response.text
        except Exception as e:
            self.logger.error(f"Error processing document from URL: {str(e)}")
            raise

    async def process_document_from_file(
        self, file: Union[bytes, BinaryIO], file_extension: str, prompt: str
    ) -> str:
        """Process a document from a file object."""
        try:
            # Convert to BytesIO if we received bytes
            file_io = io.BytesIO(file) if isinstance(file, bytes) else file
            file_io.seek(0)

            # Get the MIME type
            mime_type = self.get_mime_type(file_extension)

            # Process the file using Gemini API
            uploaded_file = await self.upload_file(file=file_io, mime_type=mime_type)

            # Generate a response using the model
            response = await asyncio.to_thread(
                self.model.generate_content,
                [uploaded_file, prompt],
                generation_config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 32,
                    "max_output_tokens": 4096,
                },
            )

            if hasattr(response, "text"):
                return response.text
            else:
                self.logger.error("No text property found in the response")
                return "Failed to process document: No text in response"

        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            return f"Error processing document: {str(e)}"

    async def process_multiple_documents(
        self, documents: List[Dict], prompt: str
    ) -> str:
        """Process multiple documents simultaneously."""
        try:
            uploaded_docs = []

            for doc in documents:
                if "url" in doc:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(doc["url"])
                        response.raise_for_status()
                        doc_data = io.BytesIO(response.content)
                else:
                    doc_data = doc["file"]

                mime_type = self.get_mime_type(doc["extension"])
                uploaded_doc = await self.upload_file(
                    file=doc_data, mime_type=mime_type
                )
                uploaded_docs.append(uploaded_doc)

            contents = uploaded_docs + [prompt]

            response = await asyncio.to_thread(
                self.model.generate_content,
                contents,
                generation_config=self.generation_config,
            )

            if hasattr(response, "text") and response.text:
                return response.text
            elif isinstance(response, str) and response:
                return response
            return "Sorry, I couldn't process the documents."

        except Exception as e:
            self.logger.error(f"Error processing multiple documents: {str(e)}")
            raise

    async def process_document_enhanced(
        self,
        file: Union[bytes, BinaryIO],
        file_extension: str,
        prompt: str,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process documents with enhanced capabilities, including code analysis.
        """
        try:
            # Create a unique document ID if not provided
            if not document_id:
                document_id = f"doc_{int(asyncio.get_event_loop().time())}"

            # Process the file content properly
            file_io = io.BytesIO(file) if isinstance(file, bytes) else file
            file_io.seek(0)

            # Get mime type
            mime_type = self.get_mime_type(file_extension)

            # Upload to Gemini
            uploaded_file = await self.upload_file(file=file_io, mime_type=mime_type)

            # Enhanced processing: Generate summary, extract entities, and answer the prompt
            # Create a more comprehensive prompt that captures multiple aspects
            enhanced_prompt = f"""
            Analyze this document with the following approach:
            1. Understand the document layout, structure, and content type
            2. Extract key information from any tables, charts, or structured data
            3. Identify main topics, entities, and important concepts
            
            Then provide:
            - A comprehensive summary of the document's main points
            - Key entities (people, organizations, locations, technologies mentioned)
            - Important dates or time-related information
            - Main topics and concepts covered
            
            Finally, address the user's specific request: {prompt}
            
            Format your response in clear sections with markdown formatting.
            """

            # Use Gemini 2.0 Flash model (updated from previous models)
            pro_model = genai.GenerativeModel("gemini-2.0-flash")

            # Generate response with enhanced configuration
            response = await asyncio.to_thread(
                pro_model.generate_content,
                [uploaded_file, {"text": enhanced_prompt}],
                generation_config={
                    "temperature": 0.2,  # Lower for more factual responses
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,  # Allow longer responses for document analysis
                    "response_mime_type": "text/plain",
                },
            )

            # Extract text from response
            if not hasattr(response, "text") or not response.text:
                self.logger.error("No text found in the Gemini API response")
                return {
                    "result": "Error: Failed to analyze document. Could not process the document.",
                    "document_id": document_id
                }

            response_text = response.text

            # Extract entities for knowledge graph
            entities_result = await self.extract_document_entities(
                file_io, file_extension
            )

            # Format entities nicely
            entities_md = ""
            if entities_result and isinstance(entities_result, dict):
                entities_md += "\n\n**Extracted Entities:**\n"
                entities = entities_result.get("entities", {})
                for category, items in entities.items():
                    if isinstance(items, list) and items:
                        entities_md += f"- **{category.capitalize()}**: {', '.join(items)}\n"
                    elif isinstance(items, dict):
                        # nested dict (e.g., structured_data)
                        for subcat, subitems in items.items():
                            if isinstance(subitems, list) and subitems:
                                entities_md += f"- **{subcat.capitalize()}**: {', '.join(subitems)}\n"
            
            # Compose final markdown response
            formatted_result = f"**Document Analysis:**\n\n{response_text}{entities_md}"

            # Add to knowledge graph if available
            knowledge_graph_summary = None
            if self.knowledge_graph and user_id:
                try:
                    file_io.seek(0)  # Rewind file pointer
                    # Extract text for knowledge graph (simpler approach than full OCR)
                    text_model = genai.GenerativeModel("gemini-2.0-flash")
                    text_extraction = await asyncio.to_thread(
                        text_model.generate_content,
                        [uploaded_file, "Extract all text content from this document."],
                        generation_config={
                            "temperature": 0.1,
                            "max_output_tokens": 16384,
                        },
                    )
                    extracted_text = (
                        text_extraction.text if hasattr(text_extraction, "text") else ""
                    )
                    knowledge_graph_summary = (
                        await self.knowledge_graph.add_document_entities(
                            document_id=document_id,
                            document_content=extracted_text,
                            user_id=user_id,
                        )
                    )
                    self.logger.info(
                        f"Added document {document_id} to knowledge graph for user {user_id}"
                    )
                except Exception as kg_error:
                    self.logger.error(
                        f"Error adding to knowledge graph: {str(kg_error)}"
                    )

            # Return formatted markdown response
            result = {
                "result": formatted_result,
                "document_id": document_id,
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in enhanced document processing: {str(e)}")
            return {
                "result": f"Error processing document: {str(e)}",
                "document_id": document_id if document_id else "unknown",
            }

    def _create_enhanced_prompt(self, user_prompt: str, file_extension: str) -> str:
        """Create an enhanced prompt for document analysis."""
        base_prompt = f"""
        Analyze this document with the following approach:
        1. Understand the document layout, structure, and content type
        2. Extract key information from any tables, charts, or structured data
        3. Identify main topics, entities, and important concepts
        
        Then provide:
        - A comprehensive summary of the document's main points
        - Key entities (people, organizations, locations, technologies mentioned)
        - Important dates or time-related information
        - Main topics and concepts covered
        
        Finally, address the user's specific request: {user_prompt}
        
        Format your response in clear sections with markdown formatting.
        """

        # Add language-specific guidance for code files
        if file_extension.lower() in self.CODE_EXTENSIONS:
            return self._get_language_specific_prompt(file_extension, base_prompt)

        return base_prompt

    async def _update_knowledge_graph(
        self, file_io: BinaryIO, uploaded_file: Dict, user_id: str, document_id: str
    ) -> Dict:
        """Update knowledge graph with document content."""
        try:
            file_io.seek(0)  # Rewind file pointer

            # Extract text for knowledge graph
            text_model = genai.GenerativeModel("gemini-2.0-flash")
            text_extraction = await asyncio.to_thread(
                text_model.generate_content,
                [uploaded_file, "Extract all text content from this document."],
                generation_config={"temperature": 0.1, "max_output_tokens": 16384},
            )

            extracted_text = (
                text_extraction.text if hasattr(text_extraction, "text") else ""
            )

            # Add to knowledge graph
            knowledge_graph_summary = await self.knowledge_graph.add_document_entities(
                document_id=document_id,
                document_content=extracted_text,
                user_id=user_id,
            )

            self.logger.info(
                f"Added document {document_id} to knowledge graph for user {user_id}"
            )
            return knowledge_graph_summary

        except Exception as kg_error:
            self.logger.error(f"Error adding to knowledge graph: {str(kg_error)}")
            return {"error": str(kg_error)}

    async def extract_document_entities(
        self, file_content: Union[bytes, BinaryIO], file_extension: str
    ) -> Dict[str, Any]:
        """Extract named entities, key topics, and structured data from documents"""
        try:
            # Ensure we have a BytesIO object
            file_io = (
                io.BytesIO(file_content)
                if isinstance(file_content, bytes)
                else file_content
            )
            file_io.seek(0)

            # Get mime type and upload file
            mime_type = self.get_mime_type(file_extension)
            uploaded_file = await self.upload_file(file=file_io, mime_type=mime_type)

            # Entity extraction prompt
            entity_extraction_prompt = """
            Extract the following from this document:
            1. Named entities (people, organizations, locations)
            2. Key topics and themes
            3. Dates and time references
            4. Structured data (tables, lists)
            5. Technologies or products mentioned
            6. Industry-specific terminology
            
            Format your response as JSON with these categories.
            Example format:
            {
              "entities": {
                "people": ["Name1", "Name2"],
                "organizations": ["Org1", "Org2"],
                "locations": ["Location1", "Location2"],
                "dates": ["Date1", "Date2"],
                "technologies": ["Tech1", "Tech2"],
                "concepts": ["Concept1", "Concept2"]
              },
              "topics": ["Topic1", "Topic2"],
              "structured_data": {
                "tables": [{
                  "description": "Table description",
                  "rows": 5,
                  "columns": 3
                }],
                "lists": [{
                  "description": "List description",
                  "items": 6
                }]
              }
            }
            """

            # Use Gemini for extraction
            pro_model = genai.GenerativeModel("gemini-2.0-flash")
            response = await asyncio.to_thread(
                pro_model.generate_content,
                [uploaded_file, entity_extraction_prompt],
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json",
                },
            )

            # Process response
            if hasattr(response, "text"):
                try:
                    # Extract JSON data from response
                    match = re.search(
                        r"```json\s*(.*?)\s*```", response.text, re.DOTALL
                    )
                    if match:
                        return json.loads(match.group(1))
                    else:
                        # Try parsing the whole text as JSON
                        return json.loads(response.text)
                except json.JSONDecodeError:
                    # Fallback to minimal entity structure
                    return {
                        "entities": {
                            "people": [],
                            "organizations": [],
                            "locations": [],
                            "technologies": [],
                            "concepts": [],
                        },
                        "raw_extraction": response.text,
                        "format_error": "Could not parse JSON",
                    }

            return {"error": "No extraction results"}

        except Exception as e:
            self.logger.error(f"Entity extraction error: {str(e)}")
            return {"error": str(e)}

    def _get_language_specific_prompt(
        self, file_extension: str, base_prompt: str
    ) -> str:
        """Get language-specific analysis prompts based on file extension."""
        language_prompts = {
            "py": """
                Python-specific analysis:
                - Check for PEP 8 compliance
                - Identify use of common libraries (NumPy, Pandas, Flask, etc.)
                - Look for Pythonic patterns and anti-patterns
            """,
            "js": """
                JavaScript-specific analysis:
                - Identify the framework if any (React, Vue, Angular, etc.)
                - Check for modern JS features and patterns
                - Look for potential performance issues
            """,
            "java": """
                Java-specific analysis:
                - Identify design patterns used
                - Check for proper exception handling
                - Review class structure and inheritance
            """,
            "cpp": """
                C++ specific analysis:
                - Check for memory management issues
                - Look for efficient use of STL
                - Identify potential performance bottlenecks
            """,
        }

        language_specific = language_prompts.get(file_extension.lower(), "")
        return (
            f"{base_prompt}\n\n{language_specific}"
            if language_specific
            else base_prompt
        )
