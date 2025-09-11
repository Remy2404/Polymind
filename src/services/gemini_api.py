"""
Modern Gemini 2.5 Flash API Integration with Google Gen AI SDK
Handles combined multimodal inputs: text + images + documents + audio + video
Clean, maintainable, and scalable implementation with tool calling support
"""

import logging
import asyncio
import io
import os
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum

from google import genai
from google.genai import types
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
)
from PIL import Image

from src.services.rate_limiter import RateLimiter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GEMINI_API_KEY is required")


class MediaType(Enum):
    """Supported media types for multimodal processing"""

    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


@dataclass
class MediaInput:
    """Represents a media input for processing"""

    type: MediaType
    data: Union[bytes, str, io.BytesIO]
    mime_type: str
    filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ToolCall:
    """Represents a tool/function call from the model"""

    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of multimodal processing"""

    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_calls: Optional[List[ToolCall]] = None  # Alias for compatibility


class MediaProcessor:
    """Handles processing of different media types for Gemini"""

    # Image processing constants
    MAX_IMAGE_SIZE = 4096
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
    IMAGE_QUALITY = 85

    # Document MIME type mapping
    DOCUMENT_MIME_TYPES = {
        # Documents
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "ppt": "application/vnd.ms-powerpoint",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        # Text files
        "txt": "text/plain",
        "csv": "text/csv",
        "md": "text/markdown",
        "html": "text/html",
        "json": "application/json",
        "xml": "text/xml",
        # Code files
        "py": "text/plain",
        "js": "text/plain",
        "ts": "text/plain",
        "java": "text/plain",
        "cpp": "text/plain",
        "c": "text/plain",
        "cs": "text/plain",
        "php": "text/plain",
        "rb": "text/plain",
        "go": "text/plain",
        "rs": "text/plain",
        "sql": "text/plain",
        "sh": "text/plain",
        "yaml": "text/plain",
        "yml": "text/plain",
    }

    @staticmethod
    def validate_image(image_data: Union[bytes, io.BytesIO]) -> bool:
        """Validate image format and size"""
        try:
            if isinstance(image_data, io.BytesIO):
                image_data.seek(0)
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data

            if len(img_bytes) > MediaProcessor.MAX_FILE_SIZE:
                return False

            with Image.open(io.BytesIO(img_bytes)) as img:
                if img.format not in MediaProcessor.SUPPORTED_IMAGE_FORMATS:
                    return False
                # Max 25MP
                if img.size[0] * img.size[1] > 25000000:
                    return False
                return True
        except Exception:
            return False

    @staticmethod
    def optimize_image(image_data: Union[bytes, io.BytesIO]) -> io.BytesIO:
        """Optimize image for Gemini processing"""
        try:
            if isinstance(image_data, io.BytesIO):
                image_data.seek(0)
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data

            with Image.open(io.BytesIO(img_bytes)) as img:
                # Convert to RGB if needed
                if img.mode in ("RGBA", "LA", "P"):
                    if img.mode == "P" and "transparency" in img.info:
                        img = img.convert("RGBA")
                    if img.mode in ("RGBA", "LA"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "RGBA":
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img, mask=img.split()[1])
                        img = background
                    else:
                        img = img.convert("RGB")

                # Resize if too large
                if max(img.size) > MediaProcessor.MAX_IMAGE_SIZE:
                    ratio = MediaProcessor.MAX_IMAGE_SIZE / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Save optimized image
                output = io.BytesIO()
                img.save(
                    output,
                    format="JPEG",
                    quality=MediaProcessor.IMAGE_QUALITY,
                    optimize=True,
                )
                output.seek(0)
                return output

        except Exception as e:
            logging.error(f"Image optimization failed: {e}")
            raise ValueError(f"Image processing failed: {e}")

    @staticmethod
    def get_image_mime_type(image_data: Union[bytes, io.BytesIO]) -> str:
        """Get MIME type for image"""
        try:
            if isinstance(image_data, io.BytesIO):
                image_data.seek(0)
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data

            with Image.open(io.BytesIO(img_bytes)) as img:
                format_map = {
                    "JPEG": "image/jpeg",
                    "PNG": "image/png",
                    "WEBP": "image/webp",
                    "GIF": "image/gif",
                }
                return format_map.get(img.format, "image/jpeg")
        except Exception:
            return "image/jpeg"

    @staticmethod
    def get_document_mime_type(filename: str) -> str:
        """Get MIME type from filename extension"""
        if not filename or "." not in filename:
            return "application/octet-stream"

        ext = filename.split(".")[-1].lower()
        return MediaProcessor.DOCUMENT_MIME_TYPES.get(ext, "application/octet-stream")

    @staticmethod
    def validate_document(file_data: Union[bytes, io.BytesIO], filename: str) -> bool:
        """Validate document for processing"""
        try:
            if isinstance(file_data, io.BytesIO):
                size = len(file_data.getvalue())
            else:
                size = len(file_data)

            # Check file size (50MB limit)
            if size > 50 * 1024 * 1024:
                return False

            # Basic validation passed
            return True
        except Exception:
            return False


class GeminiAPI:
    """
    Modern Gemini 2.5 Flash API client with multimodal and tool calling support
    Uses the latest Google Gen AI SDK for enhanced capabilities
    """

    def __init__(self, rate_limiter: RateLimiter):
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = rate_limiter
        self.media_processor = MediaProcessor()

        # Initialize the Google Gen AI client
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Generation configuration optimized for 2.5 Flash
        self.generation_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )

        self.logger.info("Gemini 2.5 Flash API initialized with Google Gen AI SDK")

    async def process_multimodal_input(
        self,
        text_prompt: str,
        media_inputs: Optional[List[MediaInput]] = None,
        context: Optional[List[Dict]] = None,
        model_name: str = "gemini-2.5-flash",
        tools: Optional[List[Union[Callable, types.Tool]]] = None,
        auto_function_calling: bool = True,
    ) -> ProcessingResult:
        """
        Process combined multimodal input with tool calling support

        Args:
            text_prompt: The main text prompt
            media_inputs: List of media inputs (images, documents, etc.)
            context: Conversation context
            model_name: Gemini model to use
            tools: List of tools/functions the model can call
            auto_function_calling: Whether to automatically execute function calls

        Returns:
            ProcessingResult with the generated response and any tool calls
        """
        try:
            await self.rate_limiter.acquire()

            # Build content parts for Gemini
            content_parts = []

            # Process media inputs first
            if media_inputs:
                for media in media_inputs:
                    processed_content = await self._process_media_input(media)
                    if processed_content:
                        content_parts.extend(processed_content)

            # Add the main text prompt
            content_parts.append(text_prompt)

            # Build generation config
            config = types.GenerateContentConfig(
                temperature=self.generation_config.temperature,
                top_p=self.generation_config.top_p,
                top_k=self.generation_config.top_k,
                max_output_tokens=self.generation_config.max_output_tokens,
            )

            # Add tools if provided
            if tools:
                config.tools = tools
                if not auto_function_calling:
                    config.automatic_function_calling = types.AutomaticFunctionCallingConfig(
                        disable=True
                    )

            # Add context to content if available
            contents = self._build_conversation_context(context, content_parts)

            # Generate response using Gemini
            response = await self._generate_with_retry(contents, model_name, config)

            if response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                # Extract response text from parts (per SDK: parts can include text, function calls, etc.)
                response_text = ""
                if candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:  # Only extract text parts
                            response_text += part.text
                
                # Extract tool calls if any
                tool_calls = []
                if hasattr(candidate, 'function_calls') and candidate.function_calls:
                    for fc in candidate.function_calls:
                        tool_calls.append(
                            ToolCall(
                                name=fc.name,
                                args=dict(fc.args) if hasattr(fc, "args") else {},
                                id=getattr(fc, "id", None),
                            )
                        )

                return ProcessingResult(
                    success=True,
                    content=response_text.strip() if response_text else None,
                    tool_calls=tool_calls,
                    function_calls=tool_calls,
                    metadata={
                        "model": model_name,
                        "media_count": len(media_inputs) if media_inputs else 0,
                        "token_count": len(response_text.split()) if response_text else 0,
                        "has_tool_calls": len(tool_calls) > 0,
                    },
                )
            else:
                return ProcessingResult(
                    success=False, error="Empty or invalid response from Gemini API"
                )

        except Exception as e:
            self.logger.error(f"Multimodal processing failed: {e}")
            return ProcessingResult(success=False, error=f"Processing failed: {str(e)}")

    async def _process_media_input(self, media: MediaInput) -> Optional[List[Any]]:
        """Process individual media input based on its type"""
        try:
            if media.type == MediaType.IMAGE:
                return await self._process_image_input(media)
            elif media.type == MediaType.DOCUMENT:
                return await self._process_document_input(media)
            elif media.type == MediaType.AUDIO:
                return [
                    f"[Audio file: {media.filename or 'audio'} - audio processing not yet implemented]"
                ]
            elif media.type == MediaType.VIDEO:
                return [
                    f"[Video file: {media.filename or 'video'} - video processing not yet implemented]"
                ]
            else:
                return [f"[Unknown media type: {media.type.value}]"]

        except Exception as e:
            self.logger.error(f"Failed to process {media.type.value}: {e}")
            return [
                f"[Error processing {media.type.value}: {media.filename or 'unknown'}]"
            ]

    async def _process_image_input(self, media: MediaInput) -> Optional[List[Any]]:
        """Process image input for Gemini using new SDK"""
        try:
            from google.genai import types

            # Validate image
            if not self.media_processor.validate_image(media.data):
                return [f"[Invalid image file: {media.filename or 'unknown'}]"]

            # Optimize image
            optimized_image = self.media_processor.optimize_image(media.data)
            mime_type = self.media_processor.get_image_mime_type(optimized_image)

            # Get image bytes
            optimized_image.seek(0)
            image_bytes = optimized_image.getvalue()

            # Return Gemini-compatible image part using new SDK
            return [types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]

        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return [f"[Image processing failed: {media.filename or 'unknown'}]"]

    async def _process_document_input(self, media: MediaInput) -> Optional[List[Any]]:
        """Process document input for Gemini using new SDK"""
        try:
            from google.genai import types

            if not media.filename:
                return ["[Document file uploaded without filename]"]

            # Validate document
            if not self.media_processor.validate_document(media.data, media.filename):
                return [f"[Document too large or invalid: {media.filename}]"]

            # Get document data
            if isinstance(media.data, io.BytesIO):
                media.data.seek(0)
                doc_bytes = media.data.getvalue()
            else:
                doc_bytes = media.data

            mime_type = self.media_processor.get_document_mime_type(media.filename)

            # For supported document formats, use new SDK
            if mime_type in [
                "application/pdf",
                "text/plain",
                "text/markdown",
                "application/json",
                "text/html",
                "text/csv",
            ]:
                try:
                    # Use new SDK file upload
                    uploaded_file = await self._upload_file_to_gemini_new_sdk(
                        doc_bytes, mime_type, media.filename
                    )
                    return [uploaded_file]
                except Exception as upload_error:
                    self.logger.error(f"File upload failed: {upload_error}")
                    # Fallback to direct bytes processing
                    return [types.Part.from_bytes(data=doc_bytes, mime_type=mime_type)]
            else:
                # For other formats, try direct processing
                return [types.Part.from_bytes(data=doc_bytes, mime_type=mime_type)]

        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return [f"[Document processing failed: {media.filename or 'unknown'}]"]

    async def _upload_file_to_gemini_new_sdk(
        self, file_bytes: bytes, mime_type: str, filename: str
    ) -> Any:
        """Upload file to Gemini using new SDK"""
        try:
            # Use new SDK's file upload functionality
            file_data = io.BytesIO(file_bytes)
            
            # Upload using the new client
            uploaded_file = await asyncio.to_thread(
                self.client.files.upload,
                file=file_data,
                mime_type=mime_type,
                display_name=filename
            )

            return uploaded_file

        except Exception as e:
            self.logger.error(f"New SDK file upload failed: {e}")
            raise

    async def _upload_file_to_gemini(
        self, file_bytes: bytes, mime_type: str, filename: str
    ) -> Any:
        """Legacy upload method - kept for compatibility"""
        return await self._upload_file_to_gemini_new_sdk(file_bytes, mime_type, filename)

    def _build_conversation_context(
        self, context: Optional[List[Dict]], content_parts: List[Any]
    ) -> List[Any]:
        """Build conversation context using new SDK patterns"""
        from google.genai import types

        contents = []

        # Add conversation history if available
        if context:
            for msg in context[-10:]:  # Last 10 messages to avoid token limits
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if content:
                    if role == "user":
                        contents.append(
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(content)]
                            )
                        )
                    elif role == "assistant" or role == "model":
                        contents.append(
                            types.Content(
                                role="model",
                                parts=[types.Part.from_text(content)]
                            )
                        )

        # Add current content parts
        if content_parts:
            # Convert string parts to Part objects
            parts = []
            for part in content_parts:
                if isinstance(part, str):
                    parts.append(types.Part.from_text(part))
                else:
                    parts.append(part)
            
            contents.append(
                types.Content(role="user", parts=parts)
            )

        return contents if contents else content_parts

    def _build_system_context(self, context: Optional[List[Dict]]) -> Optional[str]:
        """Legacy method - kept for compatibility"""
        if not context:
            return None

        system_msg = (
            "You are Gemini, Google's advanced multimodal AI assistant. You can analyze "
            "text, images, documents, and other media types. Provide helpful, accurate, "
            "and detailed responses based on all provided content."
        )

        return system_msg

    def get_system_message(self) -> str:
        """
        Return the system message for Gemini models.
        This is used by the prompt formatter for consistent system prompts.
        """
        return (
            "You are Gemini, Google's advanced multimodal AI assistant. You can analyze "
            "text, images, documents, and other media types. Provide helpful, accurate, "
            "and detailed responses based on all provided content."
        )

    async def _generate_with_retry(
        self, 
        contents: List[Any], 
        model_name: str, 
        config: Any,
        max_retries: int = 3
    ) -> Any:
        """Generate content with retry logic using new SDK"""
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return response

            except ResourceExhausted as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                raise e

            except ServiceUnavailable as e:
                if attempt < max_retries - 1:
                    wait_time = 1 + attempt
                    self.logger.warning(
                        f"Service unavailable, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise e

            except Exception as e:
                self.logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                raise e

        raise Exception("All retry attempts failed")

    async def generate_content_with_tools(
        self,
        prompt: str,
        tools: List[Union[Callable, Any]],
        context: Optional[List[Dict]] = None,
        auto_execute: bool = True,
        model_name: str = "gemini-2.5-flash",
    ) -> ProcessingResult:
        """
        Generate content with tool calling capabilities

        Args:
            prompt: The text prompt
            tools: List of functions or tool declarations
            context: Conversation context
            auto_execute: Whether to automatically execute function calls
            model_name: Model to use

        Returns:
            ProcessingResult with content and tool calls
        """
        return await self.process_multimodal_input(
            text_prompt=prompt,
            context=context,
            model_name=model_name,
            tools=tools,
            auto_function_calling=auto_execute,
        )

    async def stream_content(
        self,
        prompt: str,
        media_inputs: Optional[List[MediaInput]] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        """Stream content generation using new SDK"""
        try:
            from google.genai import types
            
            await self.rate_limiter.acquire()

            # Build content parts
            content_parts = []
            
            # Process media inputs
            if media_inputs:
                for media in media_inputs:
                    processed_content = await self._process_media_input(media)
                    if processed_content:
                        content_parts.extend(processed_content)

            # Add text prompt
            content_parts.append(types.Part.from_text(prompt))

            # Stream response
            async for chunk in await asyncio.to_thread(
                self.client.models.generate_content_stream,
                model=model_name,
                contents=content_parts,
            ):
                # Correctly parse nested structure per SDK docs
                if chunk.candidates and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            yield part.text

        except Exception as e:
            self.logger.error(f"Streaming failed: {e}")
            yield f"Error: {str(e)}"

    async def create_chat_session(
        self, 
        model_name: str = "gemini-2.5-flash",
        tools: Optional[List[Union[Callable, Any]]] = None
    ):
        """Create a chat session using new SDK"""
        try:
            config = None
            if tools:
                config = types.GenerateContentConfig(tools=tools)
            
            chat = self.client.chats.create(model=model_name, config=config)
            return chat
        except Exception as e:
            self.logger.error(f"Failed to create chat session: {e}")
            raise

    async def generate_content(
        self, prompt: str, context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generate content method for backward compatibility
        Returns a dictionary with status and content for compatibility with existing tests
        """
        try:
            result = await self.process_multimodal_input(
                text_prompt=prompt, context=context
            )

            if result.success:
                return {"status": "success", "content": result.content}
            else:
                return {
                    "status": "error",
                    "content": f"Error: {result.error}",
                    "error": result.error,
                }
        except Exception as e:
            self.logger.error(f"Error in generate_content: {e}")
            return {"status": "error", "content": f"Error: {str(e)}", "error": str(e)}

    # Legacy compatibility methods for existing code
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        image_context: Optional[str] = None,
        document_context: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Optional[str]:
        """Legacy method for backward compatibility with temperature and max_tokens support"""
        # Create generation config with provided parameters
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k,
            max_output_tokens=max_tokens,
        )

        # Build content parts
        content_parts = [prompt]

        # Add context if available
        if context:
            context_parts = []
            for msg in context[-5:]:  # Last 5 messages for compatibility
                if msg.get("role") in ["user", "assistant"]:
                    content = msg.get("content", "")
                    if content:
                        context_parts.append(f"{msg['role'].title()}: {content}")
            if context_parts:
                context_text = "\n".join(context_parts)
                content_parts.insert(0, f"Context:\n{context_text}")

        # Build conversation contents
        contents = []
        for part in content_parts:
            if isinstance(part, str):
                contents.append(types.Part.from_text(text=part))
            else:
                contents.append(part)
        
        contents = [types.Content(role="user", parts=contents)]

        try:
            # Generate response with custom config
            response = await self._generate_with_retry(contents, "gemini-2.5-flash", config)

            if response and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]

                # Extract response text
                response_text = ""
                if candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text

                return response_text.strip() if response_text else None
            else:
                return None

        except Exception as e:
            self.logger.error(f"Error in generate_response: {e}")
            return None

    async def analyze_image(
        self, image_data: Union[bytes, io.BytesIO], prompt: str
    ) -> str:
        """Legacy image analysis method"""
        media_input = MediaInput(
            type=MediaType.IMAGE,
            data=image_data,
            mime_type=self.media_processor.get_image_mime_type(image_data),
        )

        result = await self.process_multimodal_input(
            text_prompt=prompt, media_inputs=[media_input]
        )

        return result.content if result.success else f"Error: {result.error}"

    async def handle_multimodal_input(
        self, prompt: str, media_files: Optional[List[Dict]] = None
    ) -> str:
        """Legacy multimodal method"""
        media_inputs = []

        if media_files:
            for media in media_files:
                media_type_map = {
                    "photo": MediaType.IMAGE,
                    "document": MediaType.DOCUMENT,
                    "audio": MediaType.AUDIO,
                    "video": MediaType.VIDEO,
                }

                media_inputs.append(
                    MediaInput(
                        type=media_type_map.get(media["type"], MediaType.DOCUMENT),
                        data=media["data"],
                        mime_type=media.get("mime_type", "application/octet-stream"),
                        filename=media.get("filename"),
                    )
                )

        result = await self.process_multimodal_input(
            text_prompt=prompt, media_inputs=media_inputs
        )

        return result.content if result.success else f"Error: {result.error}"

    async def close(self):
        """Clean up resources"""
        self.logger.info("Gemini API client closed")

    def get_model_indicator(self) -> str:
        """Get the model indicator emoji and name for Gemini models."""
        return "✨ Gemini"


# Utility functions for easy integration
def create_image_input(
    image_data: Union[bytes, io.BytesIO], filename: Optional[str] = None
) -> MediaInput:
    """Create an image media input"""
    return MediaInput(
        type=MediaType.IMAGE,
        data=image_data,
        mime_type=MediaProcessor.get_image_mime_type(image_data),
        filename=filename,
    )


def create_document_input(
    file_data: Union[bytes, io.BytesIO], filename: str
) -> MediaInput:
    """Create a document media input"""
    return MediaInput(
        type=MediaType.DOCUMENT,
        data=file_data,
        mime_type=MediaProcessor.get_document_mime_type(filename),
        filename=filename,
    )


def create_text_input(text: str) -> MediaInput:
    """Create a text media input"""
    return MediaInput(type=MediaType.TEXT, data=text, mime_type="text/plain")


def create_audio_input(
    audio_data: Union[bytes, io.BytesIO], filename: Optional[str] = None
) -> MediaInput:
    """Create an audio media input"""
    return MediaInput(
        type=MediaType.AUDIO,
        data=audio_data,
        mime_type="audio/mpeg",
        filename=filename,
    )


def create_video_input(
    video_data: Union[bytes, io.BytesIO], filename: Optional[str] = None
) -> MediaInput:
    """Create a video media input"""
    return MediaInput(
        type=MediaType.VIDEO,
        data=video_data,
        mime_type="video/mp4",
        filename=filename,
    )
