"""
Modern Gemini 2.0 Flash API Integration
Handles combined multimodal inputs: text + images + documents + audio + video
Clean, maintainable, and scalable implementation
"""

import logging
import asyncio
import base64
import io
import os
import time
from typing import Optional, List, Dict, Any, Union, BinaryIO
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,
    ServiceUnavailable,
    GoogleAPIError,
)
from PIL import Image, UnidentifiedImageError

from services.rate_limiter import RateLimiter
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
class ProcessingResult:
    """Result of multimodal processing"""

    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


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
    Modern Gemini 2.0 Flash API client with multimodal support
    Handles text, images, documents, audio, and video in combined requests
    """

    def __init__(self, rate_limiter: RateLimiter):
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = rate_limiter
        self.media_processor = MediaProcessor()

        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)

        # Generation configuration optimized for 2.0 Flash
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        self.logger.info("Gemini 2.0 Flash API initialized successfully")

    async def process_multimodal_input(
        self,
        text_prompt: str,
        media_inputs: Optional[List[MediaInput]] = None,
        context: Optional[List[Dict]] = None,
        model_name: str = "gemini-2.0-flash-exp",
    ) -> ProcessingResult:
        """
        Process combined multimodal input (text + images + documents + audio + video)

        Args:
            text_prompt: The main text prompt
            media_inputs: List of media inputs (images, documents, etc.)
            context: Conversation context
            model_name: Gemini model to use

        Returns:
            ProcessingResult with the generated response
        """
        try:
            await self.rate_limiter.acquire()

            # Build content parts for Gemini
            content_parts = []

            # Add system context if available
            system_context = self._build_system_context(context)
            if system_context:
                content_parts.append(system_context)

            # Process media inputs
            if media_inputs:
                for media in media_inputs:
                    processed_content = await self._process_media_input(media)
                    if processed_content:
                        content_parts.extend(processed_content)

            # Add the main text prompt
            content_parts.append(text_prompt)

            # Generate response using Gemini
            response = await self._generate_with_retry(content_parts, model_name)

            if response and hasattr(response, "text") and response.text:
                return ProcessingResult(
                    success=True,
                    content=response.text.strip(),
                    metadata={
                        "model": model_name,
                        "media_count": len(media_inputs) if media_inputs else 0,
                        "token_count": (
                            len(response.text.split()) if response.text else 0
                        ),
                    },
                )
            else:
                return ProcessingResult(
                    success=False, error="Empty response from Gemini API"
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

    async def _process_image_input(self, media: MediaInput) -> Optional[List[Dict]]:
        """Process image input for Gemini"""
        try:
            # Validate image
            if not self.media_processor.validate_image(media.data):
                return [f"[Invalid image file: {media.filename or 'unknown'}]"]

            # Optimize image
            optimized_image = self.media_processor.optimize_image(media.data)
            mime_type = self.media_processor.get_image_mime_type(optimized_image)

            # Convert to base64 for Gemini
            optimized_image.seek(0)
            image_bytes = optimized_image.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Return Gemini-compatible image part
            return [{"inline_data": {"mime_type": mime_type, "data": image_b64}}]

        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            return [f"[Image processing failed: {media.filename or 'unknown'}]"]

    async def _process_document_input(self, media: MediaInput) -> Optional[List[Any]]:
        """Process document input for Gemini"""
        try:
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

            # For supported document formats, upload to Gemini File API
            if mime_type in [
                "application/pdf",
                "text/plain",
                "text/markdown",
                "application/json",
                "text/html",
                "text/csv",
            ]:
                try:
                    # Upload file to Gemini File API
                    uploaded_file = await self._upload_file_to_gemini(
                        doc_bytes, mime_type, media.filename
                    )
                    return [uploaded_file]
                except Exception as upload_error:
                    self.logger.error(f"File upload failed: {upload_error}")
                    return [f"[Document: {media.filename} - upload failed]"]
            else:
                # For unsupported formats, provide description
                return [
                    f"[Document: {media.filename} ({mime_type}) - content preview not available]"
                ]

        except Exception as e:
            self.logger.error(f"Document processing failed: {e}")
            return [f"[Document processing failed: {media.filename or 'unknown'}]"]

    async def _upload_file_to_gemini(
        self, file_bytes: bytes, mime_type: str, filename: str
    ) -> Dict:
        """Upload file to Gemini File API"""
        try:
            # Use Gemini's file upload API
            file_data = io.BytesIO(file_bytes)

            # Upload using genai.upload_file
            uploaded_file = await asyncio.to_thread(
                genai.upload_file, file_data, mime_type=mime_type, display_name=filename
            )

            return uploaded_file

        except Exception as e:
            self.logger.error(f"Gemini file upload failed: {e}")
            raise

    def _build_system_context(self, context: Optional[List[Dict]]) -> Optional[str]:
        """Build system context from conversation history"""
        if not context:
            return None

        system_msg = (
            "You are Gemini, Google's advanced multimodal AI assistant. You can analyze "
            "text, images, documents, and other media types. Provide helpful, accurate, "
            "and detailed responses based on all provided content."
        )

        # Add recent conversation context (last 10 messages to avoid token limits)
        if context:
            conversation_text = "\n\nRecent conversation context:\n"
            recent_context = context[-10:] if len(context) > 10 else context

            for msg in recent_context:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    conversation_text += (
                        f"{role}: {content[:200]}...\n"
                        if len(content) > 200
                        else f"{role}: {content}\n"
                    )

            system_msg += conversation_text

        return system_msg

    async def _generate_with_retry(
        self, content_parts: List[Any], model_name: str, max_retries: int = 3
    ) -> Any:
        """Generate content with retry logic"""
        model = genai.GenerativeModel(model_name)

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    model.generate_content,
                    content_parts,
                    generation_config=self.generation_config,
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

    # Legacy compatibility methods for existing code
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        image_context: Optional[str] = None,
        document_context: Optional[str] = None,
    ) -> Optional[str]:
        """Legacy method for backward compatibility"""
        result = await self.process_multimodal_input(
            text_prompt=prompt, context=context
        )
        return result.content if result.success else None

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
