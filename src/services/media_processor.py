"""
Media Processing Module for Gemini API

This module handles validation, optimization, and processing of different media types
for use with the Gemini multimodal API.
"""

import io
import logging
from typing import Union, Optional
from PIL import Image

from .types import MediaType, MediaInput


class MediaProcessor:
    """Handles processing of different media types for Gemini"""

    MAX_IMAGE_SIZE = 4096
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}
    IMAGE_QUALITY = 85

    DOCUMENT_MIME_TYPES = {
        "pdf": "application/pdf",
        "doc": "application/msword",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "ppt": "application/vnd.ms-powerpoint",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "xls": "application/vnd.ms-excel",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "txt": "text/plain",
        "csv": "text/csv",
        "md": "text/markdown",
        "html": "text/html",
        "json": "application/json",
        "xml": "text/xml",
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
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data

            if len(img_bytes) > MediaProcessor.MAX_FILE_SIZE:
                return False

            with Image.open(io.BytesIO(img_bytes)) as img:
                if img.format not in MediaProcessor.SUPPORTED_IMAGE_FORMATS:
                    return False
                if img.width > MediaProcessor.MAX_IMAGE_SIZE or img.height > MediaProcessor.MAX_IMAGE_SIZE:
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def optimize_image(image_data: Union[bytes, io.BytesIO]) -> io.BytesIO:
        """Optimize image for Gemini processing"""
        try:
            if isinstance(image_data, io.BytesIO):
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data

            with Image.open(io.BytesIO(img_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'L'):
                    img = img.convert('RGB')

                # Resize if too large
                if img.width > MediaProcessor.MAX_IMAGE_SIZE or img.height > MediaProcessor.MAX_IMAGE_SIZE:
                    img.thumbnail((MediaProcessor.MAX_IMAGE_SIZE, MediaProcessor.MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)

                # Save optimized image
                output = io.BytesIO()
                img.save(output, format='JPEG', quality=MediaProcessor.IMAGE_QUALITY, optimize=True)
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
                img_bytes = image_data.getvalue()
            else:
                img_bytes = image_data

            with Image.open(io.BytesIO(img_bytes)) as img:
                if img.format == 'JPEG':
                    return 'image/jpeg'
                elif img.format == 'PNG':
                    return 'image/png'
                elif img.format == 'WEBP':
                    return 'image/webp'
                elif img.format == 'GIF':
                    return 'image/gif'
                else:
                    return 'image/jpeg'
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

            if size > 50 * 1024 * 1024:  # 50MB limit for documents
                return False
            return True
        except Exception:
            return False