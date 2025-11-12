"""
Utility functions for Gemini API

This module contains helper functions for creating media inputs and other utilities.
"""

import io
from typing import Union, Optional

from .types import MediaType, MediaInput


def create_image_input(
    image_data: Union[bytes, io.BytesIO], filename: Optional[str] = None
) -> MediaInput:
    """
    Create a MediaInput object for image data.

    Args:
        image_data: Image data as bytes or BytesIO
        filename: Optional filename for the image

    Returns:
        MediaInput object for the image
    """
    if isinstance(image_data, io.BytesIO):
        image_data.seek(0)
        data = image_data.getvalue()
    else:
        data = image_data

    mime_type = "image/jpeg"
    if filename:
        if filename.lower().endswith((".png", ".PNG")):
            mime_type = "image/png"
        elif filename.lower().endswith((".webp", ".WEBP")):
            mime_type = "image/webp"
        elif filename.lower().endswith((".gif", ".GIF")):
            mime_type = "image/gif"

    return MediaInput(
        type=MediaType.IMAGE,
        data=data,
        mime_type=mime_type,
        filename=filename,
    )


def create_document_input(
    document_data: Union[bytes, io.BytesIO], filename: str
) -> MediaInput:
    """
    Create a MediaInput object for document data.

    Args:
        document_data: Document data as bytes or BytesIO
        filename: Filename of the document (used to determine MIME type)

    Returns:
        MediaInput object for the document
    """
    if isinstance(document_data, io.BytesIO):
        document_data.seek(0)
        data = document_data.getvalue()
    else:
        data = document_data

    mime_type = "application/octet-stream"
    if filename:
        ext = filename.lower().split(".")[-1]
        mime_mapping = {
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
        mime_type = mime_mapping.get(ext, mime_type)

    return MediaInput(
        type=MediaType.DOCUMENT,
        data=data,
        mime_type=mime_type,
        filename=filename,
    )
