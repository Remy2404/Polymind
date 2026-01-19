"""
Gemini API Utility Functions

Helper functions for creating media inputs and processing Gemini API data.
"""

import io
from typing import Union, Optional, Dict, Any
from src.services.types import MediaType, MediaInput


def create_image_input(
    data: Union[bytes, str, io.BytesIO],
    filename: Optional[str] = None,
    mime_type: str = "image/jpeg",
    metadata: Optional[Dict[str, Any]] = None
) -> MediaInput:
    """
    Create a MediaInput for image data.

    Args:
        data: Image data as bytes, string, or BytesIO
        filename: Optional filename for the image
        mime_type: MIME type of the image (default: image/jpeg)
        metadata: Optional metadata dictionary

    Returns:
        MediaInput instance for the image
    """
    return MediaInput(
        type=MediaType.IMAGE,
        data=data,
        mime_type=mime_type,
        filename=filename,
        metadata=metadata
    )


def create_document_input(
    data: Union[bytes, str, io.BytesIO],
    filename: Optional[str] = None,
    mime_type: str = "application/pdf",
    metadata: Optional[Dict[str, Any]] = None
) -> MediaInput:
    """
    Create a MediaInput for document data.

    Args:
        data: Document data as bytes, string, or BytesIO
        filename: Optional filename for the document
        mime_type: MIME type of the document (default: application/pdf)
        metadata: Optional metadata dictionary

    Returns:
        MediaInput instance for the document
    """
    return MediaInput(
        type=MediaType.DOCUMENT,
        data=data,
        mime_type=mime_type,
        filename=filename,
        metadata=metadata
    )
