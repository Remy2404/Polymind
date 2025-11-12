"""
Gemini API Types and Data Models

This module contains all the type definitions and data models used by the Gemini API client.
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import io


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
    function_calls: Optional[List[ToolCall]] = None
