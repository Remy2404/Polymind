from typing import List, Optional

from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    """User information extracted from init data."""

    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None
    language_code: Optional[str] = None
    is_premium: bool = False
    photo_url: Optional[str] = None


class Message(BaseModel):
    """Single message in AI SDK format."""

    id: str
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    createdAt: str
    model: Optional[str] = None


class ChatSession(BaseModel):
    """Chat session with UUID-based schema."""

    id: str  # UUID format
    title: str
    model: str
    created_at: str
    updated_at: str
    user_id: str
    message_count: int = 0
    messages: List[Message] = []
    pinned: bool = False
    pinned_at: Optional[str] = None


class Attachment(BaseModel):
    """File attachment for messages."""

    name: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    data: str = Field(..., description="Base64 encoded file content")


class ChatMessage(BaseModel):
    """Message sent to AI model."""

    message: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(
        ..., description="Model ID (e.g., 'gemini/gemini-2.0-flash-exp')"
    )
    session_id: Optional[str] = Field(
        None, description="Session UUID (creates new if not provided)"
    )
    include_context: bool = Field(True, description="Include conversation history")
    max_context_messages: int = Field(10, ge=1, le=50)
    attachments: Optional[List[Attachment]] = Field(
        None, description="File attachments (images, documents, etc.)"
    )


class ChatResponse(BaseModel):
    """AI model response."""

    response: str
    model_used: str
    session_id: str
    timestamp: float
    tokens_used: Optional[int] = None


class ModelInfo(BaseModel):
    """Available AI model information."""

    id: str
    name: str
    provider: str
    supports_vision: bool
    supports_tools: bool
    context_length: int
    is_free: bool
    accessible: bool = True


class UserPreferences(BaseModel):
    """User preferences settings."""

    preferred_model: Optional[str] = None
    theme: str = Field("light", pattern="^(light|dark)$")
    language: str = Field("en", min_length=2, max_length=5)
    notifications_enabled: bool = True
