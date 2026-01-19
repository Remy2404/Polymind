"""
Message handlers module.

This module provides modular components for message handling:
- ModelHelpersMixin: Model configuration and AI response generation
- TextMessageHandlerMixin: Text message handling
- ImageMessageHandlerMixin: Image message handling
- VoiceMessageHandlerMixin: Voice message handling
- DocumentMessageHandlerMixin: Document message handling
"""
from src.handlers.message.model_helpers import ModelHelpersMixin
from src.handlers.message.text_handler import TextMessageHandlerMixin
from src.handlers.message.image_handler import ImageMessageHandlerMixin
from src.handlers.message.voice_handler import VoiceMessageHandlerMixin
from src.handlers.message.document_handler import DocumentMessageHandlerMixin

__all__ = [
    "ModelHelpersMixin",
    "TextMessageHandlerMixin",
    "ImageMessageHandlerMixin",
    "VoiceMessageHandlerMixin",
    "DocumentMessageHandlerMixin",
]
