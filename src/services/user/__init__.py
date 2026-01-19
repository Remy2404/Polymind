"""
User data management module.

This module provides modular components for user data management:
- UserRepository: Core CRUD operations
- UserSettingsMixin: Settings management
- UserStatsMixin: Statistics management
- ConversationHistoryMixin: Conversation history
- UserPreferencesMixin: User preferences
- UserPersonalInfoMixin: Personal information
- MessagePairMixin: Message pair operations
"""
from src.services.user.user_repository import UserRepository
from src.services.user.user_settings import UserSettingsMixin
from src.services.user.user_stats import UserStatsMixin
from src.services.user.conversation_history import ConversationHistoryMixin
from src.services.user.user_preferences import UserPreferencesMixin
from src.services.user.user_personal_info import UserPersonalInfoMixin
from src.services.user.message_pair import MessagePairMixin

__all__ = [
    "UserRepository",
    "UserSettingsMixin",
    "UserStatsMixin",
    "ConversationHistoryMixin",
    "UserPreferencesMixin",
    "UserPersonalInfoMixin",
    "MessagePairMixin",
]
