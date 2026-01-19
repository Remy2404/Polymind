"""
User Data Manager - Facade class combining all user management functionality.

This is the main entry point for user data management, providing backwards
compatibility while using modular components internally.
"""
import sys
import os
import logging
from typing import Dict, List, Any, Union, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.user.user_repository import UserRepository
from src.services.user.user_settings import UserSettingsMixin
from src.services.user.user_stats import UserStatsMixin
from src.services.user.conversation_history import ConversationHistoryMixin
from src.services.user.user_preferences import UserPreferencesMixin
from src.services.user.user_personal_info import UserPersonalInfoMixin
from src.services.user.message_pair import MessagePairMixin


class UserDataManager(
    UserRepository,
    UserSettingsMixin,
    UserStatsMixin,
    ConversationHistoryMixin,
    UserPreferencesMixin,
    UserPersonalInfoMixin,
    MessagePairMixin,
):
    """
    Unified user data manager combining all user management functionality.
    
    This class provides a single interface for:
    - User CRUD operations (from UserRepository)
    - Settings management (from UserSettingsMixin)
    - Statistics tracking (from UserStatsMixin)
    - Conversation history (from ConversationHistoryMixin)
    - User preferences (from UserPreferencesMixin)
    - Personal information (from UserPersonalInfoMixin)
    - Message pair operations (from MessagePairMixin)
    """

    def __init__(self, db):
        """
        Initialize UserDataManager with a database connection.
        :param db: MongoDB database instance
        """
        self.db = db
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches
        self.user_data_cache: Dict[str, Any] = {}
        self.personal_info_cache: Dict[str, Any] = {}
        self.preference_cache: Dict[str, Any] = {}
        
        # Initialize collections
        if self.db is None:
            self.users_collection = None
            self.conversation_history = None
            self.document_history = None
            self.image_analysis_history = None
            self.logger.warning(
                "Database connection is None. Running with limited functionality."
            )
        else:
            self.users_collection = self.db.users
            self.conversation_history = self.db.conversation_history
            self.document_history = self.db.document_history
            self.image_analysis_history = self.db.image_analysis_history
