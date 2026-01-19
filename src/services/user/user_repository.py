"""
Base user repository for core CRUD operations.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional


class UserRepository:
    """Core user CRUD operations."""

    def __init__(self, db):
        """
        Initialize UserRepository with database connection.
        :param db: MongoDB database instance
        """
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.user_data_cache: Dict[str, Any] = {}

        if self.db is None:
            self.users_collection = None
            self.logger.warning("Database connection is None. Running with limited functionality.")
        else:
            self.users_collection = self.db.users

    async def initialize_user(self, user_id: Union[int, str]) -> None:
        """Initialize a new user in the database."""
        try:
            user_id = str(user_id)
            user_data = {
                "user_id": user_id,
                "conversation_history": [],
                "settings": {"markdown_enabled": True, "code_suggestions": True},
            }
            await self.update_user_data(user_id, user_data)
            self.logger.info(f"Initialized new user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error initializing user {user_id}: {str(e)}")
            raise

    async def get_user_data(self, user_id: Union[int, str]) -> Dict[str, Any]:
        """Retrieve all data for a specific user."""
        try:
            if self.db is None:
                self.logger.warning("Database connection is None, using cache")
                return self.user_data_cache.get(str(user_id), {})

            user_id = str(user_id)
            if user_id in self.user_data_cache:
                return self.user_data_cache[user_id]

            if self.users_collection is None:
                return self.user_data_cache.get(user_id, {})

            user_data = self.users_collection.find_one({"user_id": user_id})
            if not user_data:
                await self.initialize_user(user_id)
                user_data = self.users_collection.find_one({"user_id": user_id})

            if user_data:
                self.user_data_cache[user_id] = user_data
            return user_data or {}
        except Exception as e:
            self.logger.error(f"Error retrieving data for user {user_id}: {str(e)}")
            return self.user_data_cache.get(str(user_id), {})

    async def update_user_data(self, user_id: Union[int, str], user_data: dict) -> None:
        """Update user data in the database."""
        try:
            user_id = str(user_id)
            user_data["user_id"] = user_id
            if self.users_collection is None:
                self.user_data_cache[user_id] = user_data
                self.logger.info(f"Cached data for user (no DB): {user_id}")
                return
            self.users_collection.update_one(
                {"user_id": user_id}, {"$set": user_data}, upsert=True
            )
            self.logger.info(f"Updated data for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating data for user {user_id}: {str(e)}")
            raise
