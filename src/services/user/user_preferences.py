"""
User preferences management.
"""
import logging
from typing import Union, Any


class UserPreferencesMixin:
    """Mixin for user preference operations."""

    async def get_user_preference(self, user_id: Union[int, str], preference_key: str, default=None) -> Any:
        """Get a user's preference setting."""
        try:
            user_id = str(user_id)
            # Check cache first
            if hasattr(self, "preference_cache") and user_id in self.preference_cache:
                if preference_key in self.preference_cache[user_id]:
                    return self.preference_cache[user_id][preference_key]

            # Check backup
            if hasattr(self, "preference_backup") and user_id in self.preference_backup:
                if preference_key in self.preference_backup[user_id]:
                    value = self.preference_backup[user_id][preference_key]
                    self._cache_preference(user_id, preference_key, value)
                    return value

            # Get from database
            user_data = await self.get_user_data(user_id)
            if not user_data or "preferences" not in user_data:
                await self.set_user_preference(user_id, preference_key, default)
                return default

            value = user_data["preferences"].get(preference_key, default)
            self._cache_preference(user_id, preference_key, value)
            return value
        except Exception as e:
            self.logger.error(f"Error getting user preference: {e}")
            return default

    async def set_user_preference(self, user_id: Union[int, str], preference_key: str, value: Any) -> bool:
        """Set a user's preference setting."""
        try:
            user_id = str(user_id)
            await self.initialize_user(user_id)

            if self.users_collection is None:
                self._cache_preference(user_id, preference_key, value)
                self._backup_preference(user_id, preference_key, value)
                return True

            result = self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {f"preferences.{preference_key}": value}},
                upsert=True,
            )
            if result.acknowledged:
                self.logger.info(f"Set preference {preference_key} for user {user_id}: {value}")

            self._cache_preference(user_id, preference_key, value)
            self._backup_preference(user_id, preference_key, value)
            return True
        except Exception as e:
            self.logger.error(f"Error setting user preference: {e}")
            return False

    def _cache_preference(self, user_id: str, key: str, value: Any) -> None:
        """Cache a preference value."""
        if not hasattr(self, "preference_cache"):
            self.preference_cache = {}
        if user_id not in self.preference_cache:
            self.preference_cache[user_id] = {}
        self.preference_cache[user_id][key] = value
        # Also update user_data_cache
        if hasattr(self, "user_data_cache") and user_id in self.user_data_cache:
            if "preferences" not in self.user_data_cache[user_id]:
                self.user_data_cache[user_id]["preferences"] = {}
            self.user_data_cache[user_id]["preferences"][key] = value

    def _backup_preference(self, user_id: str, key: str, value: Any) -> None:
        """Backup a preference value."""
        if not hasattr(self, "preference_backup"):
            self.preference_backup = {}
        if user_id not in self.preference_backup:
            self.preference_backup[user_id] = {}
        self.preference_backup[user_id][key] = value
