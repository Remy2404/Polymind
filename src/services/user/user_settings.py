"""
User settings management.
"""
import logging
from typing import Dict, Any, Union


class UserSettingsMixin:
    """Mixin for user settings operations."""

    def get_user_settings_sync(self, user_id: Union[int, str]) -> dict:
        """Get user settings from the database (synchronous version)."""
        try:
            user_id = str(user_id)
            if self.users_collection is None:
                cached = self.user_data_cache.get(user_id, {})
                return cached.get("settings", {"markdown_enabled": True, "code_suggestions": True})
            user_data = self.users_collection.find_one({"user_id": user_id})
            if user_data and "settings" in user_data:
                return user_data["settings"]
            return {"markdown_enabled": True, "code_suggestions": True}
        except Exception as e:
            self.logger.error(f"Error getting settings for user {user_id}: {str(e)}")
            raise

    async def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user settings."""
        user_data = await self.get_user_data(user_id)
        return user_data.get("settings", {})

    def update_user_settings(self, user_id: Union[int, str], new_settings: Dict[str, Any]) -> None:
        """Update user settings (synchronous)."""
        try:
            user_id = str(user_id)
            current_settings = self.get_user_settings_sync(user_id)
            current_settings.update(new_settings)

            if self.users_collection is None:
                if user_id not in self.user_data_cache:
                    self.user_data_cache[user_id] = {}
                self.user_data_cache[user_id].setdefault("settings", {})
                self.user_data_cache[user_id]["settings"].update(new_settings)
                self.logger.info(f"Updated in-memory settings for user: {user_id}")
                return

            self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"settings": current_settings}},
                upsert=True,
            )
            self.logger.info(f"Updated settings for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating settings for user {user_id}: {str(e)}")
            raise

    async def update_user_settings_async(self, user_id: Union[int, str], new_settings: Dict[str, Any]) -> None:
        """Update user settings asynchronously."""
        try:
            user_id = str(user_id)
            user_data = await self.get_user_data(user_id)
            current_settings = user_data.get("settings", {})
            current_settings.update(new_settings)

            if self.users_collection is None:
                if user_id not in self.user_data_cache:
                    self.user_data_cache[user_id] = {}
                self.user_data_cache[user_id].setdefault("settings", {})
                self.user_data_cache[user_id]["settings"].update(new_settings)
            else:
                self.users_collection.update_one(
                    {"user_id": user_id},
                    {"$set": {"settings": current_settings}},
                    upsert=True,
                )

            if hasattr(self, "user_data_cache") and user_id in self.user_data_cache:
                if "settings" not in self.user_data_cache[user_id]:
                    self.user_data_cache[user_id]["settings"] = {}
                self.user_data_cache[user_id]["settings"].update(new_settings)

            self.logger.info(f"Async updated settings for user: {user_id} - {new_settings}")
        except Exception as e:
            self.logger.error(f"Error updating settings asynchronously for user {user_id}: {str(e)}")
            raise
