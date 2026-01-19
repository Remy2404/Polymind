"""
User personal information management.
"""
import re
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional


class UserPersonalInfoMixin:
    """Mixin for user personal info operations."""

    async def update_user_personal_info(self, user_id: Union[int, str], info_key: str, info_value: str) -> bool:
        """Store or update a piece of personal information about a user."""
        try:
            user_id = str(user_id)
            await self.initialize_user(user_id)

            if self.users_collection is None:
                if user_id not in self.personal_info_cache:
                    self.personal_info_cache[user_id] = {}
                self.personal_info_cache[user_id][info_key] = info_value
                self.logger.info(f"Cached personal info '{info_key}' for user {user_id}")
                return True

            self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {f"personal_info.{info_key}": info_value}},
                upsert=True,
            )

            if user_id not in self.personal_info_cache:
                self.personal_info_cache[user_id] = {}
            self.personal_info_cache[user_id][info_key] = info_value
            self.logger.info(f"Updated personal info '{info_key}' for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating personal info for user {user_id}: {e}")
            return False

    async def get_user_personal_info(self, user_id: int, info_key: Optional[str] = None) -> Any:
        """Retrieve personal information for a user."""
        try:
            if user_id in self.personal_info_cache:
                if info_key is None:
                    return self.personal_info_cache[user_id]
                return self.personal_info_cache[user_id].get(info_key)

            user_data = await self.get_user_data(user_id)
            if not user_data or "personal_info" not in user_data:
                return None if info_key else {}

            personal_info = user_data.get("personal_info", {})
            self.personal_info_cache[user_id] = personal_info
            return personal_info.get(info_key) if info_key else personal_info
        except Exception as e:
            self.logger.error(f"Error retrieving personal info for user {user_id}: {e}")
            return None if info_key else {}

    async def extract_personal_info_from_message(self, user_id: int, message_text: str) -> Dict[str, str]:
        """Analyzes a message to extract and store personal information."""
        try:
            extracted_info = {}
            name_patterns = [
                r"(?:my name is|i'm|i am|call me) ([A-Z][a-z]+)",
                r"(?:name's) ([A-Z][a-z]+)",
            ]
            for pattern in name_patterns:
                match = re.search(pattern, message_text, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    await self.update_user_personal_info(user_id, "name", name)
                    extracted_info["name"] = name
                    break
            return extracted_info
        except Exception as e:
            self.logger.error(f"Error extracting personal info: {e}")
            return {}
