import logging
import time
import re
from typing import Dict, Any, Optional
from pymongo.database import Database

logger = logging.getLogger(__name__)


class UserProfileManager:
    """Manages user profile data and information extraction"""

    def __init__(self, db: Optional[Database] = None):
        self.db = db
        self.user_profiles_collection = db.user_profiles if db is not None else None

    async def save_user_profile(self, user_id: int, profile_data: Dict[str, Any]):
        """Save user profile information to MongoDB"""
        try:
            if self.db is None:
                logger.warning("No database connection for user profile storage")
                return

            # Store user profile data including name and other personal info
            profile_document = {
                "user_id": user_id,
                "profile_data": profile_data,
                "last_updated": time.time(),
                "created_at": profile_data.get("created_at", time.time()),
            }

            # Update or insert user profile
            self.user_profiles_collection.update_one(
                {"user_id": user_id}, {"$set": profile_document}, upsert=True
            )

            logger.info(f"Saved user profile for user {user_id}")

        except Exception as e:
            logger.error(f"Error saving user profile for {user_id}: {e}")

    async def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve user profile information from MongoDB"""
        try:
            if self.db is None:
                logger.warning("No database connection for user profile retrieval")
                return None

            profile_doc = self.user_profiles_collection.find_one({"user_id": user_id})

            if profile_doc:
                return profile_doc.get("profile_data", {})

            return None

        except Exception as e:
            logger.error(f"Error retrieving user profile for {user_id}: {e}")
            return None

    async def update_user_profile_field(self, user_id: int, field: str, value: Any):
        """Update a specific field in user profile"""
        try:
            if self.db is None:
                logger.warning("No database connection for user profile update")
                return

            # Update specific field in profile data
            self.user_profiles_collection.update_one(
                {"user_id": user_id},
                {"$set": {f"profile_data.{field}": value, "last_updated": time.time()}},
                upsert=True,
            )

            logger.info(f"Updated {field} for user {user_id}")

        except Exception as e:
            logger.error(
                f"Error updating user profile field {field} for {user_id}: {e}"
            )

    async def extract_and_save_user_info(self, user_id: int, message_content: str):
        """Extract and save user information from message content"""
        try:
            # Simple pattern matching for name extraction
            name_patterns = [
                r"my name is (\w+)",
                r"i'm (\w+)",
                r"i am (\w+)",
                r"call me (\w+)",
                r"name's (\w+)",
            ]

            for pattern in name_patterns:
                match = re.search(pattern, message_content.lower())
                if match:
                    name = match.group(1).capitalize()
                    await self.update_user_profile_field(user_id, "name", name)
                    logger.info(f"Extracted and saved name '{name}' for user {user_id}")
                    break

        except Exception as e:
            logger.error(f"Error extracting user info from message: {e}")
