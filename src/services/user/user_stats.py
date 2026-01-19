"""
User statistics management.
"""
import logging
from datetime import datetime, timedelta
from typing import Union


class UserStatsMixin:
    """Mixin for user statistics operations."""

    async def update_stats(
        self,
        user_id: int,
        message: bool = False,
        image: bool = False,
        image_generation: bool = False,
        document: bool = False,
    ):
        """Update user statistics."""
        try:
            if self.db is None:
                self.logger.error("Cannot update stats: Database connection is None")
                return False
            users_collection = self.db.get_collection("users")
            update_dict = {"$set": {"last_active": datetime.now()}, "$inc": {}}
            if message:
                update_dict["$inc"]["messages_count"] = 1
            if image:
                update_dict["$inc"]["images_count"] = 1
            if image_generation:
                update_dict["$inc"]["images_generated_count"] = 1
            if document:
                update_dict["$inc"]["documents_count"] = 1
            users_collection.update_one({"user_id": user_id}, update_dict, upsert=True)
            self.logger.info(f"Updated stats for user: {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating user stats: {str(e)}")
            return False

    def get_user_stats(self, user_id: Union[int, str]) -> dict:
        """Get user statistics from the database."""
        try:
            user_id = str(user_id)
            default_stats = {
                "messages_sent": 0,
                "images_sent": 0,
                "voice_messages_sent": 0,
                "pdf_documents_sent": 0,
            }
            if self.users_collection is None:
                return self.user_data_cache.get(user_id, {}).get("stats", default_stats)
            user_data = self.users_collection.find_one({"user_id": user_id})
            if user_data and "stats" in user_data:
                return user_data["stats"]
            self.update_user_stats(user_id, default_stats)
            return default_stats
        except Exception as e:
            self.logger.error(f"Error getting stats for user {user_id}: {str(e)}")
            raise

    def update_user_stats(self, user_id: Union[int, str], stats: dict) -> None:
        """Update user statistics in the database."""
        try:
            user_id = str(user_id)
            if self.users_collection is None:
                if user_id not in self.user_data_cache:
                    self.user_data_cache[user_id] = {}
                self.user_data_cache[user_id]["stats"] = stats
                self.logger.info(f"Cached stats for user: {user_id}")
                return
            self.users_collection.update_one(
                {"user_id": user_id}, {"$set": {"stats": stats}}, upsert=True
            )
            self.logger.info(f"Updated stats for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating stats for user {user_id}: {str(e)}")
            raise

    def cleanup_inactive_users(self, days_threshold: int = 30) -> None:
        """Remove data for inactive users."""
        try:
            if self.users_collection is None:
                self.logger.warning("cleanup_inactive_users skipped: no DB connection")
                return
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            result = self.users_collection.delete_many(
                {"stats.last_active": {"$lt": threshold_date.isoformat()}}
            )
            self.logger.info(f"Cleaned up {result.deleted_count} inactive users")
        except Exception as e:
            self.logger.error(f"Error during cleanup of inactive users: {str(e)}")
            raise
