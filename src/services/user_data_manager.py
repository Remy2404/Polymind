from motor.motor_asyncio import AsyncIOMotorCollection
from datetime import datetime, timedelta
from services.rate_limiter import UserRateLimiter
from typing import Dict, List, Any
import logging
import warnings

class UserDataManager:
    def __init__(self, db):
        """
        Initialize UserDataManager with a database connection.

        :param db: MongoDB database instance
        """
        self.db = db
        self.users_collection: AsyncIOMotorCollection = self.db.users
        self.rate_limiter = UserRateLimiter(requests_per_hour=5)# 5  img requests per hour
        self.logger = logging.getLogger(__name__)
    async def update_user_data(self, user_id: int, user_data: dict) -> None:
        """Update user data in the database."""
        try:
            result = self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": user_data},
                upsert=True
            )
            self.logger.info(f"Updated data for user: {user_id}")
            return result
        except Exception as e:
            self.logger.error(f"Error updating data for user {user_id}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing user {user_id}: {str(e)}")
            raise

    async def initialize_user(self, user_id: int) -> None:
        """Initialize a new user in the database."""
        try:
            user_data = {
                'user_id': user_id,
                'conversation_history': [],
                'settings': {
                    'markdown_enabled': True,
                    'code_suggestions': True,
                    'image_suggestions': True,
                    'image_generation': True,
                    'file_sharing': True,
                    'language_detection': True,
                    'search_engine_results': True,
                },
                'image_cache': {},
                'stats': {},
                'last_active': datetime.now().isoformat(),
                'last_image_generation': datetime.now().isoformat(),
                'last_image_sharing': datetime.now().isoformat(),
            }
            await self.update_user_data(user_id, user_data)
            self.logger.info(f"Initialized new user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error initializing user {user_id}: {str(e)}")
            raise

    async def get_user_data(self, user_id: int) -> Dict[str, Any]:
        try:
            user_data = await self.users_collection.find_one({"user_id": user_id})
            if not user_data:
                user_data = {"user_id": user_id}
                await self.users_collection.insert_one(user_data)
                user_data = await self.users_collection.find_one({"user_id": user_id})
            return user_data
        except Exception as e:
            self.logger.error(f"Error getting user data: {e}")
            return {}

    async def acquire_rate_limit(self, user_id: int):
        await self.rate_limiter.acquire_user(user_id)

    async def get_user_capacity(self, user_id: int) -> float:
        return await self.rate_limiter.get_user_capacity(user_id)

    async def update_stats(self, user_id: str, text_message: bool = False, 
                          voice_message: bool = False, image: bool = False,
                          generated_images: bool = False) -> None:
        try:
            user = await self.get_user_data(user_id)
            stats = user.get('stats', {})
            stats['last_active'] = datetime.now().isoformat()

            stats.setdefault('messages', 0)
            stats.setdefault('voice_messages', 0)
            stats.setdefault('images', 0)
            stats.setdefault('generated_images', 0)

            if text_message:
                stats['messages'] += 1
            if voice_message:
                stats['voice_messages'] += 1
            if image:
                stats['images'] += 1
            if generated_images:
                stats['generated_images'] += 1

            self.users_collection.update_one(
                {"user_id": user_id}, 
                {"$set": {"stats": stats}},
                upsert=True
            )
            self.logger.debug(f"Updated stats for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating stats for user {user_id}: {e}")

    def clear_history(self, user_id: str) -> None:
        """
        Clear the conversation history for a user.

        :param user_id: Unique identifier for the user
        """
        try:
            self.users_collection.update_one(
                {"user_id": user_id}, 
                {"$set": {"contexts": []}}
            )
            self.logger.info(f"Cleared history for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error clearing history for user {user_id}: {str(e)}")
            raise

    def add_message(self, user_id: str, message: str) -> None:
        """
        Add a message to the user's conversation history.

        :param user_id: Unique identifier for the user
        :param message: Message to be added to the history
        """
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$push": {"contexts": message}}
            )
            self.logger.debug(f"Added message to history for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error adding message for user {user_id}: {str(e)}")
            raise

    async def set_user_context(self, user_id: int, context: List[Dict[str, str]]):
        user_data = await self.get_user_data(user_id)
        user_data['context'] = context
        await self.save_user_data(user_id, user_data)

    async def add_to_context(self, user_id: int, message: Dict[str, str]):
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$push": {"context": message}}
            )
        except Exception as e:
            self.logger.error(f"Error adding to context: {e}")

    async def get_user_settings(self, user_id: int) -> dict:
        """Get user settings from the database."""
        try:
            user_data = await self.users_collection.find_one({"user_id": user_id})
            if user_data and 'settings' in user_data:
                return user_data['settings']
            else:
                return {
                    'markdown_enabled': True,
                    'code_suggestions': True
                }
        except Exception as e:
            self.logger.error(f"Error getting settings for user {user_id}: {str(e)}")
            raise

    async def get_user_context(self, user_id: int) -> List[Dict[str, str]]:
        user_data = await self.get_user_data(user_id)  # Ensure this is awaited
        return user_data.get('context', [])

    def get_conversation_history(self, user_id: str) -> List[str]:
        """
        Retrieve the conversation history for a user.

        :param user_id: Unique identifier for the user
        :return: List of conversation context messages
        """
        user_data = self.get_user_data(user_id)
        return user_data.get("contexts", [])

    def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user settings.

        :param user_id: Unique identifier for the user
        :return: Dictionary of user settings
        """
        user_data = self.get_user_data(user_id)
        return user_data.get('settings', {})

    def update_user_settings(self, user_id: str, new_settings: Dict[str, Any]) -> None:
        """
        Update user settings.

        :param user_id: Unique identifier for the user
        :param new_settings: Dictionary of settings to update
        """
        try:
            current_settings = self.get_user_settings(user_id)
            current_settings.update(new_settings)
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"settings": current_settings}}
            )
            self.logger.info(f"Updated settings for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating settings for user {user_id}: {str(e)}")
            raise

    def cleanup_inactive_users(self, days_threshold: int = 30) -> None:
        """
        Remove data for inactive users.

        :param days_threshold: Number of days of inactivity before cleanup
        """
        try:
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            result = self.users_collection.delete_many(
                {"stats.last_active": {"$lt": threshold_date.isoformat()}}
            )
            self.logger.info(f"Cleaned up {result.deleted_count} inactive users")
        except Exception as e:
            self.logger.error(f"Error during cleanup of inactive users: {str(e)}")
            raise

    async def get_user_stats(self, user_id: int) -> dict:
        """Get user statistics from the database."""
        try:
            user_data = await self.users_collection.find_one({"user_id": user_id})
            if user_data and 'stats' in user_data:
                return user_data['stats']
            else:
                # Initialize stats if they do not exist
                stats = {
                    'messages_sent': 0,
                    'images_sent': 0,
                    'voice_messages_sent': 0,
                    'pdf_documents_sent': 0
                }
                await self.update_user_stats(user_id, stats)
                return stats
        except Exception as e:
            self.logger.error(f"Error getting stats for user {user_id}: {str(e)}")
            raise

    async def update_user_stats(self, user_id: int, stats: dict) -> None:
        """Update user statistics in the database."""
        try:
            await self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"stats": stats}},
                upsert=True
            )
            self.logger.info(f"Updated stats for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating stats for user {user_id}: {str(e)}")
            raise

    async def save_user_data(self, user_id: int, data: dict) -> None:
        """Save user data to the database."""
        try:
            await self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": data},
                upsert=True
            )
            self.logger.info(f"Saved data for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error saving data for user {user_id}: {str(e)}")
            raise

    async def get_bot_identity(self, user_id: int) -> str:
        """Retrieve the bot identity for a user."""
        user_data = await self.get_user_data(user_id)
        return user_data.get("bot_identity", "Gembot developer by Ramy")

    def reset_conversation(self, user_id: int) -> None:
        """Reset the conversation history for a user."""
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"conversation_history": []}}
            )
            self.logger.info(f"Reset conversation history for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error resetting conversation history for user {user_id}: {str(e)}")
            raise