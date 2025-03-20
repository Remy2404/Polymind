import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Keeping this in case it's needed in future extensions
from pymongo.collection import Collection
from database.connection import get_database
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import warnings

class user_data_manager:
    def __init__(self, db):
        """
        Initialize UserDataManager with a database connection.
        
        :param db: MongoDB database instance
        """
        self.db = db
        # Check if db is None and handle appropriately
        if self.db is None:
            self.users_collection = None
            self.conversation_history = None
            self.document_history = None
            self.image_analysis_history = None
            logging.warning("Database connection is None. Running with limited functionality.")
        else:
            self.users_collection = self.db.users
            self.conversation_history = self.db.conversation_history
            self.document_history = self.db.document_history
            self.image_analysis_history = self.db.image_analysis_history
        self.logger = logging.getLogger(__name__)

    async def initialize_user(self, user_id: int) -> None:
        """Initialize a new user in the database."""
        try:
            user_data = {
                'user_id': user_id,
                'conversation_history': [],
                'settings': {
                    'markdown_enabled': True,
                    'code_suggestions': True
                }
            }
            await self.update_user_data(user_id, user_data)
            self.logger.info(f"Initialized new user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error initializing user {user_id}: {str(e)}")
            raise
    def update_stats(self, user_id: int, message: bool = False, image: bool = False, image_generation: bool = False, document: bool = False):
        """Update user statistics."""
        try:
            if self.db is None:
                self.logger.error("Cannot update stats: Database connection is None")
                return
                
            users_collection = self.db.get_collection("users")
            
            # Create update dictionary
            update_dict = {
                "$set": {"last_active": datetime.now()},
                "$inc": {}
            }

            
            # Increment message count if applicable
            if message:
                update_dict["$inc"]["messages_count"] = 1
                
            # Increment image count if applicable
            if image:
                update_dict["$inc"]["images_count"] = 1
                
            # Increment image generation count if applicable
            if image_generation:
                update_dict["$inc"]["images_generated_count"] = 1
                
            # Increment document count if applicable (NEW)
            if document:
                update_dict["$inc"]["documents_count"] = 1
                
            # Update user record
            users_collection.update_one(
                {"user_id": user_id},
                update_dict,
                upsert=True
            )
        except Exception as e:
            logging.error(f"Error updating user stats: {str(e)}")

    async def update_user_data(self, user_id: int, user_data: dict) -> None:
        """Update user data in the database."""
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": user_data},
                upsert=True
            )
            self.logger.info(f"Updated data for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating data for user {user_id}: {str(e)}")
            raise

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
    def add_to_context(self, user_id: str, message: str) -> None:
        warnings.warn("add_to_context is deprecated, use add_message instead", DeprecationWarning, stacklevel=2)
        self.add_message(user_id, message)

    async def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve all data for a specific user.
        """
        try:
            user_data = self.users_collection.find_one({"user_id": user_id})
            if not user_data:
                await self.initialize_user(user_id)  # Fixed: properly await the coroutine
                user_data = self.users_collection.find_one({"user_id": user_id})
            return user_data
        except Exception as e:
            self.logger.error(f"Error retrieving data for user {user_id}: {str(e)}")
            raise
    # Rename this method to avoid conflict with the async version
    def get_user_settings_sync(self, user_id: int) -> dict:
        """Get user settings from the database (synchronous version)."""
        try:
            user_data = self.users_collection.find_one({"user_id": user_id})
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
    async def get_user_context(self, user_id: str) -> List[str]:
        """
        Retrieve the context for a specific user.
        
        :param user_id: Unique identifier for the user
        :return: List of context messages for the user
        """
        user_data = await self.get_user_data(user_id)
        # Add null check before attempting to use .get()
        if user_data is None:
            return []
        return user_data.get("contexts", [])
    
    async def get_conversation_history(self, user_id: str) -> List[str]:
        """
        Retrieve the conversation history for a user.
        
        :param user_id: Unique identifier for the user
        :return: List of conversation context messages
        """
        user_data = await self.get_user_data(user_id)
        return user_data.get("contexts", [])
    
    async def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user settings.
        
        :param user_id: Unique identifier for the user
        :return: Dictionary of user settings
        """
        user_data = await self.get_user_data(user_id)
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

    def get_user_stats(self, user_id: int) -> dict:
        """Get user statistics from the database."""
        try:
            user_data = self.users_collection.find_one({"user_id": user_id})
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
                self.update_user_stats(user_id, stats)
                return stats
        except Exception as e:
            self.logger.error(f"Error getting stats for user {user_id}: {str(e)}")
            raise

    def update_user_stats(self, user_id: int, stats: dict) -> None:
        """Update user statistics in the database."""
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"stats": stats}},
                upsert=True
            )
            self.logger.info(f"Updated stats for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating stats for user {user_id}: {str(e)}")
            raise

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

    # Convert these methods to async since they call async methods
    async def get_user_preference(self, user_id: int, preference_key: str, default=None):
        """Get a user's preference setting."""
        try:
            user_data = await self.get_user_data(user_id)
            if not user_data or 'preferences' not in user_data:
                return default
            return user_data['preferences'].get(preference_key, default)
        except Exception as e:
            self.logger.error(f"Error getting user preference: {e}")
            return default
    
    async def set_user_preference(self, user_id: int, preference_key: str, value):
        """Set a user's preference setting."""
        try:
            # Initialize user if not yet initialized
            await self.initialize_user(user_id)
            
            # Update preference
            update_query = {
                "$set": {f"preferences.{preference_key}": value}
            }
            self.users_collection.update_one({"user_id": user_id}, update_query)
            
            # Also update in-memory cache if we have one
            if hasattr(self, 'user_data_cache') and user_id in self.user_data_cache:
                if 'preferences' not in self.user_data_cache[user_id]:
                    self.user_data_cache[user_id]['preferences'] = {}
                self.user_data_cache[user_id]['preferences'][preference_key] = value
            
            return True
        except Exception as e:
            self.logger.error(f"Error setting user preference: {e}")
            return False
        
db, client = get_database()
UserDataManager = user_data_manager(db)