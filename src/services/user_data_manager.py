import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Keeping this in case it's needed in future extensions
from pymongo.collection import Collection
from database.connection import get_database
from datetime import datetime, timedelta
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
        # Check if db is None and handle appropriately
        if self.db is None:
            self.users_collection = None
            self.conversation_history = None
            self.document_history = None
            self.image_analysis_history = None
            logging.warning(
                "Database connection is None. Running with limited functionality."
            )
        else:
            self.users_collection = self.db.users
            self.conversation_history = self.db.conversation_history
            self.document_history = self.db.document_history
            self.image_analysis_history = self.db.image_analysis_history
        self.logger = logging.getLogger(__name__)

        # Add in-memory cache for user data to improve performance
        self.user_data_cache = {}

        # Add personal information memory
        self.personal_info_cache = {}

        # Add preference memory cache to ensure consistency
        self.preference_cache = {}

    async def initialize_user(self, user_id: int) -> None:
        """Initialize a new user in the database."""
        try:
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
                return

            users_collection = self.db.get_collection("users")

            # Create update dictionary
            update_dict = {"$set": {"last_active": datetime.now()}, "$inc": {}}

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
            users_collection.update_one({"user_id": user_id}, update_dict, upsert=True)
            self.logger.info(f"Updated stats for user: {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating user stats: {str(e)}")
            return False

    async def update_user_data(self, user_id: int, user_data: dict) -> None:
        """Update user data in the database."""
        try:
            self.users_collection.update_one(
                {"user_id": user_id}, {"$set": user_data}, upsert=True
            )
            self.logger.info(f"Updated data for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating data for user {user_id}: {str(e)}")
            raise

    def clear_history(self, user_id: str) -> None:
        """
        Clear the conversation history for a user.
        Now clears both contexts and conversation_history collection.

        :param user_id: Unique identifier for the user
        """
        try:
            # Convert user_id to string for consistency
            user_id = str(user_id)
            
            # Clear user contexts
            self.users_collection.update_one(
                {"user_id": user_id}, {"$set": {"contexts": []}}
            )
            
            # Clear conversation_history collection
            if self.conversation_history is not None:
                result = self.conversation_history.delete_many({"user_id": user_id})
                self.logger.info(f"Deleted {result.deleted_count} messages from conversation_history collection for user {user_id}")
            
            # Clear cache for this user
            if user_id in self.user_data_cache:
                del self.user_data_cache[user_id]
                
            self.logger.info(f"Cleared history for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error clearing history for user {user_id}: {str(e)}")
            raise

    def add_message(self, user_id: str, message: Dict[str, str]) -> None:
        """
        Add a message dictionary to the user's conversation history.
        Now saves to both contexts (for backwards compatibility) and conversation_history collection.

        :param user_id: Unique identifier for the user
        :param message: Message dictionary ({'role': 'user/assistant', 'content': '...'}) to be added
        """
        try:
            # Ensure message is a dictionary with required keys
            if (
                not isinstance(message, dict)
                or "role" not in message
                or "content" not in message
            ):
                self.logger.error(
                    f"Invalid message format for user {user_id}: {message}"
                )
                return

            # Convert user_id to string for consistency
            user_id = str(user_id)
            
            # Add timestamp and message_id if not present
            current_time = datetime.now()
            if "timestamp" not in message:
                message["timestamp"] = current_time.timestamp()
            if "message_id" not in message:
                message["message_id"] = f"{message['role']}_{user_id}_{int(current_time.timestamp() * 1000)}"

            # Save to conversation_history collection for persistence
            if self.conversation_history is not None:
                conversation_doc = {
                    "user_id": user_id,
                    "role": message["role"],
                    "content": message["content"],
                    "timestamp": current_time,
                    "message_id": message["message_id"],
                    "model_used": message.get("model_used", "unknown"),
                    "created_at": current_time
                }
                self.conversation_history.insert_one(conversation_doc)
                self.logger.debug(f"Saved message to conversation_history collection for user: {user_id}")

            # Also save to user contexts for backwards compatibility
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$push": {"contexts": message}},  # Push the dictionary
            )
            self.logger.debug(f"Added message to history for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error adding message for user {user_id}: {str(e)}")
            raise

    def add_to_context(self, user_id: str, message: Dict[str, str]) -> None:
        warnings.warn(
            "add_to_context is deprecated, use add_message instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.add_message(user_id, message)

    async def get_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve all data for a specific user with improved error handling.
        """
        try:
            if self.db is None:
                self.logger.warning("Database connection is None, using cache")
                return self.user_data_cache.get(str(user_id), {})
            
            # Convert user_id to string for consistency
            user_id = str(user_id)
            
            # Check cache first for performance
            if user_id in self.user_data_cache:
                return self.user_data_cache[user_id]
            
            user_data = self.users_collection.find_one({"user_id": user_id})
            if not user_data:
                await self.initialize_user(user_id)
                user_data = self.users_collection.find_one({"user_id": user_id})
            
            # Update cache
            if user_data:
                self.user_data_cache[user_id] = user_data
            
            return user_data or {}
        except Exception as e:
            self.logger.error(f"Error retrieving data for user {user_id}: {str(e)}")
            # Return cached data if available, otherwise empty dict
            return self.user_data_cache.get(str(user_id), {})

    # Rename this method to avoid conflict with the async version
    def get_user_settings_sync(self, user_id: int) -> dict:
        """Get user settings from the database (synchronous version)."""
        try:
            user_data = self.users_collection.find_one({"user_id": user_id})
            if user_data and "settings" in user_data:
                return user_data["settings"]
            else:
                return {"markdown_enabled": True, "code_suggestions": True}
        except Exception as e:
            self.logger.error(f"Error getting settings for user {user_id}: {str(e)}")
            raise

    async def get_user_context(self, user_id: str) -> List[Dict[str, str]]:
        """
        Retrieve the context (list of message dictionaries) for a specific user.
        Enhanced for web app memory persistence - now gets from conversation_history collection first.

        :param user_id: Unique identifier for the user
        :return: List of context message dictionaries for the user
        """
        try:
            # Convert user_id to string for consistency
            user_id = str(user_id)
            
            # First try to get from conversation_history collection (more persistent)
            persistent_history = []
            if self.conversation_history is not None:
                try:
                    cursor = self.conversation_history.find(
                        {"user_id": user_id}
                    ).sort("timestamp", 1)  # Sort by timestamp ascending
                    
                    for doc in cursor:
                        persistent_history.append({
                            "role": doc.get("role", "unknown"),
                            "content": doc.get("content", ""),
                            "timestamp": doc.get("timestamp", time.time()),
                            "message_id": doc.get("message_id", ""),
                            "model_used": doc.get("model_used", "unknown")
                        })
                    
                    if persistent_history:
                        self.logger.info(f"Retrieved {len(persistent_history)} messages from conversation_history collection for user {user_id}")
                        return persistent_history
                        
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve from conversation_history collection: {e}")
            
            # Fallback to user contexts if conversation_history is empty or failed
            user_data = await self.get_user_data(user_id)
            if user_data is None:
                return []
            
            # Get contexts with proper fallback
            context = user_data.get("contexts", [])
            if not isinstance(context, list):
                self.logger.warning(
                    f"User {user_id} context is not a list: {type(context)}. Clearing."
                )
                await self.clear_history(user_id)
                return []
            
            # Filter out any invalid entries and ensure proper format
            valid_context = []
            for item in context:
                if isinstance(item, dict) and ("role" in item and "content" in item):
                    valid_context.append(item)
                elif isinstance(item, dict):
                    # Try to fix legacy format
                    if "user" in item or "assistant" in item:
                        if "user" in item:
                            valid_context.append({"role": "user", "content": item["user"]})
                        if "assistant" in item:
                            valid_context.append({"role": "assistant", "content": item["assistant"]})
            
            self.logger.info(f"Fallback: Retrieved {len(valid_context)} messages from user contexts for user {user_id}")
            return valid_context
            
        except Exception as e:
            self.logger.error(f"Error retrieving context for user {user_id}: {str(e)}")
            return []

    async def get_conversation_history(self, user_id: str) -> List[Dict[str, str]]:
        """
        Retrieve the conversation history for a user.
        Enhanced with better error handling and caching.

        :param user_id: Unique identifier for the user
        :return: List of conversation context message dictionaries
        """
        return await self.get_user_context(user_id)

    async def get_user_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieve user settings.

        :param user_id: Unique identifier for the user
        :return: Dictionary of user settings
        """
        user_data = await self.get_user_data(user_id)
        return user_data.get("settings", {})

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
                {"user_id": user_id}, {"$set": {"settings": current_settings}}
            )
            self.logger.info(f"Updated settings for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating settings for user {user_id}: {str(e)}")
            raise

    async def update_user_settings_async(
        self, user_id: str, new_settings: Dict[str, Any]
    ) -> None:
        """
        Update user settings asynchronously.

        :param user_id: Unique identifier for the user
        :param new_settings: Dictionary of settings to update
        """
        try:
            user_data = await self.get_user_data(user_id)
            current_settings = user_data.get("settings", {})
            current_settings.update(new_settings)

            # Update in database
            self.users_collection.update_one(
                {"user_id": user_id}, {"$set": {"settings": current_settings}}
            )

            # Update in memory cache if exists
            if hasattr(self, "user_data_cache") and user_id in self.user_data_cache:
                if "settings" not in self.user_data_cache[user_id]:
                    self.user_data_cache[user_id]["settings"] = {}
                self.user_data_cache[user_id]["settings"].update(new_settings)

            self.logger.info(
                f"Async updated settings for user: {user_id} - {new_settings}"
            )
        except Exception as e:
            self.logger.error(
                f"Error updating settings asynchronously for user {user_id}: {str(e)}"
            )
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
            if user_data and "stats" in user_data:
                return user_data["stats"]
            else:
                # Initialize stats if they do not exist
                stats = {
                    "messages_sent": 0,
                    "images_sent": 0,
                    "voice_messages_sent": 0,
                    "pdf_documents_sent": 0,
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
                {"user_id": user_id}, {"$set": {"stats": stats}}, upsert=True
            )
            self.logger.info(f"Updated stats for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error updating stats for user {user_id}: {str(e)}")
            raise

    def reset_conversation(self, user_id: int) -> None:
        """Reset the conversation history for a user."""
        try:
            self.users_collection.update_one(
                {"user_id": user_id}, {"$set": {"conversation_history": []}}
            )
            self.logger.info(f"Reset conversation history for user: {user_id}")
        except Exception as e:
            self.logger.error(
                f"Error resetting conversation history for user {user_id}: {str(e)}"
            )
            raise

    # Convert these methods to async since they call async methods
    async def get_user_preference(
        self, user_id: int, preference_key: str, default=None
    ):
        """Get a user's preference setting."""
        try:
            # Check the in-memory preference cache first
            if hasattr(self, "preference_cache") and user_id in self.preference_cache:
                if preference_key in self.preference_cache[user_id]:
                    value = self.preference_cache[user_id][preference_key]
                    self.logger.info(
                        f"Retrieved preference {preference_key} for user {user_id} from cache: {value}"
                    )
                    return value

            # Also check the backup if it exists
            if hasattr(self, "preference_backup") and user_id in self.preference_backup:
                if preference_key in self.preference_backup[user_id]:
                    value = self.preference_backup[user_id][preference_key]
                    self.logger.info(
                        f"Retrieved preference {preference_key} for user {user_id} from backup: {value}"
                    )
                    # Store in primary cache for next time
                    if not hasattr(self, "preference_cache"):
                        self.preference_cache = {}
                    if user_id not in self.preference_cache:
                        self.preference_cache[user_id] = {}
                    self.preference_cache[user_id][preference_key] = value
                    return value

            # If not in cache, try to get from database
            user_data = await self.get_user_data(user_id)

            if not user_data or "preferences" not in user_data:
                self.logger.info(
                    f"No preferences found for user {user_id}, returning default: {default}"
                )
                # Initialize preferences for this user
                await self.set_user_preference(user_id, preference_key, default)
                return default

            value = user_data["preferences"].get(preference_key, default)

            # Store in cache for future fast access
            if not hasattr(self, "preference_cache"):
                self.preference_cache = {}
            if user_id not in self.preference_cache:
                self.preference_cache[user_id] = {}
            self.preference_cache[user_id][preference_key] = value

            self.logger.info(
                f"Retrieved preference {preference_key} for user {user_id} from database: {value}"
            )
            return value
        except Exception as e:
            self.logger.error(f"Error getting user preference: {e}")
            return default

    async def set_user_preference(self, user_id: int, preference_key: str, value):
        """Set a user's preference setting."""
        try:
            # Initialize user if not yet initialized
            await self.initialize_user(user_id)

            # Update preference directly with a dedicated update operation
            # This ensures the preference is properly saved
            result = self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {f"preferences.{preference_key}": value}},
                upsert=True,
            )

            # Check if the update was successful
            if result.acknowledged:
                self.logger.info(
                    f"Set preference {preference_key} for user {user_id} to: {value} (matched: {result.matched_count}, modified: {result.modified_count})"
                )
            else:
                self.logger.warning(
                    f"Preference update not acknowledged for user {user_id}"
                )

            # Always update the preference_cache to ensure consistency
            if not hasattr(self, "preference_cache"):
                self.preference_cache = {}
            if user_id not in self.preference_cache:
                self.preference_cache[user_id] = {}
            self.preference_cache[user_id][preference_key] = value

            # Also update in-memory cache if we have one
            if hasattr(self, "user_data_cache") and user_id in self.user_data_cache:
                if "preferences" not in self.user_data_cache[user_id]:
                    self.user_data_cache[user_id]["preferences"] = {}
                self.user_data_cache[user_id]["preferences"][preference_key] = value

            # Store preference in a persistent in-memory backup
            if not hasattr(self, "preference_backup"):
                self.preference_backup = {}
            if user_id not in self.preference_backup:
                self.preference_backup[user_id] = {}
            self.preference_backup[user_id][preference_key] = value

            return True
        except Exception as e:
            self.logger.error(f"Error setting user preference: {e}")
            return False

    async def update_user_personal_info(
        self, user_id: int, info_key: str, info_value: str
    ) -> bool:
        """
        Store or update a piece of personal information about a user.

        :param user_id: User's unique identifier
        :param info_key: Type of information (name, location, preference, etc.)
        :param info_value: The actual information value
        :return: Success status
        """
        try:
            # Initialize user if not yet initialized
            await self.initialize_user(user_id)

            # Update or create personal info
            update_query = {"$set": {f"personal_info.{info_key}": info_value}}
            self.users_collection.update_one({"user_id": user_id}, update_query)

            # Update in-memory cache
            if user_id not in self.personal_info_cache:
                self.personal_info_cache[user_id] = {}
            self.personal_info_cache[user_id][info_key] = info_value

            self.logger.info(f"Updated personal info '{info_key}' for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating personal info for user {user_id}: {e}")
            return False

    async def get_user_personal_info(self, user_id: int, info_key: str = None) -> Any:
        """
        Retrieve personal information for a user.

        :param user_id: User's unique identifier
        :param info_key: Specific piece of information to retrieve, or None for all
        :return: The requested information or None if not found
        """
        try:
            # Check cache first
            if user_id in self.personal_info_cache:
                if info_key is None:
                    return self.personal_info_cache[user_id]
                return self.personal_info_cache[user_id].get(info_key)

            # If not in cache, get from database
            user_data = await self.get_user_data(user_id)
            if not user_data or "personal_info" not in user_data:
                return None if info_key else {}

            # Update cache
            personal_info = user_data.get("personal_info", {})
            self.personal_info_cache[user_id] = personal_info

            return personal_info.get(info_key) if info_key else personal_info
        except Exception as e:
            self.logger.error(f"Error retrieving personal info for user {user_id}: {e}")
            return None if info_key else {}

    async def extract_personal_info_from_message(
        self, user_id: int, message_text: str
    ) -> Dict[str, str]:
        """
        Analyzes a message to extract and store personal information like name, location, etc.

        :param user_id: User's unique identifier
        :param message_text: Text to analyze
        :return: Dictionary of extracted information
        """
        try:
            extracted_info = {}

            # Extract name
            name_patterns = [
                r"(?:my name is|i'm|i am|call me) ([A-Z][a-z]+)",
                r"(?:name's) ([A-Z][a-z]+)",
            ]

            import re

            for pattern in name_patterns:
                match = re.search(pattern, message_text, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    await self.update_user_personal_info(user_id, "name", name)
                    extracted_info["name"] = name
                    break

            # Can be extended for other types of personal info

            return extracted_info
        except Exception as e:
            self.logger.error(f"Error extracting personal info: {e}")
            return {}

    async def reset_conversation(self, user_id: int) -> None:
        """Reset the conversation history for a user."""
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"contexts": [], "conversation_history": []}},
            )
            self.logger.info(f"Reset conversation history for user: {user_id}")
        except Exception as e:
            self.logger.error(
                f"Error resetting conversation history for user {user_id}: {str(e)}"
            )
            raise

    async def save_message_pair(self, user_id: str, user_message: str, assistant_message: str, model_id: str = None) -> None:
        """
        Save a user-assistant message pair for web app memory context.
        This method ensures proper conversation flow and memory persistence.
        """
        try:
            user_id = str(user_id)
            
            if self.db is None:
                # Store in cache for non-database environments
                if user_id not in self.user_data_cache:
                    self.user_data_cache[user_id] = {"contexts": []}
                
                self.user_data_cache[user_id]["contexts"].extend([
                    {"role": "user", "content": user_message, "timestamp": datetime.now().isoformat()},
                    {"role": "assistant", "content": assistant_message, "timestamp": datetime.now().isoformat(), "model_id": model_id}
                ])
                return
            
            # Add both messages to the conversation history
            timestamp = datetime.now().isoformat()
            messages_to_add = [
                {
                    "role": "user", 
                    "content": user_message, 
                    "timestamp": timestamp
                },
                {
                    "role": "assistant", 
                    "content": assistant_message, 
                    "timestamp": timestamp,
                    "model_id": model_id or "unknown"
                }
            ]
            
            # Update database
            self.users_collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {"contexts": {"$each": messages_to_add}},
                    "$set": {"last_updated": datetime.now()}
                },
                upsert=True
            )
            
            # Update cache
            if user_id in self.user_data_cache:
                self.user_data_cache[user_id]["contexts"] = self.user_data_cache[user_id].get("contexts", [])
                self.user_data_cache[user_id]["contexts"].extend(messages_to_add)
            
            self.logger.debug(f"Saved message pair for user {user_id} with model {model_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving message pair for user {user_id}: {str(e)}")

    async def add_message_async(self, user_id: str, message: Dict[str, str]) -> None:
        """
        Async version of add_message with enhanced error handling.
        """
        try:
            user_id = str(user_id)
            
            # Validate message format
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                self.logger.error(f"Invalid message format for user {user_id}: {message}")
                return
            
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()
            
            if self.db is None:
                # Store in cache
                if user_id not in self.user_data_cache:
                    self.user_data_cache[user_id] = {"contexts": []}
                self.user_data_cache[user_id]["contexts"].append(message)
                return
            
            # Update database
            self.users_collection.update_one(
                {"user_id": user_id},
                {
                    "$push": {"contexts": message},
                    "$set": {"last_updated": datetime.now()}
                },
                upsert=True
            )
            
            # Update cache
            if user_id in self.user_data_cache:
                if "contexts" not in self.user_data_cache[user_id]:
                    self.user_data_cache[user_id]["contexts"] = []
                self.user_data_cache[user_id]["contexts"].append(message)
            
            self.logger.debug(f"Added message to context for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error adding message for user {user_id}: {str(e)}")

    async def save_message_pair(
        self, 
        user_id: str, 
        user_message: str, 
        assistant_message: str, 
        model_id: str = "unknown"
    ) -> None:
        """
        Save a user-assistant conversation pair to the conversation history.
        This ensures proper persistence across sessions.

        :param user_id: Unique identifier for the user
        :param user_message: Content of user's message
        :param assistant_message: Content of assistant's response
        :param model_id: Model that generated the response
        """
        try:
            # Convert user_id to string for consistency
            user_id = str(user_id)
            
            # Save user message
            user_msg = {
                "role": "user",
                "content": user_message,
                "model_used": model_id
            }
            self.add_message(user_id, user_msg)
            
            # Save assistant message
            assistant_msg = {
                "role": "assistant", 
                "content": assistant_message,
                "model_used": model_id
            }
            self.add_message(user_id, assistant_msg)
            
            self.logger.info(f"Saved message pair for user {user_id} with model {model_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving message pair for user {user_id}: {str(e)}")
            raise
