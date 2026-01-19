"""
Conversation history management.
"""
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Union


class ConversationHistoryMixin:
    """Mixin for conversation history operations."""

    def clear_history(self, user_id: Union[int, str]) -> None:
        """Clear the conversation history for a user."""
        try:
            user_id = str(user_id)
            if self.users_collection is None:
                if user_id in self.user_data_cache:
                    self.user_data_cache[user_id].setdefault("contexts", [])
                    self.user_data_cache[user_id]["contexts"].clear()
                self.logger.info(f"Cleared in-memory history for user: {user_id}")
            else:
                self.users_collection.update_one(
                    {"user_id": user_id}, {"$set": {"contexts": []}}
                )
            if self.conversation_history is not None:
                result = self.conversation_history.delete_many({"user_id": user_id})
                self.logger.info(
                    f"Deleted {result.deleted_count} messages from conversation_history for user {user_id}"
                )
            if user_id in self.user_data_cache:
                del self.user_data_cache[user_id]
            self.logger.info(f"Cleared history for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error clearing history for user {user_id}: {str(e)}")
            raise

    def add_message(self, user_id: Union[int, str], message: Dict[str, Any]) -> None:
        """Add a message dictionary to the user's conversation history."""
        try:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                self.logger.error(f"Invalid message format for user {user_id}: {message}")
                return
            user_id = str(user_id)
            current_time = datetime.now()
            if "timestamp" not in message:
                message["timestamp"] = current_time.timestamp()
            if "message_id" not in message:
                message["message_id"] = f"{message['role']}_{user_id}_{int(current_time.timestamp() * 1000)}"

            if self.conversation_history is not None:
                conversation_doc = {
                    "user_id": user_id,
                    "role": message["role"],
                    "content": message["content"],
                    "timestamp": current_time,
                    "message_id": message["message_id"],
                    "model_used": message.get("model_used", "unknown"),
                    "created_at": current_time,
                }
                self.conversation_history.insert_one(conversation_doc)

            if self.users_collection is None:
                if user_id not in self.user_data_cache:
                    self.user_data_cache[user_id] = {"contexts": []}
                self.user_data_cache[user_id]["contexts"].append(message)
            else:
                self.users_collection.update_one(
                    {"user_id": user_id}, {"$push": {"contexts": message}}, upsert=True
                )
            self.logger.debug(f"Added message to history for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error adding message for user {user_id}: {str(e)}")
            raise

    async def get_user_context(self, user_id: str) -> List[Dict[str, str]]:
        """Retrieve the context for a specific user."""
        try:
            user_id = str(user_id)
            # Try conversation_history collection first
            if self.conversation_history is not None:
                try:
                    cursor = self.conversation_history.find({"user_id": user_id}).sort("timestamp", 1)
                    persistent_history = [
                        {
                            "role": doc.get("role", "unknown"),
                            "content": doc.get("content", ""),
                            "timestamp": doc.get("timestamp", time.time()),
                            "message_id": doc.get("message_id", ""),
                            "model_used": doc.get("model_used", "unknown"),
                        }
                        for doc in cursor
                    ]
                    if persistent_history:
                        self.logger.info(f"Retrieved {len(persistent_history)} messages for user {user_id}")
                        return persistent_history
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve from conversation_history: {e}")

            # Fallback to user contexts
            user_data = await self.get_user_data(user_id)
            if user_data is None:
                return []
            context = user_data.get("contexts", [])
            if not isinstance(context, list):
                self.clear_history(user_id)
                return []
            return [item for item in context if isinstance(item, dict) and "role" in item and "content" in item]
        except Exception as e:
            self.logger.error(f"Error retrieving context for user {user_id}: {str(e)}")
            return []

    async def get_conversation_history(self, user_id: str) -> List[Dict[str, str]]:
        """Retrieve the conversation history for a user."""
        return await self.get_user_context(user_id)

    def reset_conversation(self, user_id: Union[int, str]) -> None:
        """Reset the conversation history for a user."""
        try:
            user_id = str(user_id)
            if self.users_collection is None:
                if user_id in self.user_data_cache:
                    self.user_data_cache[user_id]["conversation_history"] = []
                self.logger.info(f"Reset in-memory conversation for user: {user_id}")
                return
            self.users_collection.update_one(
                {"user_id": user_id}, {"$set": {"conversation_history": []}}
            )
            self.logger.info(f"Reset conversation history for user: {user_id}")
        except Exception as e:
            self.logger.error(f"Error resetting conversation for user {user_id}: {str(e)}")
            raise
