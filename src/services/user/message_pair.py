"""
Message pair operations for web app memory context.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional


class MessagePairMixin:
    """Mixin for message pair operations."""

    async def save_message_pair(
        self,
        user_id: Union[int, str],
        user_message: str,
        assistant_message: str,
        model_id: Optional[str] = None,
    ) -> None:
        """Save a user-assistant message pair for web app memory context."""
        try:
            user_id = str(user_id)
            timestamp = datetime.now()

            if self.db is None:
                if user_id not in self.user_data_cache:
                    self.user_data_cache[user_id] = {"contexts": []}
                self.user_data_cache[user_id]["contexts"].extend([
                    {"role": "user", "content": user_message, "timestamp": timestamp.isoformat()},
                    {"role": "assistant", "content": assistant_message, "timestamp": timestamp.isoformat(), "model_id": model_id},
                ])
                return

            messages_to_add = [
                {"role": "user", "content": user_message, "timestamp": timestamp.isoformat()},
                {"role": "assistant", "content": assistant_message, "timestamp": timestamp.isoformat(), "model_id": model_id or "unknown"},
            ]

            if self.users_collection is not None:
                self.users_collection.update_one(
                    {"user_id": user_id},
                    {"$push": {"contexts": {"$each": messages_to_add}}, "$set": {"last_updated": datetime.now()}},
                    upsert=True,
                )

            if user_id in self.user_data_cache:
                self.user_data_cache[user_id]["contexts"] = self.user_data_cache[user_id].get("contexts", [])
                self.user_data_cache[user_id]["contexts"].extend(messages_to_add)

            self.logger.debug(f"Saved message pair for user {user_id} with model {model_id}")
        except Exception as e:
            self.logger.error(f"Error saving message pair for user {user_id}: {str(e)}")

    async def add_message_async(self, user_id: Union[int, str], message: Dict[str, Any]) -> None:
        """Async version of add_message with enhanced error handling."""
        try:
            user_id = str(user_id)
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                self.logger.error(f"Invalid message format for user {user_id}: {message}")
                return

            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()

            if self.db is None or self.users_collection is None:
                if user_id not in self.user_data_cache:
                    self.user_data_cache[user_id] = {"contexts": []}
                self.user_data_cache[user_id]["contexts"].append(message)
                return

            self.users_collection.update_one(
                {"user_id": user_id},
                {"$push": {"contexts": message}, "$set": {"last_updated": datetime.now()}},
                upsert=True,
            )

            if user_id in self.user_data_cache:
                if "contexts" not in self.user_data_cache[user_id]:
                    self.user_data_cache[user_id]["contexts"] = []
                self.user_data_cache[user_id]["contexts"].append(message)

            self.logger.debug(f"Added message to context for user {user_id}")
        except Exception as e:
            self.logger.error(f"Error adding message for user {user_id}: {str(e)}")
