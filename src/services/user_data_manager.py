from pymongo import MongoClient
from pymongo.collection import Collection
from bson.objectid import ObjectId
import os
from datetime import datetime

class UserDataManager:
    def __init__(self, db):  # Accept the database instance as an argument
        self.db = db  # Store the database instance for later use
        self.users_collection: Collection = self.db.users  # Reference to the users collection

    def initialize_user(self, user_id: str) -> None:
        """Initialize user data in the database if it doesn't exist."""
        if not self.users_collection.find_one({"user_id": user_id}):
            now = datetime.now().isoformat()
            self.users_collection.insert_one({
                "user_id": user_id,
                "contexts": [],
                "settings": {
                    'markdown_enabled': True,
                    'code_suggestions': True
                },
                "stats": {
                    'messages': 0,
                    'last_active': now,
                    'joined_date': now
                }
            })

    def clear_history(self, user_id: str) -> None:
        """Clear the conversation history for a user."""
        self.users_collection.update_one({"user_id": user_id}, {"$set": {"contexts": []}})

    def add_message(self, user_id: str, message: str) -> None:
        """Add a message to the user's conversation history."""
        self.users_collection.update_one(
            {"user_id": user_id},
            {"$push": {"contexts": {"role": "user", "content": message}}}
        )

    def get_user_data(self, user_id: str) -> dict:
        """Retrieve user data from the database."""
        user_data = self.users_collection.find_one({"user_id": user_id})
        return user_data

    def get_conversation_history(self, user_id: str) -> list:
        """Retrieve the conversation history for a user."""
        user_data = self.get_user_data(user_id)
        return user_data.get("contexts", []) if user_data else []

    def update_user_stats(self, user_id: str, message_count: int = 1) -> None:
        """Update user statistics."""
        user = self.users_collection.find_one({"user_id": user_id})
        if not user:
            self.initialize_user(user_id)
            user = self.users_collection.find_one({"user_id": user_id})

        stats = user['stats']
        stats['messages'] += message_count
        stats['last_active'] = datetime.now().isoformat()
        self.users_collection.update_one({"user_id": user_id}, {"$set": {"stats": stats}})

    def get_user_context(self, user_id: str) -> list:
        """Retrieve the user's context (conversation history) from the database."""
        user_data = self.users_collection.find_one({"user_id": user_id})
        return user_data['contexts'] if user_data else []
    def reset_user_data(self, user_id: str) -> None:
        """Reset the user's data."""
        self.users_collection.update_one({"user_id": user_id}, {"$set": {"contexts": []}})

    def get_user_settings(self, user_id: str) -> dict:
        """Retrieve user settings."""
        user = self.users_collection.find_one({"user_id": user_id})
        return user['settings'] if user else {
            'markdown_enabled': True,
            'code_suggestions': True
        }

    def toggle_setting(self, user_id: str, setting: str) -> None:
        """Toggle a user setting."""
        user = self.users_collection.find_one({"user_id": user_id})
        if not user:
            self.initialize_user(user_id)
            user = self.users_collection.find_one({"user_id": user_id})

        settings = user['settings']
        settings[setting] = not settings.get(setting, False)
        self.users_collection.update_one({"user_id": user_id}, {"$set": {"settings": settings}})

    def update_user_settings(self, user_id: str, new_settings: dict) -> None:
        """Update user settings."""
        user = self.users_collection.find_one({"user_id": user_id})
        if not user:
            self.initialize_user(user_id)
            user = self.users_collection.find_one({"user_id": user_id})

        settings = user['settings']
        settings.update(new_settings)
        self.users_collection.update_one({"user_id": user_id}, {"$set": {"settings": settings}})

    def cleanup_inactive_users(self, days_threshold: int = 30) -> None:
        """Remove users who have been inactive for a specified number of days."""
        current_time = datetime.now()
        users = self.users_collection.find()

        for user in users:
            last_active = datetime.fromisoformat(user['stats']['last_active'])
            if (current_time - last_active).days > days_threshold:
                self.users_collection.delete_one({"user_id": user['user_id']})

    def get_user_statistics(self, user_id: str) -> dict:
        """Retrieve user statistics."""
        user = self.users_collection.find_one({"user_id": user_id})
        return user['stats'] if user else {}