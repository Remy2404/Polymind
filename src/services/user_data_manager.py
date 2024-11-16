import json
import os
from datetime import datetime
from typing import Dict, List, Any

class UserDataManager:
    def __init__(self):
        self.user_contexts: Dict[int, List[Dict[str, str]]] = {}
        self.user_settings: Dict[int, Dict[str, bool]] = {}
        self.user_stats: Dict[int, Dict[str, Any]] = {}
        self.data_file = "data/user_data.json"
        self.load_data()

    def load_data(self) -> None:
        """Load user data from persistent storage"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.user_contexts = data.get('contexts', {})
                self.user_settings = data.get('settings', {})
                self.user_stats = data.get('stats', {})

    def save_data(self) -> None:
        """Save user data to persistent storage"""
        with open(self.data_file, 'w') as f:
            json.dump({
                'contexts': self.user_contexts,
                'settings': self.user_settings,
                'stats': self.user_stats
            }, f, indent=4)

    def validate_user_id(self, user_id: int) -> bool:
        """Validate user ID format"""
        if not isinstance(user_id, int):
            raise ValueError("User ID must be an integer")
        return True

    def initialize_user(self, user_id: int) -> None:
        """Initialize new user data"""
        self.validate_user_id(user_id)
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = []
        if user_id not in self.user_settings:
            self.user_settings[user_id] = {
                'markdown_enabled': True,
                'code_suggestions': True
            }
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {
                'messages': 0,
                'last_active': datetime.now().isoformat(),
                'joined_date': datetime.now().isoformat()
            }
        self.save_data()

    def update_user_stats(self, user_id: int, message_count: int = 1) -> None:
        """Update user activity statistics"""
        self.validate_user_id(user_id)
        if user_id not in self.user_stats:
            self.initialize_user(user_id)
        
        self.user_stats[user_id]['messages'] += message_count
        self.user_stats[user_id]['last_active'] = datetime.now().isoformat()
        self.save_data()

    def get_user_context(self, user_id: int) -> list:
        """Get user's conversation context"""
        self.validate_user_id(user_id)
        return self.user_contexts.get(user_id, [])

    def update_user_context(self, user_id: int, message: str, response: str) -> None:
        """Update user's conversation context"""
        self.validate_user_id(user_id)
        if user_id not in self.user_contexts:
            self.initialize_user(user_id)
        
        self.user_contexts[user_id].append({"role": "user", "content": message})
        self.user_contexts[user_id].append({"role": "assistant", "content": response})
        
        if len(self.user_contexts[user_id]) > 20:
            self.user_contexts[user_id] = self.user_contexts[user_id][-20:]
        
        self.update_user_stats(user_id)
        self.save_data()

    def reset_user_data(self, user_id: int) -> None:
        """Reset user's conversation history"""
        self.validate_user_id(user_id)
        self.user_contexts[user_id] = []
        self.save_data()

    def get_user_settings(self, user_id: int) -> dict:
        """Get user's settings"""
        self.validate_user_id(user_id)
        return self.user_settings.get(user_id, {
            'markdown_enabled': True,
            'code_suggestions': True
        })

    def toggle_setting(self, user_id: int, setting: str) -> None:
        """Toggle a user setting"""
        self.validate_user_id(user_id)
        if user_id not in self.user_settings:
            self.initialize_user(user_id)
        
        current_value = self.user_settings[user_id].get(setting, False)
        self.user_settings[user_id][setting] = not current_value
        self.save_data()

    def update_user_settings(self, user_id: int, new_settings: dict) -> None:
        """Update user settings"""
        self.validate_user_id(user_id)
        if user_id not in self.user_settings:
            self.initialize_user(user_id)
        
        self.user_settings[user_id].update(new_settings)
        self.save_data()

    def cleanup_inactive_users(self, days_threshold: int = 30) -> None:
        """Remove data for inactive users"""
        current_time = datetime.now()
        inactive_users = []
        
        for user_id, stats in self.user_stats.items():
            last_active = datetime.fromisoformat(stats['last_active'])
            if (current_time - last_active).days > days_threshold:
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            self.user_contexts.pop(user_id, None)
            self.user_settings.pop(user_id, None)
            self.user_stats.pop(user_id, None)
        
        if inactive_users:
            self.save_data()

    def get_user_statistics(self, user_id: int) -> dict:
        """Get user statistics"""
        self.validate_user_id(user_id)
        return self.user_stats.get(user_id, {})