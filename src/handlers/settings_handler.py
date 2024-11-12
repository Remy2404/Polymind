from telegram import Update
from telegram.ext import ContextTypes, CommandHandler
from typing import Dict

class UserSettings:
    def __init__(self):
        self.settings_store: Dict[int, Dict] = {}
        self.default_settings = {
            'language': 'en',
            'temperature': 0.7,
            'notifications': True
        }

    def get_user_settings(self, user_id: int) -> Dict:
        return self.settings_store.get(user_id, self.default_settings.copy())

    def update_user_settings(self, user_id: int, new_settings: Dict):
        current = self.get_user_settings(user_id)
        current.update(new_settings)
        self.settings_store[user_id] = current

class SettingsHandler:
    def __init__(self):
        self.user_settings = UserSettings()

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message:
            return

        user_id = update.effective_user.id
        settings = self.user_settings.get_user_settings(user_id)
        
        settings_text = "Current Settings:\n"
        for key, value in settings.items():
            settings_text += f"{key}: {value}\n"
        
        await update.message.reply_text(settings_text)

    async def update_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message or not context.args:
            return

        user_id = update.effective_user.id
        try:
            setting_name = context.args[0]
            setting_value = context.args[1]
            
            new_settings = {setting_name: setting_value}
            self.user_settings.update_user_settings(user_id, new_settings)
            
            await update.message.reply_text(f"Updated {setting_name} to {setting_value}")
        except Exception as e:
            await update.message.reply_text(f"Error updating settings: {str(e)}")

    def get_handlers(self):
        return [
            CommandHandler('settings', self.settings_command),
            CommandHandler('set', self.update_settings)
        ]
