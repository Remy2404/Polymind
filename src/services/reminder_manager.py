import asyncio
import logging
from datetime import datetime

class ReminderManager:
    def __init__(self, bot):
        self.bot = bot
        self.reminders = {}
        self.logger = logging.getLogger(__name__)
        self.reminder_check_task = None

    async def set_reminder(self, user_id: int, time: datetime, message: str):
        self.reminders[user_id] = {
            'time': time,
            'message': message
        }
        
    async def check_reminders(self):
        now = datetime.now()
        for user_id, reminder in self.reminders.items():
            if reminder['time'] <= now:
                await self.bot.send_message(
                    chat_id=user_id,
                    text=f"⏰ Reminder: {reminder['message']}"
                )
                del self.reminders[user_id]

    async def start_reminder_checks(self):
        """Start periodic reminder checks"""
        while True:
            try:
                await self.check_reminders()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in reminder check: {str(e)}")

    async def start(self):
        """Initialize and start the reminder service"""
        self.reminder_check_task = asyncio.create_task(self.start_reminder_checks())
        self.logger.info("Reminder service started")

    async def stop(self):
        """Stop the reminder service"""
        if self.reminder_check_task:
            self.reminder_check_task.cancel()
            self.logger.info("Reminder service stopped")
