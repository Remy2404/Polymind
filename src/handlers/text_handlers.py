from telegram import Update
from telegram.ext import ContextTypes, MessageHandler, filters

class TextHandler:
    def __init__(self, gemini_api):
        self.gemini_api = gemini_api

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        response = await self.gemini_api.get_text_response(update.message.text)
        await update.message.reply_text(response)

    def get_handlers(self):
        return [
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text)
        ]
