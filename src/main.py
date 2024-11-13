import logging
import asyncio
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from handlers import command_handlers, text_handlers, code_handlers, error_handler
from config import TELEGRAM_TOKEN, LOG_FORMAT, LOG_LEVEL
from services.gemini_api import GeminiAPI
from services.user_data_manager import UserDataManager
from telegram import Update
from utils.telegramlog import TelegramLogger
logging.basicConfig(
    format=LOG_FORMAT,
    level=LOG_LEVEL
)

logger = logging.getLogger(__name__)

class GeminiBot:
    def __init__(self):
        self.application = Application.builder().token(TELEGRAM_TOKEN).build()
        self.gemini_api = GeminiAPI()
        self.user_data_manager = UserDataManager()
        self._register_handlers()
        logger.info("Bot initialized successfully")
    #handle text messages
    async def _handle_text_message(self, update: Update, context):
        user_id = update.effective_user.id
        TelegramLogger.log_message(user_id , "Received text message: {update.message.text}")
        await text_handlers.handle_text_message(update, context, self.gemini_api, self.user_data_manager)
    def _register_handlers(self):
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("reset", self._reset_command))

        # Message handlers
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            self._handle_text_message
        ))
        self.application.add_handler(MessageHandler(
            filters.PHOTO, 
            self._handle_image_message
        ))
        self.application.add_handler(MessageHandler(
            filters.Regex(r'/code'), 
            self._handle_code_message
        ))

        # Error handler
        self.application.add_error_handler(self._error_handler)

    async def _start_command(self, update, context):
        logger.info(f"Start command received from user {update.effective_user.id}")
        await command_handlers.start(update, context, self.user_data_manager)

    async def _help_command(self, update, context):
        await command_handlers.help(update, context)

    async def _reset_command(self, update, context):
        await command_handlers.reset(update, context, self.user_data_manager)

    async def _handle_text_message(self, update, context):
        await text_handlers.handle_text_message(update, context, self.gemini_api, self.user_data_manager)

    async def _handle_image_message(self, update, context):
        await text_handlers.handle_image_message(update, context, self.gemini_api)

    async def _handle_code_message(self, update, context):
        await code_handlers.handle_code(update, context, self.gemini_api)

    async def _error_handler(self, update, context):
        await error_handler.handle_error(update, context)

    async def run(self):
        logger.info("Starting bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES)   

async def main():
    bot = GeminiBot()
    try:
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == '__main__':
    asyncio.run(main())

