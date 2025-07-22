import logging
from telegram import Update
from telegram.ext import ContextTypes

async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error("Exception while handling an update:", exc_info=context.error)
    if update and getattr(update, "message", None):
        try:
            await update.message.reply_text("Sorry, something went wrong. Please try again later.")
        except Exception as e:
            logging.error("Failed to send error message to user.", exc_info=e)
