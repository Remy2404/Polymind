import logging
from telegram import Update
from telegram.ext import ContextTypes

async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logging.error(f"Update {update} caused error {context.error}")
    if update.message:
        await update.message.reply_text("An error occurred while processing your request.")
