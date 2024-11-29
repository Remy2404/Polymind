# utils/telegramlog.py

import logging

# Use a child logger
telegram_logger = logging.getLogger('telegram_bot')

# No need to add handlers here if already configured in main.py
# Ensure the logger's level is set appropriately
telegram_logger.setLevel(logging.INFO)