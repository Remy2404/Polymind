import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

class TelegramLogger:
    def __init__(self):
        # Create logs directory if it doesn't exist
        self.logs_dir = 'logs'
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        # Configure logger
        self.logger = logging.getLogger('TelegramBot')
        self.logger.setLevel(logging.INFO)

        # Create handlers
        self._setup_file_handler()
        self._setup_console_handler()

    def _setup_file_handler(self):
        # Create rotating file handler
        log_file = os.path.join(self.logs_dir, f'telegram_bot_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - User ID: %(user_id)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _setup_console_handler(self):
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_message(self, user_id: int, message: str, level: str = 'info'):
        extra = {'user_id': user_id}
        if level.lower() == 'error':
            self.logger.error(message, extra=extra)
        elif level.lower() == 'warning':
            self.logger.warning(message, extra=extra)
        else:
            self.logger.info(message, extra=extra)

    def log_command(self, user_id: int, command: str):
        self.log_message(user_id, f"Command received: {command}")

    def log_error(self, user_id: int, error: Exception):
        self.log_message(user_id, f"Error occurred: {str(error)}", level='error')

    def log_api_response(self, user_id: int, status: str):
        self.log_message(user_id, f"API Response Status: {status}")

# Usage example:
telegram_logger = TelegramLogger()
