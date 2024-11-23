from datetime import datetime
from logging.handlers import RotatingFileHandler
import os
import logging

class TelegramLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logs_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self._setup_file_handler()
        self._setup_console_handler()

    def _setup_file_handler(self):
        log_file = os.path.join(self.logs_dir, f'telegram_bot_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - User ID: %(user_id)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _setup_console_handler(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - User ID: %(user_id)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_message(self, message: str, user_id: int, level: str = 'info'):
        extra = {'user_id': user_id}
        if level == 'error':
            self.logger.error(message, extra=extra)
        else:
            self.logger.info(message, extra=extra)

    def log_command(self, command: str, user_id: int):
        self.log_message(f"Command received: {command}", user_id)

    def log_error(self, error: Exception, user_id: int):
        self.log_message(f"Error occurred: {str(error)}", user_id, level='error')

    def log_api_response(self, status: str, user_id: int):
        self.log_message(f"API Response Status: {status}", user_id)

# Create an instance of TelegramLogger
telegram_logger = TelegramLogger()