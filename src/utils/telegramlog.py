import logging
import os
import datetime
import sys

# Configure output encoding for logging to handle non-Latin characters
if sys.stdout.encoding != "utf-8":
    try:
        # For Windows specifically
        if os.name == "nt":
            import codecs

            # Force UTF-8 encoding for stdout
            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
    except Exception as e:
        print(f"Could not set UTF-8 encoding: {e}")


class TelegramLogger:
    def __init__(self):
        self.logger = logging.getLogger("src.utils.telegramlog")
        self.logs_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
        )

        # Create logs directory if it doesn't exist
        os.makedirs(self.logs_dir, exist_ok=True)

        # Configure file handler for Unicode logs
        file_handler = logging.FileHandler(
            os.path.join(self.logs_dir, "telegram_bot.log"), encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)

        # Configure formatter that includes user_id when available
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(user_id)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add file handler to logger
        self.logger.addHandler(file_handler)

    def log_message(self, message, user_id=None):
        """Log a message with user_id if available."""
        extra = {"user_id": user_id if user_id else "SYSTEM"}

        try:
            # Ensure message is a string
            message_str = str(message)
            self.logger.info(message_str, extra=extra)
        except UnicodeEncodeError:
            # If encoding fails, use repr to get a safe representation
            self.logger.info(
                f"Message contained special characters: {repr(message)}", extra=extra
            )

    def log_error(self, error, user_id=None):
        """Log an error with user_id if available."""
        extra = {"user_id": user_id if user_id else "SYSTEM"}
        self.logger.error(str(error), extra=extra)

    def log_warning(self, warning, user_id=None):
        """Log a warning with user_id if available."""
        extra = {"user_id": user_id if user_id else "SYSTEM"}
        self.logger.warning(str(warning), extra=extra)


# Create a global instance of the logger
telegram_logger = TelegramLogger()
