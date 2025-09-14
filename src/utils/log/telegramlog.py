import logging
import os
import sys

if sys.stdout.encoding != "utf-8":
    try:
        if os.name == "nt":
            import codecs

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
        os.makedirs(self.logs_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(self.logs_dir, "telegram_bot.log"), encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(user_id)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def log_message(self, message, user_id=None):
        """Log a message with user_id if available."""
        extra = {"user_id": user_id if user_id else "SYSTEM"}
        try:
            message_str = str(message)
            self.logger.info(message_str, extra=extra)
        except UnicodeEncodeError:
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


telegram_logger = TelegramLogger()
