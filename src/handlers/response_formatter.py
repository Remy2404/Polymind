import logging
from typing import List, Optional
from telegram import Message
from telegramify_markdown import convert


class ResponseFormatter:
    """Formats response text for sending via Telegram."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def format_telegram_markdown(self, text: str) -> str:
        """Format text with Telegram's MarkdownV2 parser."""
        try:
            return convert(text)
        except Exception as e:
            self.logger.error(f"Error formatting markdown: {str(e)}")
            return text.replace("*", "").replace("_", "").replace("`", "")

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        """Split long message into chunks respecting Telegram's message size limit."""
        if len(text) <= max_length:
            return [text]

        chunks = []
        current_chunk = ""

        for line in text.split("\n"):
            if len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def format_with_model_indicator(
        self, text: str, model_indicator: str, is_reply: bool = False
    ) -> str:
        """Add model indicator and optional reply indicator to the response."""
        if is_reply:
            reply_indicator = "↪️ Replying to message"
            return f"{model_indicator}\n{reply_indicator}\n\n{text}"
        else:
            return f"{model_indicator}\n\n{text}"
