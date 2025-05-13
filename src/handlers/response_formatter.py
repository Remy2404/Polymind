import logging
import html
from typing import List
from telegramify_markdown import convert


class ResponseFormatter:
    """Formats response text for sending via Telegram."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def format_telegram_markdown(self, text: str) -> str:
        try:

            formatted_text = convert(text)
            return formatted_text
        except Exception as e:
            self.logger.error(f"Error formatting markdown: {str(e)}")
            # escape Telegram MarkdownV2 reserved characters if conversion fails
            for ch in [
                "_",
                "*",
                "[",
                "]",
                "(",
                ")",
                "~",
                "`",
                ">",
                "#",
                "+",
                "-",
                "=",
                "|",
                "{",
                "}",
                ".",
                "!",
            ]:
                text = text.replace(ch, f"\\{ch}")
            return text

    async def format_telegram_html(self, text: str) -> str:
        """Formats text for sending via Telegram using HTML parse mode."""
        try:
            # Escape HTML special characters
            escaped_text = html.escape(text)
            return escaped_text
        except Exception as e:
            self.logger.error(f"Error escaping HTML: {e}")
            # Fallback: escape text again
            return html.escape(text)

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
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
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

        if is_reply:
            reply_indicator = "↪️ Replying to message"
            return f"{model_indicator}\n{reply_indicator}\n\n{text}"
        else:
            return f"{model_indicator}\n\n{text}"
