import logging
import html
from typing import List
from telegramify_markdown import convert, escape_markdown, markdownify, customize

class ResponseFormatter:
    """Formats response text for sending via Telegram."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def format_telegram_markdown(self, text: str) -> str:
        try:
            # Ensure text is properly formatted
            if not isinstance(text, str):
                text = str(text)

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

    async def escape_markdown_text(self, text: str) -> str:
        """
        Escape special Telegram MarkdownV2 characters in text.
        
        This method ensures that special characters in text don't interfere with 
        Telegram MarkdownV2 formatting by escaping them properly.
        
        Args:
            text: The text to escape
            
        Returns:
            Properly escaped text for Telegram MarkdownV2
        """
        try:
            if not isinstance(text, str):
                text = str(text)
                
            escaped_text = escape_markdown(text)
            return escaped_text
        except Exception as e:
            self.logger.error(f"Error escaping markdown: {str(e)}")
            # Fallback to manual escaping
            for ch in [
                "_", "*", "[", "]", "(", ")", "~", "`", ">", 
                "#", "+", "-", "=", "|", "{", "}", ".", "!"
            ]:
                text = text.replace(ch, f"\\{ch}")
            return text

    async def markdownify_text(self, markdown_text: str, normalize_whitespace: bool = False) -> str:
        """
        Convert standard markdown to Telegram MarkdownV2 format.
        
        This provides a more comprehensive conversion of markdown content,
        handling complex formatting including headings, lists, code blocks, etc.
        
        Args:
            markdown_text: Raw markdown text to convert
            normalize_whitespace: Whether to normalize whitespace in the output
            
        Returns:
            Text formatted for Telegram's MarkdownV2 parse mode
        """
        try:
            if not isinstance(markdown_text, str):
                markdown_text = str(markdown_text)
                
            # Use more customization options if needed
            # customize.markdown_symbol.head_level_1 = "📌"  # Customize heading symbols
            # customize.strict_markdown = False  # For handling __underline__ properly
                
            converted_text = markdownify(
                markdown_text,
                normalize_whitespace=normalize_whitespace,
                latex_escape=True  # Set to False if LaTeX processing is not needed
            )
            return converted_text
        except Exception as e:
            self.logger.error(f"Error converting markdown with markdownify: {str(e)}")
            # Fallback to simpler conversion
            return await self.format_telegram_markdown(markdown_text)

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
        # Validate inputs
        if not text or not isinstance(text, str):
            return [""]

        if max_length <= 0:
            max_length = 4096

        if len(text) <= max_length:
            return [text]

        chunks = []
        current_chunk = ""

        for line in text.split("\n"):
            # Handle lines that are too long by themselves
            if len(line) > max_length:
                # Add current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Split the long line into smaller pieces
                words = line.split(" ")
                temp_line = ""
                for word in words:
                    if len(temp_line + " " + word) > max_length:
                        if temp_line:
                            chunks.append(temp_line)
                            temp_line = word
                        else:
                            # Single word is too long, force split
                            chunks.append(word[:max_length])
                            temp_line = (
                                word[max_length:] if len(word) > max_length else ""
                            )
                    else:
                        temp_line += " " + word if temp_line else word

                if temp_line:
                    current_chunk = temp_line
            elif len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
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
