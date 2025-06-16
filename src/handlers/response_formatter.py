import logging
import html
from typing import List, Optional, Union, Any
from telegramify_markdown import convert, escape_markdown, markdownify, customize
from telegram.error import BadRequest


class ResponseFormatter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def format_telegram_markdown(self, text: str) -> str:
        """Convert raw Markdown to Telegram MarkdownV2."""
        text = str(text)
        try:
            return convert(text)
        except Exception as e:
            self.logger.error(f"format_telegram_markdown error: {e}")
            return self._escape_all(text)

    async def escape_markdown_text(self, text: str) -> str:
        """Escape MarkdownV2 special characters."""
        text = str(text)
        try:
            return escape_markdown(text)
        except Exception as e:
            self.logger.error(f"escape_markdown_text error: {e}")
            return self._escape_all(text)

    async def markdownify_text(
        self, md: str, normalize_whitespace: bool = False
    ) -> str:
        """Convert Markdown to Telegram format, with optional whitespace normalization."""
        md = str(md)
        try:
            return markdownify(
                md, normalize_whitespace=normalize_whitespace, latex_escape=True
            )
        except Exception as e:
            self.logger.error(f"markdownify_text error: {e}")
            return await self.format_telegram_markdown(md)

    async def format_telegram_html(self, text: str) -> str:
        """Escape text for Telegram HTML mode."""
        try:
            return html.escape(str(text))
        except Exception as e:
            self.logger.error(f"format_telegram_html error: {e}")
            return html.escape(str(text))

    def set_markdown_options(self, **opts) -> "ResponseFormatter":
        """Customize telegramify-markdown behavior."""
        cfg = customize
        if "strict_markdown" in opts:
            cfg.strict_markdown = bool(opts["strict_markdown"])
        if "cite_expandable" in opts:
            cfg.cite_expandable = bool(opts["cite_expandable"])
        symbols = opts.get("markdown_symbols", {})
        if isinstance(symbols, dict):
            for lvl in range(1, 7):
                if f"head_level_{lvl}" in symbols:
                    setattr(
                        cfg.markdown_symbol,
                        f"head_level_{lvl}",
                        symbols[f"head_level_{lvl}"],
                    )
            for name in ("link", "image", "item", "task_list"):
                if name in symbols:
                    setattr(cfg.markdown_symbol, name, symbols[name])
        return self

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        """Break long text into Telegram-friendly chunks."""
        text = str(text or "")
        if max_length <= 0:
            max_length = 4096
        if len(text) <= max_length:
            return [text]

        chunks, current = [], ""
        for line in text.splitlines():
            if len(line) > max_length:
                if current:
                    chunks.append(current)
                    current = ""
                for word in line.split():
                    if len((temp := (current + " " + word).strip())) > max_length:
                        chunks.append(current)
                        current = word
                    else:
                        current = temp
                continue
            if len(current) + len(line) + 1 > max_length:
                chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)
        return chunks

    def format_with_model_indicator(
        self, text: str, model: str, is_reply: bool = False
    ) -> str:
        """Add model name header (and optional 'replying to')."""
        text = str(text)
        header = model + ("\n↪️ Replying to message\n" if is_reply else "\n")
        return f"{header}{text}"

    def _escape_all(self, text: str) -> str:
        """Helper: manually escape all MarkdownV2 specials."""
        specials = r"_*[]()~`>#+\-=|{}.!".split()
        for ch in specials:
            text = text.replace(ch, f"\\{ch}")
        return text

    async def safe_send_message(
        self, 
        message, 
        text: str, 
        reply_to_message_id: Optional[int] = None
    ) -> Optional[Any]:
        """
        Safely send a message with multiple formatting fallbacks.
        
        Tries multiple approaches in order:
        1. MarkdownV2 formatting
        2. HTML formatting
        3. Escaped text
        4. Plain text
        
        Args:
            message: The original message to reply to (Message or MockMessage)
            text: The text content to send
            reply_to_message_id: Optional message ID to reply to
            
        Returns:
            The sent Message object, or None if all attempts failed
        """
        if not text or not text.strip():
            self.logger.warning("Attempted to send empty message")
            return None
            
        chat_id = message.chat_id
        # Handle both real Message objects and MockMessage objects
        bot = getattr(message, 'bot', None) or message.get_bot()
        
        # Try MarkdownV2 first
        try:
            formatted_text = await self.format_telegram_markdown(text)
            return await bot.send_message(
                chat_id=chat_id,
                text=formatted_text,
                parse_mode='MarkdownV2',
                reply_to_message_id=reply_to_message_id
            )
        except Exception as e:
            self.logger.debug(f"MarkdownV2 formatting failed: {e}")
        
        # Try HTML formatting
        try:
            html_text = await self.format_telegram_html(text)
            return await bot.send_message(
                chat_id=chat_id,
                text=html_text,
                parse_mode='HTML',
                reply_to_message_id=reply_to_message_id
            )
        except Exception as e:
            self.logger.debug(f"HTML formatting failed: {e}")
        
        # Try escaped markdown
        try:
            escaped_text = await self.escape_markdown_text(text)
            return await bot.send_message(
                chat_id=chat_id,
                text=escaped_text,
                parse_mode='MarkdownV2',
                reply_to_message_id=reply_to_message_id
            )
        except Exception as e:
            self.logger.debug(f"Escaped markdown failed: {e}")
        
        # Final fallback: plain text
        try:
            return await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id
            )
        except Exception as e:
            self.logger.error(f"All message sending attempts failed: {e}")
            return None
