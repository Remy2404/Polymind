import logging
import html
import re
import subprocess
import tempfile
from typing import List, Optional, Union, Any
from telegramify_markdown import convert, escape_markdown, markdownify, customize
from telegram.error import BadRequest


class ResponseFormatter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # --- NEW FEATURE: Table conversion support ---
    def _convert_tables(self, text: str) -> str:
        pattern = r"(?:\|.*\|\s*\n)(?:\|[-: ]+\|\s*\n)(?:\|.*\|\s*\n?)+"

        def repl(match):
            rows = match.group(0).strip().splitlines()
            return "\n".join(f"```\n{row}\n```" for row in rows)

        return re.sub(pattern, repl, text)

    # --- NEW FEATURE: Spoiler & underline support ---
    def _handle_spoilers_and_underlines(self, text: str) -> str:
        text = re.sub(
            r"\|\|(.+?)\|\|", lambda m: f"||{escape_markdown(m.group(1))}||", text
        )
        text = re.sub(
            r"__(.+?)__", lambda m: f"__{escape_markdown(m.group(1))}__", text
        )
        return text

    # --- NEW FEATURE: Mermaid rendering support ---
    def _render_mermaid_to_image(self, mmd_text: str) -> Any:
        with tempfile.NamedTemporaryFile(suffix=".mmd", delete=False) as src:
            src.write(mmd_text.encode())
            src.flush()
            png = src.name.replace(".mmd", ".png")
        subprocess.run(["mmdc", "-i", src.name, "-o", png], check=True)
        return open(png, "rb")

    async def format_telegram_markdown(self, text: str) -> str:
        text = str(text)
        text = self._clean_unwanted_dashes(text)
        try:
            return convert(text)
        except Exception as e:
            self.logger.error(f"format_telegram_markdown error: {e}")
            return self._escape_all(text)

    async def escape_markdown_text(self, text: str) -> str:
        text = str(text)
        try:
            return escape_markdown(text)
        except Exception as e:
            self.logger.error(f"escape_markdown_text error: {e}")
            return self._escape_all(text)

    # --- OVERRIDDEN: Markdownify with table + spoiler/underline handling ---
    async def markdownify_text(
        self, md: str, normalize_whitespace: bool = False
    ) -> str:
        md = str(md)
        md = self._clean_unwanted_dashes(md)
        md = self._convert_tables(md)
        md = self._handle_spoilers_and_underlines(md)
        try:
            return markdownify(
                md, normalize_whitespace=normalize_whitespace, latex_escape=True
            )
        except Exception as e:
            self.logger.error(f"markdownify_text error: {e}")
            return await self.format_telegram_markdown(md)

    async def format_telegram_html(self, text: str) -> str:
        try:
            return html.escape(str(text))
        except Exception as e:
            self.logger.error(f"format_telegram_html error: {e}")
            return html.escape(str(text))

    def set_markdown_options(self, **opts) -> "ResponseFormatter":
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

    # --- OVERRIDDEN: Markdown-safe splitting + table pre-processing ---
    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        """Split long messages into chunks that fit within Telegram's limits"""
        text = self._convert_tables(str(text or ""))
        
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        
        # Try to split by paragraphs first (double newlines)
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If a single paragraph is too long, we need to split it further
            if len(paragraph) > max_length:
                # If we have content in current_chunk, save it first
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split the long paragraph by sentences or lines
                lines = paragraph.split('\n')
                for line in lines:
                    if len(line) > max_length:
                        # If a single line is too long, split by characters
                        while len(line) > max_length:
                            split_point = max_length
                            # Try to find a good split point (space, comma, etc.)
                            for i in range(max_length - 100, max_length):
                                if i < len(line) and line[i] in ' ,.;:':
                                    split_point = i + 1
                                    break
                            
                            chunks.append(line[:split_point].strip())
                            line = line[split_point:]
                        
                        if line.strip():
                            current_chunk = line.strip()
                    else:
                        # Check if adding this line would exceed the limit
                        if len(current_chunk) + len(line) + 1 > max_length:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                            current_chunk = line
                        else:
                            if current_chunk:
                                current_chunk += "\n" + line
                            else:
                                current_chunk = line
            else:
                # Check if adding this paragraph would exceed the limit
                if len(current_chunk) + len(paragraph) + 2 > max_length:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
        
        # Add the last chunk if there's any content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Ensure no chunk is empty and all are within limits
        final_chunks = []
        for chunk in chunks:
            if chunk.strip() and len(chunk) <= max_length:
                final_chunks.append(chunk)
            elif chunk.strip() and len(chunk) > max_length:
                # Emergency split if we still have oversized chunks
                while len(chunk) > max_length:
                    final_chunks.append(chunk[:max_length-3] + "...")
                    chunk = "..." + chunk[max_length-3:]
                if chunk.strip():
                    final_chunks.append(chunk)
        
        return final_chunks if final_chunks else [text[:max_length]]

    def format_with_model_indicator(
        self, text: str, model: str, is_reply: bool = False
    ) -> str:
        text = str(text)
        header = model + ("\n↪️ Replying to message\n" if is_reply else "\n")
        return f"{header}{text}"

    def _escape_all(self, text: str) -> str:
        specials = r"_*[]()~`>#+\-=|{}.!".split()
        for ch in specials:
            text = text.replace(ch, f"\\{ch}")
        return text

    def _clean_unwanted_dashes(self, text: str) -> str:
        """Remove standalone dashes that appear at the end of lines"""
        # Remove lines that contain only dashes, spaces, or are empty
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just dashes, asterisks, or empty
            if stripped and not re.match(r'^[-*\s]+$', stripped):
                cleaned_lines.append(line)
            elif not stripped:  # Keep empty lines for formatting
                cleaned_lines.append(line)
                
        return '\n'.join(cleaned_lines)

    # --- OVERRIDDEN: Detect mermaid blocks and render as image ---
    async def safe_send_message(
        self,
        message,
        text: str,
        reply_to_message_id: Optional[int] = None,
        disable_web_page_preview: bool = False,
    ) -> Optional[Any]:
        if text and text.strip().startswith("```mermaid"):
            mmd = re.sub(r"^```mermaid\s*|```$", "", text.strip(), flags=re.DOTALL)
            try:
                img = self._render_mermaid_to_image(mmd)
                return await message.reply_photo(
                    photo=img,
                    caption="Mermaid diagram",
                    reply_to_message_id=reply_to_message_id,
                    disable_web_page_preview=disable_web_page_preview,
                )
            except Exception as e:
                self.logger.error(f"Mermaid rendering failed: {e}")
                # fallback below

        # Check if message is too long and split it
        if len(text) > 4000:  # Leave some margin below Telegram's 4096 limit
            chunks = await self.split_long_message(text, max_length=4000)
            results = []
            for i, chunk in enumerate(chunks):
                try:
                    # Only use reply_to_message_id for the first chunk
                    reply_id = reply_to_message_id if i == 0 else None
                    result = await self._send_single_message(
                        message, chunk, reply_id, disable_web_page_preview
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to send chunk {i+1}: {e}")
            return results[0] if results else None

        # For normal-length messages, try different formatting approaches
        return await self._send_single_message(
            message, text, reply_to_message_id, disable_web_page_preview
        )

    async def _send_single_message(
        self,
        message,
        text: str,
        reply_to_message_id: Optional[int] = None,
        disable_web_page_preview: bool = False,
    ) -> Optional[Any]:
        """Send a single message with multiple formatting fallbacks"""
        
        # Try MarkdownV2 first
        try:
            formatted_text = await self.format_telegram_markdown(text)
            return await message.reply_text(
                text=formatted_text,
                parse_mode="MarkdownV2",
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception:
            pass

        # Try HTML
        try:
            html_text = await self.format_telegram_html(text)
            return await message.reply_text(
                text=html_text,
                parse_mode="HTML",
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception:
            pass

        # Try escaped MarkdownV2
        try:
            escaped = await self.escape_markdown_text(text)
            return await message.reply_text(
                text=escaped,
                parse_mode="MarkdownV2",
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception:
            pass

        # Final fallback - plain text
        try:
            return await message.reply_text(
                text=text,
                reply_to_message_id=reply_to_message_id,
                disable_web_page_preview=disable_web_page_preview,
            )
        except Exception as e:
            self.logger.error(f"All send attempts failed: {e}")
            return None

    async def format_response(
        self, content: str, user_id: int = None, model_name: str = None
    ) -> str:
        try:
            content = str(content) if content else ""
            if model_name:
                model_badges = {
                    "gemini-2.0-flash": "🤖 *Gemini 2\\.0 Flash*",
                    "gemini": "🤖 *Gemini*",
                    "deepseek": "🧠 *DeepSeek*",
                    "openrouter": "🔀 *OpenRouter*",
                    "gpt": "🔥 *GPT*",
                    "claude": "🌟 *Claude*",
                    "moonshot": "🌙 *Moonshot Kimi*",
                    "kimi": "🌙 *Kimi*",
                }
                badge = next(
                    (b for k, b in model_badges.items() if k in model_name.lower()),
                    None,
                )
                if not badge:
                    disp = model_name.replace("-", "\\-").replace(".", "\\.")
                    badge = f"🤖 *{disp}*"
                content = f"{badge}\n\n{content}"
            return await self.format_telegram_markdown(content)
        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            return await self.escape_markdown_text(str(content or ""))
