import logging
import html
from typing import List
from telegramify_markdown import convert


class ResponseFormatter:
    """Formats response text for sending via Telegram."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def format_telegram_markdown(self, text: str) -> str:
        """Format text for Telegram MarkdownV2 with improved error handling."""
        try:
            # Ensure text is properly formatted
            if not isinstance(text, str):
                text = str(text)

            # Clean the text first
            text = self.safe_format_text(text)
            
            # Pre-process for better academic formatting
            text = self._improve_academic_formatting(text)
            
            # First, try the telegramify_markdown converter
            formatted_text = convert(text)
            return formatted_text
        except Exception as e:
            self.logger.warning(f"Telegramify markdown conversion failed: {str(e)}")
            
            # Fallback: Manual escaping for MarkdownV2
            try:
                escaped_text = self._escape_markdownv2(text)
                return escaped_text
            except Exception as escape_error:
                self.logger.error(f"Manual markdown escaping failed: {str(escape_error)}")
                # Final fallback: return safe plain text
                return self.safe_format_text(text)

    def _escape_markdownv2(self, text: str) -> str:
        """Manually escape MarkdownV2 special characters."""
        # First escape backslashes to avoid double escaping
        text = text.replace('\\', '\\\\')
        
        # MarkdownV2 special characters that need escaping
        special_chars = [
            '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', 
            '+', '-', '=', '|', '{', '}', '.', '!'
        ]
        
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        
        return text

    async def format_telegram_html(self, text: str) -> str:
        """Formats text for sending via Telegram using HTML parse mode."""
        try:
            # Apply academic formatting first
            text = self._improve_academic_formatting(text)
            
            # Convert markdown-style formatting to HTML
            import re
            
            # Convert bullet points to HTML lists
            text = re.sub(r'\n• \*\*(.*?)\*\*:', r'\n• <b>\1</b>:', text)
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
            
            # Escape HTML special characters (but preserve our formatting)
            text = text.replace('&', '&amp;')
            text = text.replace('<', '&lt;').replace('>', '&gt;')
            text = text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
            text = text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')
            
            return text
        except Exception as e:
            self.logger.error(f"Error formatting HTML: {e}")
            # Fallback: basic HTML escape
            return html.escape(text)

    async def split_long_message(self, text: str, max_length: int = 4096) -> List[str]:
        """Split long messages with improved handling of formatting and structure."""
        # Validate inputs
        if not text or not isinstance(text, str):
            return [""]

        if max_length <= 0:
            max_length = 4096

        if len(text) <= max_length:
            return [text]

        chunks = []
        current_chunk = ""

        # Split by sections first (double newlines) to preserve structure
        sections = text.split("\n\n")
        
        for section in sections:
            # If adding this section would exceed the limit
            if len(current_chunk) + len(section) + 2 > max_length:
                # Add current chunk if it exists
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If the section itself is too long, split it carefully
                if len(section) > max_length:
                    # Split by sentences first, then by lines if needed
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', section)
                    
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) + 1 <= max_length:
                            temp_chunk += (" " if temp_chunk else "") + sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = sentence
                            else:
                                # Single sentence too long - split by lines
                                lines = sentence.split('\n')
                                for line in lines:
                                    if len(line) <= max_length:
                                        if len(current_chunk) + len(line) + 1 <= max_length:
                                            current_chunk += ("\n" if current_chunk else "") + line
                                        else:
                                            if current_chunk:
                                                chunks.append(current_chunk.strip())
                                            current_chunk = line
                                    else:
                                        # Line too long - split by words
                                        words = line.split(' ')
                                        temp_line = ""
                                        for word in words:
                                            test_line = f"{temp_line} {word}".strip()
                                            if len(test_line) <= max_length:
                                                temp_line = test_line
                                            else:
                                                if temp_line:
                                                    if len(current_chunk) + len(temp_line) + 1 <= max_length:
                                                        current_chunk += ("\n" if current_chunk else "") + temp_line
                                                    else:
                                                        if current_chunk:
                                                            chunks.append(current_chunk.strip())
                                                        current_chunk = temp_line
                                                    temp_line = word
                                                else:
                                                    # Single word too long
                                                    if len(word) > max_length:
                                                        chunks.append(word[:max_length])
                                                        temp_line = word[max_length:]
                                                    else:
                                                        temp_line = word
                                        
                                        if temp_line:
                                            if len(current_chunk) + len(temp_line) + 1 <= max_length:
                                                current_chunk += ("\n" if current_chunk else "") + temp_line
                                            else:
                                                if current_chunk:
                                                    chunks.append(current_chunk.strip())
                                                current_chunk = temp_line
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = section
            else:
                # Add section to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Ensure no empty chunks and clean up
        cleaned_chunks = []
        for chunk in chunks:
            cleaned_chunk = chunk.strip()
            if cleaned_chunk:
                cleaned_chunks.append(cleaned_chunk)
        
        return cleaned_chunks if cleaned_chunks else [text[:max_length]]

    def safe_format_text(self, text: str) -> str:
        """Safely format text by removing potentially problematic characters and improving readability."""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove or replace problematic characters that often cause formatting issues
        replacements = {
            '\u200b': '',  # Zero-width space
            '\u200c': '',  # Zero-width non-joiner
            '\u200d': '',  # Zero-width joiner
            '\u2060': '',  # Word joiner
            '\ufeff': '',  # Byte order mark
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Improve academic content structure
        import re
        
        # Fix common academic formatting issues
        # Add line breaks after periods followed by numbers (likely enumerated points)
        text = re.sub(r'(\w\.)\s+(\d+\.)', r'\1\n\n\2', text)
        
        # Add line breaks before key academic terms that start new sections
        academic_headers = [
            'Early detection', 'Existing approaches', 'Datasets used', 
            'Considerations:', 'In summary', 'Further research'
        ]
        
        for header in academic_headers:
            text = re.sub(f'({header})', r'\n\n\1', text, flags=re.IGNORECASE)
        
        # Clean up excessive whitespace while preserving intentional structure
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on lines
        text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces on lines
        
        # Don't collapse newlines completely - preserve paragraph structure
        # Only collapse 3+ newlines to double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    async def safe_send_message(self, bot, chat_id: int, text: str, **kwargs) -> None:
        """Safely send a message with MarkdownV2, falling back to plain text if needed."""
        try:
            # First try with MarkdownV2 formatting
            formatted_text = await self.format_telegram_markdown(text)
            await bot.send_message(
                chat_id=chat_id,
                text=formatted_text,
                parse_mode="MarkdownV2",
                **kwargs
            )
        except Exception as e:
            self.logger.warning(f"MarkdownV2 send failed, falling back to plain text: {str(e)}")
            try:
                # Fallback to HTML formatting
                html_text = await self.format_telegram_html(text)
                await bot.send_message(
                    chat_id=chat_id,
                    text=html_text,
                    parse_mode="HTML",
                    **kwargs
                )
            except Exception as html_error:
                self.logger.warning(f"HTML send failed, using plain text: {str(html_error)}")
                # Final fallback to plain text
                safe_text = self.safe_format_text(text)
                await bot.send_message(
                    chat_id=chat_id,
                    text=safe_text,
                    **kwargs
                )

    def format_with_model_indicator(
        self, text: str, model_indicator: str, is_reply: bool = False
    ) -> str:
        """Add model indicator and optional reply indicator to the response."""
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        if not isinstance(model_indicator, str):
            model_indicator = str(model_indicator)
            
        # Apply safe formatting to the text first
        text = self.safe_format_text(text)
        
        if is_reply:
            reply_indicator = "↪️ Replying to message"
            return f"{model_indicator}\n{reply_indicator}\n\n{text}"
        else:
            return f"{model_indicator}\n\n{text}"
    
    def _improve_academic_formatting(self, text: str) -> str:
        """Improve formatting for academic and structured content."""
        import re
        
        # First, normalize the text and add strategic line breaks
        # Add line breaks before numbered points and key sections
        text = re.sub(r'(\w\.)\s+(\d+\.)', r'\1\n\n\2', text)
        text = re.sub(r'(\w)\s+(Early detection|Existing approaches|Datasets used|Considerations)', r'\1\n\n\2', text, flags=re.IGNORECASE)
        
        # Split into paragraphs for processing
        paragraphs = text.split('\n\n')
        formatted_paragraphs = []
        
        for paragraph in paragraphs:
            # Skip empty paragraphs
            if not paragraph.strip():
                continue
            
            # Handle numbered lists and academic content structure
            if any(indicator in paragraph for indicator in ['1.', '2.', '3.', '4.']):
                # This looks like an enumerated list - improve structure
                # Split by numbered points
                parts = re.split(r'(\d+\.)', paragraph)
                formatted_parts = []
                
                current_item = ""
                for i, part in enumerate(parts):
                    if re.match(r'^\d+\.$', part):
                        # This is a number
                        if current_item.strip():
                            formatted_parts.append(current_item.strip())
                        current_item = f"**{part}**"
                    else:
                        current_item += part
                
                if current_item.strip():
                    formatted_parts.append(current_item.strip())
                
                # Format each numbered item nicely
                formatted_items = []
                for item in formatted_parts:
                    if item.startswith('**') and '**' in item[2:]:
                        # This is a numbered item - fix the formatting
                        item = item.replace('**', '', 1).replace(':**', ':', 1)
                        # Split number and content
                        if ':' in item:
                            number_part, content_part = item.split(':', 1)
                            formatted_items.append(f"\n• **{number_part.strip()}:** {content_part.strip()}")
                        else:
                            formatted_items.append(f"\n• **{item}**")
                    elif item.strip():
                        formatted_items.append(item)
                
                paragraph = '\n'.join(formatted_items)
            
            # Format key sections
            elif any(keyword in paragraph.lower() for keyword in [
                'criteria for detection', 'existing approaches', 'detection methods',
                'early detection', 'datasets used', 'considerations'
            ]):
                # This is a section header - make it stand out
                paragraph = f"**{paragraph.strip()}**"
            
            # Handle sentences that contain colons (definitions/explanations)
            elif ':' in paragraph and len(paragraph.split(':')) == 2:
                parts = paragraph.split(':', 1)
                if len(parts[0]) < 100:  # Likely a key-value pair
                    paragraph = f"**{parts[0].strip()}:** {parts[1].strip()}"
            
            # Add better spacing for long paragraphs
            elif len(paragraph) > 400:
                # Split long sentences for better readability
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                if len(sentences) > 2:
                    # Group sentences into smaller chunks
                    chunks = []
                    current_chunk = []
                    
                    for sentence in sentences:
                        current_chunk.append(sentence)
                        if len(' '.join(current_chunk)) > 250:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                    
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    
                    paragraph = '\n\n'.join(chunks)
            
            formatted_paragraphs.append(paragraph)
        
        # Join paragraphs with proper spacing
        result = '\n\n'.join(formatted_paragraphs)
        
        # Clean up excessive newlines but preserve structure
        result = re.sub(r'\n{4,}', '\n\n\n', result)
        
        # Ensure proper spacing around bullet points
        result = re.sub(r'\n•', '\n\n•', result)
        
        return result
