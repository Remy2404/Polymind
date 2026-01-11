import logging
from typing import Tuple, Optional
from telegram import Message

logger = logging.getLogger(__name__)

def extract_reply_context(message: Message) -> Tuple[Optional[str], Optional[int]]:
    """
    Extract quoted text and message ID from a reply message.
    Args:
        message: The telegram Message object
    Returns:
        A tuple containing (quoted_text, quoted_message_id)
        Both can be None if there is no reply
    """
    if not message or not message.reply_to_message:
        return None, None
    
    quoted_message_id = message.reply_to_message.message_id
    quoted_text = None
    
    if message.reply_to_message.text:
        quoted_text = message.reply_to_message.text
        logger.info(
            f"User is replying to text message with length: {len(quoted_text)} characters"
        )
    elif message.reply_to_message.caption:
        quoted_text = (
            f"[Image/Document with caption: {message.reply_to_message.caption}]"
        )
    elif message.reply_to_message.photo:
        quoted_text = "[Image without caption]"
    elif message.reply_to_message.document:
        quoted_text = f"[Document: {message.reply_to_message.document.file_name}]"
    elif message.reply_to_message.voice:
        quoted_text = "[Voice message: Transcribed audio]"
    elif message.reply_to_message.sticker:
        quoted_text = (
            f"[Sticker: {getattr(message.reply_to_message.sticker, 'emoji', '😊')}]"
        )
    elif message.reply_to_message.animation:
        quoted_text = "[GIF/Animation]"
    elif message.reply_to_message.video:
        quoted_text = "[Video]" + (
            f" with caption: {message.reply_to_message.caption}"
            if message.reply_to_message.caption
            else ""
        )
    else:
        quoted_text = "[Message of unsupported type]"
        
    return quoted_text, quoted_message_id

def format_prompt_with_quote(prompt: str, quoted_text: str) -> str:
    """
    Format user prompt by including the quoted message context.
    Args:
        prompt: The original user prompt/message
        quoted_text: The text that was quoted/replied to
    Returns:
        Enhanced prompt with quoted context
    """
    if not quoted_text:
        return prompt
    return f'The user is replying to this message: "{quoted_text}"\n\nUser\'s reply: {prompt}'
