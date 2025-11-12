import logging
from typing import Tuple, Optional
from telegram import Message


class MessageContextHandler:
    """
    Handler for extracting and managing message context information.
    This includes handling quoted texts, referred images, documents, etc.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_reply_context(
        self, message: Message
    ) -> Tuple[Optional[str], Optional[int]]:
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
            self.logger.info(
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

    def format_prompt_with_quote(self, prompt: str, quoted_text: str) -> str:
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

    def detect_reference_to_image(self, text: str) -> bool:
        """
        Detect if a message is referring to a previously shared image.
        Args:
            text: The message text to check
        Returns:
            True if the message appears to refer to an image
        """
        image_related_keywords = [
            "image",
            "picture",
            "photo",
            "pic",
            "img",
            "that image",
            "the picture",
            "this image",
            "in the photo",
            "in this image",
            "on the image",
            "from the image",
            "from this picture",
            "the image above",
            "the photo I sent",
            "the picture I shared",
            "what's in the image",
            "what is in the picture",
            "what does the image show",
            "what did you see in the image",
            "image analysis",
            "analyze the image",
            "tell me about the image",
            "describe the image",
            "explain the image",
            "what's on the image",
            "can you see the",
            "in the background",
            "in the foreground",
            "on the left",
            "on the right",
            "screenshot",
            "snapshot",
            "photograph",
            "imgur",
            "flickr",
            "picasa",
            "tinypic",
            "image hosting",
            "photo sharing",
            "picture gallery",
            "view image",
            "open image",
            "download image",
            "upload image",
            "send image",
            "share image",
            "image link",
            "photo link",
            "picture link",
            "img link",
            "jpeg",
            "jpg",
            "png",
            "gif",
            "bmp",
            "tiff",
            "webp",
            "image file",
            "photo file",
            "picture file",
            "img file",
            "image attachment",
            "photo attachment",
            "picture attachment",
            "img attachment",
        ]
        return any(keyword in text.lower() for keyword in image_related_keywords)

    def detect_reference_to_document(self, text: str) -> bool:
        """
        Detect if a message is referring to a previously shared document.
        Args:
            text: The message text to check
        Returns:
            True if the message appears to refer to a document
        """
        document_related_keywords = [
            "document",
            "doc",
            "file",
            "pdf",
            "that document",
            "the file",
            "the pdf",
            "tell me more",
            "more information",
            "more details",
            "explain further",
            "tell me about it",
            "what else",
            "elaborate",
            "attachment",
            "send file",
            "share file",
            "open document",
            "view document",
            "download document",
            "upload document",
            "file link",
            "document link",
            "doc link",
            "pdf link",
            "file format",
            "document format",
            "doc format",
            "pdf format",
            "readable document",
            "editable document",
            "word document",
            "excel file",
            "powerpoint file",
            "text file",
            "csv file",
            "xml file",
            "json file",
            "zip file",
            "rar file",
            "tar file",
            "gzip file",
            "bzip2 file",
            "7z file",
            "file attachment",
            "document attachment",
            "doc attachment",
            "pdf attachment",
        ]
        return any(keyword in text.lower() for keyword in document_related_keywords)

    def should_use_reply_format(self, quoted_text: str, quoted_message_id: int) -> bool:
        return quoted_text is not None and quoted_message_id is not None

    def format_response_with_quote_indicator(
        self, response: str, model_indicator: str, is_reply: bool = False
    ) -> str:
        if is_reply:
            reply_indicator = "↪️ Replying to message"
            return f"{model_indicator}\n{reply_indicator}\n\n{response}"
        else:
            return f"{model_indicator}\n\n{response}"
