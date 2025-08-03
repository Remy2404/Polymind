"""
Utility functions for text and media processing.
"""

import logging

logger = logging.getLogger(__name__)


class MediaUtilities:
    """Utility functions for media handling"""

    @staticmethod
    def get_mime_type(file_extension: str) -> str:
        """
        Get MIME type from file extension

        Args:
            file_extension: File extension string including the dot (e.g. ".jpg")

        Returns:
            MIME type string
        """
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".pdf": "application/pdf",
            ".txt": "text/plain",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".json": "application/json",
            ".zip": "application/zip",
        }

        return mime_types.get(file_extension.lower(), "application/octet-stream")

    @staticmethod
    def is_image_file(file_extension: str) -> bool:
        """Check if file extension is for an image"""
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"]
        return file_extension.lower() in image_extensions

    @staticmethod
    def is_video_file(file_extension: str) -> bool:
        """Check if file extension is for a video"""
        video_extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"]
        return file_extension.lower() in video_extensions

    @staticmethod
    def is_audio_file(file_extension: str) -> bool:
        """Check if file extension is for an audio file"""
        audio_extensions = [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"]
        return file_extension.lower() in audio_extensions

    @staticmethod
    def is_document_file(file_extension: str) -> bool:
        """Check if file extension is for a document"""
        document_extensions = [
            ".pdf",
            ".doc",
            ".docx",
            ".txt",
            ".rtf",
            ".ppt",
            ".pptx",
            ".xls",
            ".xlsx",
        ]
        return file_extension.lower() in document_extensions


class MessagePreprocessor:
    """Process messages before sending to the AI"""

    @staticmethod
    def clean_message(message_text: str) -> str:
        """Clean message text from special characters or problematic patterns"""
        # Remove redundant whitespace
        cleaned = " ".join(message_text.split())
        # Remove null bytes and other special characters if needed
        cleaned = cleaned.replace("\0", "")
        return cleaned

    @staticmethod
    def extract_command_args(message_text: str, command: str) -> str:
        """Extract arguments after a command"""
        if message_text.startswith(command):
            args = message_text[len(command) :].strip()
            return args
        return ""
