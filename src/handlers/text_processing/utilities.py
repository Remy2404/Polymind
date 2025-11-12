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
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
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

        # Ensure we have a valid extension
        if not file_extension:
            logger.warning("Empty file extension provided to get_mime_type")
            return "application/octet-stream"

        result = mime_types.get(file_extension.lower(), "application/octet-stream")
        logger.debug(f"MIME type for extension '{file_extension}': {result}")
        return result

    @staticmethod
    def detect_mime_from_content(file_data: bytes) -> str:
        """
        Detect MIME type from file content (magic bytes)
        Args:
            file_data: First few bytes of the file
        Returns:
            MIME type string
        """
        if not file_data or len(file_data) < 4:
            return "application/octet-stream"

        # Check magic bytes for common image formats
        if file_data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif file_data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif file_data.startswith(b"GIF87a") or file_data.startswith(b"GIF89a"):
            return "image/gif"
        elif file_data.startswith(b"RIFF") and b"WEBP" in file_data[:12]:
            return "image/webp"
        elif file_data.startswith(b"BM"):
            return "image/bmp"
        elif file_data.startswith(b"%PDF"):
            return "application/pdf"

        return "application/octet-stream"

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
        cleaned = " ".join(message_text.split())
        cleaned = cleaned.replace("\0", "")
        return cleaned

    @staticmethod
    def extract_command_args(message_text: str, command: str) -> str:
        """Extract arguments after a command"""
        if message_text.startswith(command):
            args = message_text[len(command) :].strip()
            return args
        return ""
