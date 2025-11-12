"""
Document Sender for Telegram Bot
Handles sending Word documents and other files created by MCP tools to Telegram users.
"""

import logging
import os
from typing import Optional
from pathlib import Path


class DocumentSender:
    """Handles sending documents to Telegram users"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supported_extensions = [
            ".docx",
            ".pdf",
            ".txt",
            ".xlsx",
            ".pptx",
            ".doc",
            ".xls",
            ".ppt",
        ]

    async def send_document(
        self,
        bot,
        chat_id: int,
        file_path: str,
        caption: Optional[str] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> bool:
        """
        Send a document file to a Telegram user.

        Args:
            bot: Telegram bot instance
            chat_id: Telegram chat ID
            file_path: Path to the document file
            caption: Optional caption for the document
            reply_to_message_id: Optional message ID to reply to

        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate file exists
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return False

            # Check file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_extensions:
                self.logger.warning(f"Unsupported file extension: {file_ext}")

            # Get file size
            file_size = os.path.getsize(file_path)
            max_size = 50 * 1024 * 1024  # 50MB Telegram limit

            if file_size > max_size:
                self.logger.error(f"File too large: {file_size} bytes (max {max_size})")
                return False

            self.logger.info(
                f"Sending document: {file_path} ({file_size} bytes) to chat {chat_id}"
            )

            # Send the document
            with open(file_path, "rb") as doc_file:
                kwargs = {
                    "chat_id": chat_id,
                    "document": doc_file,
                }

                if caption:
                    kwargs["caption"] = caption

                if reply_to_message_id:
                    kwargs["reply_to_message_id"] = reply_to_message_id

                await bot.send_document(**kwargs)

            self.logger.info(
                f"Successfully sent document {file_path} to chat {chat_id}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error sending document: {str(e)}")
            return False

    async def send_document_from_mcp_result(
        self,
        bot,
        chat_id: int,
        mcp_result: dict,
        reply_to_message_id: Optional[int] = None,
    ) -> bool:
        """
        Send a document created by MCP office-word tools.

        Args:
            bot: Telegram bot instance
            chat_id: Telegram chat ID
            mcp_result: Result from MCP tool execution
            reply_to_message_id: Optional message ID to reply to

        Returns:
            True if document was sent, False otherwise
        """
        try:
            # Extract file path from MCP result
            # Office-word MCP tools typically return file paths in their results
            content = mcp_result.get("content", [])

            if not content:
                return False

            # Look for file paths in the result
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "")
                elif isinstance(item, str):
                    text = item
                else:
                    continue

                # Check if this looks like a file path
                if ".docx" in text or ".pdf" in text:
                    # Extract the file path
                    # Common patterns: "Document created at: /path/to/file.docx"
                    # or just "/path/to/file.docx"
                    lines = text.split("\n")
                    for line in lines:
                        if any(ext in line for ext in self.supported_extensions):
                            # Try to extract path
                            potential_path = line.strip()

                            # Remove common prefixes
                            for prefix in [
                                "Document created at:",
                                "File:",
                                "Path:",
                                "Saved to:",
                            ]:
                                if potential_path.startswith(prefix):
                                    potential_path = potential_path[
                                        len(prefix) :
                                    ].strip()

                            # Check if file exists
                            if os.path.exists(potential_path):
                                caption = "📄 Document created successfully!"
                                return await self.send_document(
                                    bot,
                                    chat_id,
                                    potential_path,
                                    caption,
                                    reply_to_message_id,
                                )

            return False

        except Exception as e:
            self.logger.error(f"Error sending MCP document: {str(e)}")
            return False

    def find_recent_documents(
        self, directory: str = ".", max_age_seconds: int = 300
    ) -> list:
        """
        Find recently created documents in a directory.

        Args:
            directory: Directory to search (default: current directory)
            max_age_seconds: Maximum age of files in seconds (default: 5 minutes)

        Returns:
            List of file paths for recent documents
        """
        import time

        recent_docs = []
        current_time = time.time()

        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = Path(file_path).suffix.lower()

                    if file_ext in self.supported_extensions:
                        # Check file age
                        file_mtime = os.path.getmtime(file_path)
                        age = current_time - file_mtime

                        if age <= max_age_seconds:
                            recent_docs.append(file_path)

        except Exception as e:
            self.logger.error(f"Error finding recent documents: {str(e)}")

        return sorted(recent_docs, key=os.path.getmtime, reverse=True)
