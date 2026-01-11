import io
import os
import asyncio
from typing import Tuple, List, Optional, Any
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from src.handlers.text_processing.utilities import MediaUtilities

class MediaHelpers:
    """Helper class for media extraction and chat actions."""

    @staticmethod
    async def extract_media_files(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Tuple[bool, List[dict], Optional[str]]:
        """Extract media files from the update."""
        has_attached_media = False
        media_files = []
        media_type = None

        if not update.message:
             return False, [], None

        if update.message.photo:
            has_attached_media = True
            media_type = "photo"
            photo = update.message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            photo_bytes = await photo_file.download_as_bytearray()
            media_files.append(
                {
                    "type": "photo",
                    "data": io.BytesIO(photo_bytes),
                    "mime": "image/jpeg",
                    "filename": f"photo_{photo.file_id}.jpg",
                }
            )
        elif update.message.video:
            has_attached_media = True
            media_type = "video"
            video = update.message.video
            video_file = await context.bot.get_file(video.file_id)
            video_bytes = await video_file.download_as_bytearray()
            media_files.append(
                {
                    "type": "video",
                    "data": io.BytesIO(video_bytes),
                    "mime": "video/mp4",
                    "filename": (
                        video.file_name
                        if hasattr(video, "file_name")
                        else f"video_{video.file_id}.mp4"
                    ),
                }
            )
        elif update.message.voice or update.message.audio:
            has_attached_media = True
            media_type = "audio"
            audio = update.message.voice or update.message.audio
            audio_file = await context.bot.get_file(audio.file_id)
            audio_bytes = await audio_file.download_as_bytearray()
            file_name = (
                getattr(audio, "file_name", None) or f"audio_{audio.file_id}.ogg"
            )
            media_files.append(
                {
                    "type": "audio",
                    "data": io.BytesIO(audio_bytes),
                    "mime": "audio/ogg",
                    "filename": file_name,
                }
            )
        elif update.message.document:
            has_attached_media = True
            media_type = "document"
            document = update.message.document
            document_file = await context.bot.get_file(document.file_id)
            document_bytes = await document_file.download_as_bytearray()

            file_ext = (
                os.path.splitext(document.file_name)[1].lower()
                if document.file_name
                else ""
            )
            mime_type = MediaUtilities.get_mime_type(file_ext)

            # Fallback to content-based detection if extension detection fails
            if mime_type == "application/octet-stream" and document_bytes:
                content_mime = MediaUtilities.detect_mime_from_content(
                    document_bytes[:50]
                )
                if content_mime != "application/octet-stream":
                    mime_type = content_mime

            # Special handling for image documents
            if MediaUtilities.is_image_file(file_ext):
                media_type = "photo"  # Reclassify image documents as photos

            media_files.append(
                {
                    "type": media_type,
                    "data": io.BytesIO(document_bytes),
                    "mime": mime_type,
                    "filename": document.file_name
                    or f"document_{document.file_id}",
                }
            )
        elif update.message.media_group_id:
            # Media group handling logic
            # Note: This logic depends on context.bot_data["media_groups"]
            # We will handle the extraction part here, but the grouping logic might need 
            # to remain or be passed the context appropriately.
            
            has_attached_media = True
            media_type = "media_group"
            if "media_groups" not in context.bot_data:
                context.bot_data["media_groups"] = {}
            media_group_id = update.message.media_group_id
            
            # Since this is a static helper, it modifies the context object passed to it
            # which is the correct behavior for shared state.
            
            if media_group_id in context.bot_data["media_groups"]:
                if update.message.photo:
                    photo = update.message.photo[-1]
                    photo_file = await context.bot.get_file(photo.file_id)
                    photo_bytes = await photo_file.download_as_bytearray()
                    context.bot_data["media_groups"][media_group_id].append(
                        {
                            "type": "photo",
                            "data": io.BytesIO(photo_bytes),
                            "mime": "image/jpeg",
                            "filename": f"photo_{photo.file_id}.jpg",
                        }
                    )
                elif update.message.document:
                    document = update.message.document
                    document_file = await context.bot.get_file(document.file_id)
                    document_bytes = await document_file.download_as_bytearray()
                    file_ext = (
                        os.path.splitext(document.file_name)[1].lower()
                        if document.file_name
                        else ""
                    )
                    mime_type = MediaUtilities.get_mime_type(file_ext)

                    # Special handling for image documents in media groups
                    doc_type = (
                        "photo"
                        if MediaUtilities.is_image_file(file_ext)
                        else "document"
                    )

                    context.bot_data["media_groups"][media_group_id].append(
                        {
                            "type": doc_type,
                            "data": io.BytesIO(document_bytes),
                            "mime": mime_type,
                            "filename": document.file_name
                            or f"document_{document.file_id}",
                        }
                    )
                return False, [], None
            else:
                context.bot_data["media_groups"][media_group_id] = []
                if update.message.photo:
                    photo = update.message.photo[-1]
                    photo_file = await context.bot.get_file(photo.file_id)
                    photo_bytes = await photo_file.download_as_bytearray()
                    context.bot_data["media_groups"][media_group_id].append(
                        {
                            "type": "photo",
                            "data": io.BytesIO(photo_bytes),
                            "mime": "image/jpeg",
                            "filename": f"photo_{photo.file_id}.jpg",
                        }
                    )
                elif update.message.document:
                    document = update.message.document
                    document_file = await context.bot.get_file(document.file_id)
                    document_bytes = await document_file.download_as_bytearray()
                    file_ext = (
                        os.path.splitext(document.file_name)[1].lower()
                        if document.file_name
                        else ""
                    )
                    mime_type = MediaUtilities.get_mime_type(file_ext)

                    # Special handling for image documents in media groups (initialization)
                    doc_type = (
                        "photo"
                        if MediaUtilities.is_image_file(file_ext)
                        else "document"
                    )

                    context.bot_data["media_groups"][media_group_id].append(
                        {
                            "type": doc_type,
                            "data": io.BytesIO(document_bytes),
                            "mime": mime_type,
                            "filename": document.file_name
                            or f"document_{document.file_id}",
                        }
                    )
                
                # The caller should handle triggering the delayed processing
                # We return a signal that this is a new media group
                return True, [], "media_group_start"
                
        return has_attached_media, media_files, media_type

    @staticmethod
    async def send_appropriate_chat_action(
        update: Update, context: ContextTypes.DEFAULT_TYPE, has_attached_media: bool, media_type: str
    ):
        """Send appropriate chat action based on message type."""
        action = ChatAction.TYPING
        if has_attached_media:
            if media_type == "photo":
                action = ChatAction.UPLOAD_PHOTO
            elif media_type == "video":
                action = ChatAction.UPLOAD_VIDEO
            elif media_type == "audio":
                action = ChatAction.RECORD_VOICE
            elif media_type == "document":
                action = ChatAction.UPLOAD_DOCUMENT
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id, action=action
        )
