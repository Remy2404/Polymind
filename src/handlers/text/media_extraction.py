"""
Media extraction functionality for text handlers.
"""
import io
import os
import asyncio
import logging
from telegram import Update
from telegram.ext import ContextTypes
from src.handlers.text_processing.utilities import MediaUtilities


class MediaExtractionMixin:
    """Mixin providing media extraction functionality."""

    async def _extract_media_files(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Extract media files from the update."""
        has_attached_media = False
        media_files = []
        media_type = None

        if not update.message:
            return has_attached_media, media_files, media_type

        if update.message.photo:
            return await self._extract_photo(update, context)
        elif update.message.video:
            return await self._extract_video(update, context)
        elif update.message.voice or update.message.audio:
            return await self._extract_audio(update, context)
        elif update.message.document:
            return await self._extract_document(update, context)
        elif update.message.media_group_id:
            return await self._extract_media_group(update, context)

        return has_attached_media, media_files, media_type

    async def _extract_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Extract photo files."""
        photo = update.message.photo[-1]
        photo_file = await context.bot.get_file(photo.file_id)
        photo_bytes = await photo_file.download_as_bytearray()
        return True, [{
            "type": "photo",
            "data": io.BytesIO(photo_bytes),
            "mime": "image/jpeg",
            "filename": f"photo_{photo.file_id}.jpg",
        }], "photo"

    async def _extract_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Extract video files."""
        video = update.message.video
        video_file = await context.bot.get_file(video.file_id)
        video_bytes = await video_file.download_as_bytearray()
        filename = getattr(video, "file_name", None) or f"video_{video.file_id}.mp4"
        return True, [{
            "type": "video",
            "data": io.BytesIO(video_bytes),
            "mime": "video/mp4",
            "filename": filename,
        }], "video"

    async def _extract_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Extract audio/voice files."""
        audio = update.message.voice or update.message.audio
        audio_file = await context.bot.get_file(audio.file_id)
        audio_bytes = await audio_file.download_as_bytearray()
        filename = getattr(audio, "file_name", None) or f"audio_{audio.file_id}.ogg"
        return True, [{
            "type": "audio",
            "data": io.BytesIO(audio_bytes),
            "mime": "audio/ogg",
            "filename": filename,
        }], "audio"

    async def _extract_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Extract document files."""
        document = update.message.document
        document_file = await context.bot.get_file(document.file_id)
        document_bytes = await document_file.download_as_bytearray()

        file_ext = os.path.splitext(document.file_name)[1].lower() if document.file_name else ""
        mime_type = MediaUtilities.get_mime_type(file_ext)

        # Fallback to content-based detection
        if mime_type == "application/octet-stream" and document_bytes:
            content_mime = MediaUtilities.detect_mime_from_content(document_bytes[:50])
            if content_mime != "application/octet-stream":
                mime_type = content_mime

        # Reclassify image documents as photos
        media_type = "photo" if MediaUtilities.is_image_file(file_ext) else "document"

        return True, [{
            "type": media_type,
            "data": io.BytesIO(document_bytes),
            "mime": mime_type,
            "filename": document.file_name or f"document_{document.file_id}",
        }], media_type

    async def _extract_media_group(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Extract media group files."""
        media_group_id = update.message.media_group_id

        if "media_groups" not in context.bot_data:
            context.bot_data["media_groups"] = {}

        # Handle existing or new media group
        if media_group_id not in context.bot_data["media_groups"]:
            context.bot_data["media_groups"][media_group_id] = []
            # Schedule processing after delay
            asyncio.create_task(
                self._process_complete_media_group(
                    media_group_id,
                    update.effective_chat.id,
                    update.effective_user.id,
                    update.message.caption or "",
                    context,
                )
            )

        # Add current media to group
        await self._add_to_media_group(update, context, media_group_id)
        return False, [], None

    async def _add_to_media_group(self, update: Update, context: ContextTypes.DEFAULT_TYPE, group_id: str):
        """Add current message's media to a media group."""
        if update.message.photo:
            photo = update.message.photo[-1]
            photo_file = await context.bot.get_file(photo.file_id)
            photo_bytes = await photo_file.download_as_bytearray()
            context.bot_data["media_groups"][group_id].append({
                "type": "photo",
                "data": io.BytesIO(photo_bytes),
                "mime": "image/jpeg",
                "filename": f"photo_{photo.file_id}.jpg",
            })
        elif update.message.document:
            document = update.message.document
            document_file = await context.bot.get_file(document.file_id)
            document_bytes = await document_file.download_as_bytearray()
            file_ext = os.path.splitext(document.file_name)[1].lower() if document.file_name else ""
            mime_type = MediaUtilities.get_mime_type(file_ext)
            doc_type = "photo" if MediaUtilities.is_image_file(file_ext) else "document"
            context.bot_data["media_groups"][group_id].append({
                "type": doc_type,
                "data": io.BytesIO(document_bytes),
                "mime": mime_type,
                "filename": document.file_name or f"document_{document.file_id}",
            })
