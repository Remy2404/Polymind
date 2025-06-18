"""
Multimodal Processing Helper for Telegram Bot
Simplifies the integration of text + images + documents processing
Updated for new Gemini 2.0 Flash API
"""

import io
import logging
from typing import List, Dict, Any, Optional, Union
from telegram import Update, Message, PhotoSize, Document, Audio, Video, Voice

from services.gemini_api import (
    GeminiAPI, 
    MediaInput, 
    MediaType, 
    ProcessingResult,
    create_image_input,
    create_document_input,
    create_text_input
)


class TelegramMultimodalProcessor:
    """
    Helper class to process Telegram messages with multiple media types
    Integrates seamlessly with the new Gemini API
    """
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)
    
    async def process_telegram_message(
        self,
        message: Message,
        custom_prompt: Optional[str] = None,
        context: Optional[List[Dict]] = None
    ) -> ProcessingResult:
        """
        Process a Telegram message with all its media content
        
        Args:
            message: Telegram message object
            custom_prompt: Optional custom prompt to override message text
            context: Conversation context
            
        Returns:
            ProcessingResult with the AI response
        """
        try:
            # Extract text prompt
            text_prompt = custom_prompt or message.text or message.caption or "Please analyze this content."
            
            # Extract media inputs
            media_inputs = await self._extract_media_from_message(message)
            
            # Process with Gemini
            result = await self.gemini_api.process_multimodal_input(
                text_prompt=text_prompt,
                media_inputs=media_inputs,
                context=context
            )
            
            self.logger.info(f"Processed message with {len(media_inputs)} media items")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process Telegram message: {e}")
            return ProcessingResult(
                success=False,
                error=f"Message processing failed: {str(e)}"
            )
    
    async def _extract_media_from_message(self, message: Message) -> List[MediaInput]:
        """Extract all media from a Telegram message"""
        media_inputs = []
        
        try:
            # Process photos
            if message.photo:
                photo_input = await self._process_telegram_photo(message.photo)
                if photo_input:
                    media_inputs.append(photo_input)
            
            # Process documents
            if message.document:
                doc_input = await self._process_telegram_document(message.document)
                if doc_input:
                    media_inputs.append(doc_input)
            
            # Process audio
            if message.audio:
                audio_input = await self._process_telegram_audio(message.audio)
                if audio_input:
                    media_inputs.append(audio_input)
            
            # Process video
            if message.video:
                video_input = await self._process_telegram_video(message.video)
                if video_input:
                    media_inputs.append(video_input)
            
            # Process voice messages
            if message.voice:
                voice_input = await self._process_telegram_voice(message.voice)
                if voice_input:
                    media_inputs.append(voice_input)
                    
        except Exception as e:
            self.logger.error(f"Failed to extract media from message: {e}")
        
        return media_inputs
    
    async def _process_telegram_photo(self, photos: List[PhotoSize]) -> Optional[MediaInput]:
        """Process Telegram photo"""
        try:
            # Get the highest resolution photo
            photo = max(photos, key=lambda p: p.width * p.height)
            
            # Download photo data
            photo_file = await photo.get_file()
            photo_data = io.BytesIO()
            await photo_file.download_to_memory(photo_data)
            
            return create_image_input(photo_data, f"photo_{photo.file_id}.jpg")
            
        except Exception as e:
            self.logger.error(f"Failed to process photo: {e}")
            return None
    
    async def _process_telegram_document(self, document: Document) -> Optional[MediaInput]:
        """Process Telegram document"""
        try:
            # Check file size (limit to 50MB)
            if document.file_size and document.file_size > 50 * 1024 * 1024:
                self.logger.warning(f"Document too large: {document.file_size} bytes")
                return MediaInput(
                    type=MediaType.DOCUMENT,
                    data=f"[Large document: {document.file_name} ({document.file_size} bytes) - too large to process]",
                    mime_type="text/plain",
                    filename=document.file_name
                )
            
            # Download document
            doc_file = await document.get_file()
            doc_data = io.BytesIO()
            await doc_file.download_to_memory(doc_data)
            
            return create_document_input(doc_data, document.file_name or "document")
            
        except Exception as e:
            self.logger.error(f"Failed to process document: {e}")
            return None
    
    async def _process_telegram_audio(self, audio: Audio) -> Optional[MediaInput]:
        """Process Telegram audio"""
        try:
            return MediaInput(
                type=MediaType.AUDIO,
                data=f"[Audio file: {audio.file_name or 'audio'} - {audio.duration}s]",
                mime_type="audio/mpeg",
                filename=audio.file_name,
                metadata={"duration": audio.duration, "performer": audio.performer, "title": audio.title}
            )
        except Exception as e:
            self.logger.error(f"Failed to process audio: {e}")
            return None
    
    async def _process_telegram_video(self, video: Video) -> Optional[MediaInput]:
        """Process Telegram video"""
        try:
            return MediaInput(
                type=MediaType.VIDEO,
                data=f"[Video file: {video.file_name or 'video'} - {video.duration}s, {video.width}x{video.height}]",
                mime_type="video/mp4",
                filename=video.file_name,
                metadata={"duration": video.duration, "width": video.width, "height": video.height}
            )
        except Exception as e:
            self.logger.error(f"Failed to process video: {e}")
            return None
    
    async def _process_telegram_voice(self, voice: Voice) -> Optional[MediaInput]:
        """Process Telegram voice message"""
        try:
            return MediaInput(
                type=MediaType.AUDIO,
                data=f"[Voice message - {voice.duration}s]",
                mime_type="audio/ogg",
                filename=f"voice_{voice.file_id}.ogg",
                metadata={"duration": voice.duration}
            )
        except Exception as e:
            self.logger.error(f"Failed to process voice: {e}")
            return None


class BatchProcessor:
    """
    Process multiple files at once
    Useful for handling albums or multiple documents
    """
    
    def __init__(self, gemini_api: GeminiAPI):
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)
    
    async def process_file_batch(
        self,
        files: List[Dict[str, Any]],
        prompt: str = "Analyze these files and provide a comprehensive summary.",
        context: Optional[List[Dict]] = None
    ) -> ProcessingResult:
        """
        Process multiple files in a single request
        
        Args:
            files: List of file dictionaries with 'data', 'filename', 'type'
            prompt: Text prompt for analysis
            context: Conversation context
            
        Returns:
            ProcessingResult with combined analysis
        """
        try:
            media_inputs = []
            
            for file_info in files:
                if file_info['type'] == 'image':
                    media_inputs.append(create_image_input(
                        file_info['data'], 
                        file_info.get('filename', 'image')
                    ))
                elif file_info['type'] == 'document':
                    media_inputs.append(create_document_input(
                        file_info['data'],
                        file_info.get('filename', 'document')
                    ))
                else:
                    # Handle other types
                    media_inputs.append(MediaInput(
                        type=MediaType.DOCUMENT,
                        data=file_info['data'],
                        mime_type="application/octet-stream",
                        filename=file_info.get('filename', 'file')
                    ))
            
            # Process all files together
            enhanced_prompt = f"Analyzing {len(media_inputs)} files: {prompt}"
            
            result = await self.gemini_api.process_multimodal_input(
                text_prompt=enhanced_prompt,
                media_inputs=media_inputs,
                context=context
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return ProcessingResult(
                success=False,
                error=f"Batch processing failed: {str(e)}"
            )


# Utility functions for common use cases
async def quick_image_analysis(gemini_api: GeminiAPI, image_data: bytes, prompt: str) -> str:
    """Quick image analysis helper"""
    media_input = create_image_input(image_data)
    result = await gemini_api.process_multimodal_input(
        text_prompt=prompt,
        media_inputs=[media_input]
    )
    return result.content if result.success else f"Error: {result.error}"


async def quick_document_analysis(gemini_api: GeminiAPI, doc_data: bytes, filename: str, prompt: str) -> str:
    """Quick document analysis helper"""
    media_input = create_document_input(doc_data, filename)
    result = await gemini_api.process_multimodal_input(
        text_prompt=prompt,
        media_inputs=[media_input]
    )
    return result.content if result.success else f"Error: {result.error}"


async def analyze_mixed_content(
    gemini_api: GeminiAPI, 
    text: str, 
    image_data: Optional[bytes] = None, 
    doc_data: Optional[bytes] = None,
    doc_filename: Optional[str] = None
) -> str:
    """Analyze mixed content (text + optional image + optional document)"""
    media_inputs = []
    
    if image_data:
        media_inputs.append(create_image_input(image_data))
    
    if doc_data and doc_filename:
        media_inputs.append(create_document_input(doc_data, doc_filename))
    
    result = await gemini_api.process_multimodal_input(
        text_prompt=text,
        media_inputs=media_inputs if media_inputs else None
    )
    
    return result.content if result.success else f"Error: {result.error}"
