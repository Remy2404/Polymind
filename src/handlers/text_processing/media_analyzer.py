"""
Media analyzer module for processing different types of media in Telegram messages.
"""
import os
import logging
from typing import List, Dict
from src.services.gemini_api import GeminiAPI, create_image_input, create_document_input
logger = logging.getLogger(__name__)
class MediaAnalyzer:
    def __init__(self, gemini_api: GeminiAPI):
        """
        Initialize the media analyzer with API connections
        Args:
            gemini_api: Instance of GeminiAPI for processing media
        """
        self.gemini_api = gemini_api
        self.logger = logging.getLogger(__name__)
    async def analyze_media(
        self, media_files: List[Dict], prompt: str, model: str = "gemini"
    ) -> str:
        """
        Analyze media files using the appropriate API based on media type
        Args:
            media_files: List of media file dictionaries with type, data, and mime
            prompt: The user's prompt or caption
            model: Model name to use
        Returns:
            Analysis text
        """
        if not media_files:
            return None
        media = media_files[0]
        media_type = media["type"]
        try:
            if not prompt or prompt.strip() == "":
                prompt = self._get_default_prompt_for_media_type(media_type, media)
            if media_type == "photo":
                return await self._analyze_image(media, prompt)
            elif media_type == "video":
                return await self._analyze_video(media, prompt)
            elif media_type == "audio":
                return await self._analyze_audio(media, prompt)
            elif media_type == "document":
                return await self._analyze_document(media, prompt)
            return None
        except Exception as e:
            self.logger.error(f"Error analyzing {media_type}: {str(e)}")
            return None
    def _get_default_prompt_for_media_type(self, media_type: str, media: Dict) -> str:
        """Generate a default prompt based on media type if user didn't provide one"""
        if media_type == "photo":
            return "Describe this image in detail."
        elif media_type == "video":
            return "Analyze what's happening in this video."
        elif media_type == "audio":
            return "Transcribe and analyze this audio."
        elif media_type == "document":
            file_ext = os.path.splitext(media.get("filename", ""))[1]
            return f"Analyze the contents of this {file_ext} file."
        else:
            return "Analyze this content."
    async def _analyze_image(self, media: Dict, prompt: str) -> str:
        """Process and analyze image media"""
        try:
            # Create MediaInput for Gemini API using the module-level function
            image_input = create_image_input(media["data"], media.get("filename"))
            
            # Use process_multimodal_input for image analysis
            result = await self.gemini_api.process_multimodal_input(
                text_prompt=prompt,
                media_inputs=[image_input]
            )
            
            if result.success and result.content:
                return result.content
            else:
                error_msg = result.error if result.error else "Unknown error"
                return f"Error analyzing image: {error_msg}"
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"
    async def _analyze_video(self, media: Dict, prompt: str) -> str:
        """Process and analyze video media"""
        return f"Analysis of video: {prompt}\n\nThis feature is currently in development. Soon Gemini will be able to fully analyze video files."
    async def _analyze_audio(self, media: Dict, prompt: str) -> str:
        """Process and analyze audio media"""
        return f"Analysis of audio: {prompt}\n\nThis feature is currently in development. Soon Gemini will be able to fully analyze audio files."
    async def _analyze_document(self, media: Dict, prompt: str) -> str:
        """Process and analyze document media"""
        filename = media.get("filename", "document")
        try:
            # Create MediaInput for Gemini API using the module-level function
            doc_input = create_document_input(media["data"], filename)
            
            # Use process_multimodal_input for document analysis
            result = await self.gemini_api.process_multimodal_input(
                text_prompt=prompt,
                media_inputs=[doc_input]
            )
            
            if result.success and result.content:
                return result.content
            else:
                error_msg = result.error if result.error else "Unknown error"
                return f"Error analyzing document {filename}: {error_msg}"
        except Exception as e:
            self.logger.error(f"Error analyzing document {filename}: {str(e)}")
            return f"Error analyzing document {filename}: {str(e)}"
