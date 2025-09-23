"""
Media analyzer module for processing different types of media in Telegram messages.
"""
import os
import logging
from typing import List, Dict
from src.services.gemini_api import GeminiAPI, create_image_input, create_document_input
from src.services.model_handlers.model_configs import ModelConfigurations, Provider


logger = logging.getLogger(__name__)


class MediaAnalyzer:
    def __init__(self, gemini_api: GeminiAPI, openrouter_api=None):
        """
        Initialize the media analyzer with API connections
        Args:
            gemini_api: Instance of GeminiAPI for processing media
            openrouter_api: Instance of OpenRouterAPI for processing vision-capable models
        """
        self.gemini_api = gemini_api
        self.openrouter_api = openrouter_api
        self.logger = logging.getLogger(__name__)

    async def analyze_media(
        self, media_files: List[Dict], prompt: str, model: str = "gemini"
    ) -> str:
        """
        Analyze media files using the appropriate API based on model provider
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
                return await self._analyze_image(media, prompt, model)
            elif media_type == "video":
                return await self._analyze_video(media, prompt, model)
            elif media_type == "audio":
                return await self._analyze_audio(media, prompt, model)
            elif media_type == "document":
                return await self._analyze_document(media, prompt, model)
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

    def _get_provider_for_model(self, model: str) -> Provider:
        """Determine the provider for a given model"""
        model_config = ModelConfigurations.get_all_models().get(model)
        if model_config:
            return model_config.provider
        
        # Fallback logic based on model name
        if model == "gemini" or model.startswith("gemini"):
            return Provider.GEMINI
        elif model.startswith("google/") and ":free" in model:
            return Provider.OPENROUTER
        elif model.startswith("gpt-") or model.startswith("claude-") or ":" in model:
            return Provider.OPENROUTER
        else:
            return Provider.GEMINI  # Default fallback

    async def _analyze_image(self, media: Dict, prompt: str, model: str) -> str:
        """Process and analyze image media with provider routing"""
        try:
            provider = self._get_provider_for_model(model)
            self.logger.info(f"Using provider {provider.value} for image analysis with model {model}")
            
            if provider == Provider.GEMINI:
                return await self._analyze_image_gemini(media, prompt)
            elif provider == Provider.OPENROUTER:
                return await self._analyze_image_openrouter(media, prompt, model)
            else:
                self.logger.warning(f"Unsupported provider {provider.value} for image analysis, falling back to Gemini")
                return await self._analyze_image_gemini(media, prompt)
                
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"

    async def _analyze_image_gemini(self, media: Dict, prompt: str) -> str:
        """Process image using Gemini API"""
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
                return f"Error analyzing image with Gemini: {error_msg}"
        except Exception as e:
            self.logger.error(f"Error analyzing image with Gemini: {str(e)}")
            return f"Error analyzing image with Gemini: {str(e)}"

    async def _analyze_image_openrouter(self, media: Dict, prompt: str, model: str) -> str:
        """Process image using OpenRouter API"""
        try:
            if not self.openrouter_api:
                self.logger.error("OpenRouter API not available for image analysis")
                return "Error: OpenRouter API not configured for image analysis"
            
            # Check if model supports vision
            model_config = ModelConfigurations.get_all_models().get(model)
            if model_config and hasattr(model_config, 'supports_images') and not model_config.supports_images:
                self.logger.warning(f"Model {model} may not support vision, attempting anyway")
            
            result = await self.openrouter_api.generate_vision_response(
                prompt=prompt,
                image_data=media["data"],
                model=model,
                timeout=120.0  # Longer timeout for vision processing
            )
            
            if result:
                return result
            else:
                return f"Error: No response from OpenRouter vision API for model {model}"
                
        except Exception as e:
            self.logger.error(f"Error analyzing image with OpenRouter: {str(e)}")
            return f"Error analyzing image with OpenRouter: {str(e)}"

    async def _analyze_video(self, media: Dict, prompt: str, model: str) -> str:
        """Process and analyze video media"""
        return f"Analysis of video: {prompt}\n\nThis feature is currently in development. Soon video analysis will be fully supported."

    async def _analyze_audio(self, media: Dict, prompt: str, model: str) -> str:
        """Process and analyze audio media"""
        return f"Analysis of audio: {prompt}\n\nThis feature is currently in development. Soon audio analysis will be fully supported."

    async def _analyze_document(self, media: Dict, prompt: str, model: str) -> str:
        """Process and analyze document media with provider routing"""
        filename = media.get("filename", "document")
        try:
            provider = self._get_provider_for_model(model)
            self.logger.info(f"Using provider {provider.value} for document analysis with model {model}")
            
            if provider == Provider.GEMINI:
                return await self._analyze_document_gemini(media, prompt, filename)
            elif provider == Provider.OPENROUTER:
                # For now, documents are primarily handled by Gemini
                self.logger.info("Document analysis with OpenRouter not yet implemented, falling back to Gemini")
                return await self._analyze_document_gemini(media, prompt, filename)
            else:
                return await self._analyze_document_gemini(media, prompt, filename)
                
        except Exception as e:
            self.logger.error(f"Error analyzing document {filename}: {str(e)}")
            return f"Error analyzing document {filename}: {str(e)}"

    async def _analyze_document_gemini(self, media: Dict, prompt: str, filename: str) -> str:
        """Process document using Gemini API"""
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
                return f"Error analyzing document {filename} with Gemini: {error_msg}"
        except Exception as e:
            self.logger.error(f"Error analyzing document {filename} with Gemini: {str(e)}")
            return f"Error analyzing document {filename} with Gemini: {str(e)}"
