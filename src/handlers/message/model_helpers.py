"""
Model helper functions and utilities.
"""
import logging
from typing import Optional
from src.services.model_handlers.model_configs import ModelConfigurations, Provider, ModelConfig


class ModelHelpersMixin:
    """Mixin providing model-related helper methods."""

    def get_all_models(self) -> dict:
        """Get all available models."""
        return self.all_models

    def get_models_by_provider(self, provider: Provider) -> dict:
        """Get models by specific provider."""
        return ModelConfigurations.get_models_by_provider(provider)

    def get_free_models(self) -> dict:
        """Get all free models."""
        return ModelConfigurations.get_free_models()

    def log_model_verification(self, model_id: str) -> bool:
        """Log verification for a specific model and return if it exists."""
        model_config = self.get_model_config(model_id)
        if model_config:
            self.logger.info(f"Model '{model_id}' verified: {model_config.display_name}")
            return True
        self.logger.warning(f"Model '{model_id}' not found in configurations")
        return False

    def get_model_stats(self) -> dict:
        """Get statistics about available models."""
        total_models = len(self.all_models)
        free_models = len(self.get_free_models())
        provider_counts = {
            provider.value: len(self.get_models_by_provider(provider))
            for provider in Provider
        }
        return {
            "total_models": total_models,
            "free_models": free_models,
            "provider_counts": provider_counts,
            "model_ids": list(self.all_models.keys())[:10],
        }

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID."""
        return self.all_models.get(model_id)

    def get_model_indicator_and_config(self, model_id: str) -> tuple[str, Optional[ModelConfig]]:
        """Get model indicator emoji and configuration for a model."""
        model_config = self.get_model_config(model_id)
        if model_config:
            return f"{model_config.indicator_emoji} {model_config.display_name}", model_config
        self.logger.warning(f"Unknown model ID: {model_id}, using default")
        return "Unknown Model", None

    async def check_model_supports_media(self, user_id: int, media_type: str) -> tuple[bool, str, str]:
        """Check if the user's preferred model supports a specific media type."""
        try:
            preferred_model = await self.user_data_manager.get_user_preference(
                user_id, "preferred_model", default="gemini"
            )
            model_config = self.get_model_config(preferred_model)
            if not model_config:
                return False, preferred_model, f"Model '{preferred_model}' not found."

            capability_map = {
                "image": ("supports_images", "images"),
                "document": ("supports_documents", "documents"),
                "audio": ("supports_audio", "audio"),
                "voice": ("supports_audio", "audio"),
                "video": ("supports_video", "video"),
            }

            if media_type not in capability_map:
                return False, preferred_model, f"Unsupported media type: {media_type}"

            attr_name, capability_name = capability_map[media_type]
            supports_media = getattr(model_config, attr_name, False)

            if not supports_media:
                error_message = (
                    f"**{model_config.display_name}** doesn't support {capability_name}.\n\n"
                    f"To process {capability_name}, please switch to a vision-capable model:\n"
                    f"Use `/switchmodel` command\n"
                    f"Recommended: **Gemini 2.5 Flash** (supports images & documents)"
                )
                return False, preferred_model, error_message

            return True, preferred_model, ""
        except Exception as e:
            self.logger.error(f"Error checking model media support: {str(e)}")
            return False, "unknown", f"Error checking model capabilities: {str(e)}"

    async def generate_ai_response(
        self, prompt: str, model_id: str, user_id: int, conversation_context: list = None
    ) -> str:
        """Generate AI response using the specified model with conversation context."""
        model_config = self.get_model_config(model_id)
        if not model_config:
            return f"Sorry, the model '{model_id}' is not available."

        try:
            if model_config.provider == Provider.GEMINI:
                return await self.gemini_api.generate_response(prompt)
            elif model_config.provider == Provider.DEEPSEEK and self.deepseek_api:
                return await self.deepseek_api.generate_response(prompt)
            elif model_config.provider == Provider.OPENROUTER and self.openrouter_api:
                if model_config.openrouter_model_key:
                    return await self.openrouter_api.generate_response_with_model_key(
                        prompt=prompt,
                        openrouter_model_key=model_config.openrouter_model_key,
                        context=conversation_context,
                    )
                return await self.openrouter_api.generate_response(
                    prompt=prompt, context=conversation_context, model=model_id
                )
            return f"Sorry, the {model_config.display_name} model is currently unavailable."
        except Exception as e:
            self.logger.error(f"Error generating response with {model_id}: {str(e)}")
            return f"Sorry, there was an error with the {model_config.display_name} model: {str(e)}"
