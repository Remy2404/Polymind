import logging
from src.services.model_handlers.model_configs import ModelConfigurations, Provider, ModelConfig

class ModelLogic:
    """
    Handles model configuration, capability checking, and AI response generation.
    Moves logic from MessageHandlers to a specialized class.
    """

    def __init__(self, gemini_api, openrouter_api=None, deepseek_api=None):
        self.logger = logging.getLogger(__name__)
        self.gemini_api = gemini_api
        self.openrouter_api = openrouter_api
        self.deepseek_api = deepseek_api
        self.all_models = ModelConfigurations.get_all_models()
        self._log_initialization()

    def _log_initialization(self):
        self.logger.info(f"Initialized with {len(self.all_models)} available models")
        for provider in Provider:
            provider_models = ModelConfigurations.get_models_by_provider(provider)
            if provider_models:
                self.logger.info(
                    f"{provider.value.title()} models: {len(provider_models)} available"
                )
        free_models = ModelConfigurations.get_free_models()
        self.logger.info(f"Free OpenRouter models available: {len(free_models)}")

    def get_all_models(self) -> dict:
        """Get all available models - useful for external access."""
        return self.all_models

    def get_models_by_provider(self, provider: Provider) -> dict:
        """Get models by specific provider."""
        return ModelConfigurations.get_models_by_provider(provider)

    def get_free_models(self) -> dict:
        """Get all free models."""
        return ModelConfigurations.get_free_models()

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get model configuration by ID."""
        return self.all_models.get(model_id)

    def get_model_indicator_and_config(self, model_id: str) -> tuple[str, ModelConfig]:
        """Get model indicator emoji and configuration for a model."""
        model_config = self.get_model_config(model_id)
        if model_config:
            return (
                f"{model_config.indicator_emoji} {model_config.display_name}",
                model_config,
            )
        else:
            self.logger.warning(f"Unknown model ID: {model_id}, using default")
            return "Unknown Model", None

    async def check_model_supports_media(
        self, user_data_manager, user_id: int, media_type: str
    ) -> tuple[bool, str, str]:
        """
        Check if the user's preferred model supports a specific media type.
        """
        try:
            # Get user's preferred model
            preferred_model = await user_data_manager.get_user_preference(
                user_id, "preferred_model", default="gemini"
            )

            model_config = self.get_model_config(preferred_model)
            if not model_config:
                return False, preferred_model, f"Model '{preferred_model}' not found."

            # Check capabilities based on media type
            if media_type == "image":
                supports_media = getattr(model_config, "supports_images", False)
                capability_name = "images"
            elif media_type == "document":
                supports_media = getattr(model_config, "supports_documents", False)
                capability_name = "documents"
            elif media_type in ["audio", "voice"]:
                supports_media = getattr(model_config, "supports_audio", False)
                capability_name = "audio"
            elif media_type == "video":
                supports_media = getattr(model_config, "supports_video", False)
                capability_name = "video"
            else:
                return False, preferred_model, f"Unsupported media type: {media_type}"

            if not supports_media:
                error_message = (
                    f"❌ **{model_config.display_name}** doesn't support {capability_name}.\n\n"
                    f"💡 **To process {capability_name}, please switch to a vision-capable model:**\n"
                    f"• Use `/switchmodel` command\n"
                    f"• Look for models with 👁️ or ✨ emoji\n"
                    f"• Recommended: **Gemini 2.5 Flash** (supports images & documents)"
                )
                return False, preferred_model, error_message

            return True, preferred_model, ""

        except Exception as e:
            self.logger.error(f"Error checking model media support: {str(e)}")
            return False, "unknown", f"Error checking model capabilities: {str(e)}"

    async def generate_ai_response(
        self,
        prompt: str,
        model_id: str,
        user_id: int,
        conversation_context: list = None,
    ) -> str:
        """Generate AI response using the specified model with conversation context."""
        model_config = self.get_model_config(model_id)
        if not model_config:
            self.logger.error(f"Model configuration not found for: {model_id}")
            return f"Sorry, the model '{model_id}' is not available."
        
        try:
            self.logger.info(f"Generating response with model: {model_id}")
            
            if model_config.provider == Provider.GEMINI:
                return await self.gemini_api.generate_response(prompt)
            elif (
                model_config.provider == Provider.DEEPSEEK
                and hasattr(self, "deepseek_api")
                and self.deepseek_api
            ):
                return await self.deepseek_api.generate_response(prompt)
            elif (
                model_config.provider == Provider.OPENROUTER
                and hasattr(self, "openrouter_api")
                and self.openrouter_api
            ):
                if model_config.openrouter_model_key:
                    return await self.openrouter_api.generate_response_with_model_key(
                        prompt=prompt,
                        openrouter_model_key=model_config.openrouter_model_key,
                        context=conversation_context,
                    )
                else:
                    return await self.openrouter_api.generate_response(
                        prompt=prompt, context=conversation_context, model=model_id
                    )
            else:
                return f"Sorry, the {model_config.display_name} model is currently unavailable."
        except Exception as e:
            self.logger.error(f"Error generating response with {model_id}: {str(e)}")
            return f"Sorry, there was an error with the {model_config.display_name} model: {str(e)}"
