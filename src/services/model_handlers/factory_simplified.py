from typing import Dict, Optional
from services.model_handlers import ModelHandler
from services.model_handlers.unified_handler import UnifiedModelHandler
from services.model_handlers.model_configs import (
    ModelConfigurations,
    Provider,
    ModelConfig,
)
from services.gemini_api import GeminiAPI
from services.openrouter_api import OpenRouterAPI
from services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM
import logging

logger = logging.getLogger(__name__)


class ModelHandlerFactory:
    """
    Simplified factory that creates unified handlers based on model configurations.
    This eliminates the need for separate handler classes for each model.
    """

    _handlers: Dict[str, ModelHandler] = {}
    _model_configs = ModelConfigurations.get_all_models()

    @classmethod
    def get_model_handler(
        cls,
        model_name: str,
        gemini_api: GeminiAPI = None,
        deepseek_api: DeepSeekLLM = None,
        openrouter_api: OpenRouterAPI = None,
    ) -> ModelHandler:
        """
        Get or create a model handler for the specified model.

        Args:
            model_name: The model identifier
            gemini_api: Gemini API instance
            deepseek_api: DeepSeek API instance
            openrouter_api: OpenRouter API instance

        Returns:
            ModelHandler instance for the specified model
        """
        if model_name not in cls._handlers:
            # Get model configuration
            model_config = cls._model_configs.get(model_name)
            if not model_config:
                raise ValueError(f"Unknown model: {model_name}")

            # Get the appropriate API instance based on provider
            api_instance = None
            if model_config.provider == Provider.GEMINI:
                if gemini_api is None:
                    raise ValueError(
                        f"GeminiAPI instance is required for model: {model_name}"
                    )
                api_instance = gemini_api
            elif model_config.provider == Provider.OPENROUTER:
                if openrouter_api is None:
                    raise ValueError(
                        f"OpenRouterAPI instance is required for model: {model_name}"
                    )
                api_instance = openrouter_api
            elif model_config.provider == Provider.DEEPSEEK:
                if deepseek_api is None:
                    raise ValueError(
                        f"DeepSeekLLM instance is required for model: {model_name}"
                    )
                api_instance = deepseek_api
            else:
                raise ValueError(f"Unsupported provider: {model_config.provider}")

            # Create unified handler
            cls._handlers[model_name] = UnifiedModelHandler(
                model_id=model_config.model_id,
                provider=model_config.provider.value,
                display_name=model_config.display_name,
                api_instance=api_instance,
                system_message=model_config.system_message,
                model_indicator=f"{model_config.indicator_emoji} {model_config.display_name}",
                openrouter_model_key=model_config.openrouter_model_key,
            )

        return cls._handlers[model_name]

    @classmethod
    def get_available_models(cls) -> Dict[str, ModelConfig]:
        """Get all available model configurations."""
        return cls._model_configs

    @classmethod
    def add_custom_model(cls, model_config: ModelConfig) -> None:
        """Add a custom model configuration."""
        cls._model_configs[model_config.model_id] = model_config

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the handler cache (useful for testing or reloading)."""
        cls._handlers.clear()
