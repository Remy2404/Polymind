from typing import Dict
from services.model_handlers import ModelHandler
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
    @classmethod
    def get_model_handler(
        cls,
        model_name: str,
        gemini_api: GeminiAPI = None,
        deepseek_api: DeepSeekLLM = None,
        openrouter_api: OpenRouterAPI = None,
    ) -> ModelHandler:
        if model_name not in cls._handlers:
            model_configs = ModelConfigurations.get_all_models()
            model_config = model_configs.get(model_name)
            if not model_config:
                raise ValueError(f"Unknown model: {model_name}")
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
            cls._handlers[model_name] = api_instance
        return cls._handlers[model_name]
    @classmethod
    def get_available_models(cls) -> Dict[str, ModelConfig]:
        """Get all available model configurations."""
        return ModelConfigurations.get_all_models()
    @classmethod
    def add_custom_model(cls, model_config: ModelConfig) -> None:
        """Add a custom model configuration."""
        cls.clear_cache()
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the handler cache (useful for testing or reloading)."""
        cls._handlers.clear()
