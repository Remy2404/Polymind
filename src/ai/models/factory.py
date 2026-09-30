from typing import Dict
from src.ai.models.model_configs import (
    ModelConfigurations,
    Provider,
    ModelConfig,
)
from src.ai.providers.gemini_api import GeminiAPI
from src.ai.providers.openrouter_api import OpenRouterAPI
from src.ai.providers.deepseek_api import DeepSeekLLM
import logging

logger = logging.getLogger(__name__)


class ModelHandlerFactory:
    """
    Simplified factory that creates unified handlers based on model configurations.
    This eliminates the need for separate handler classes for each model.
    """

    @classmethod
    def get_model_handler(
        cls,
        model_name: str,
        gemini_api: GeminiAPI = None,
        deepseek_api: DeepSeekLLM = None,
        openrouter_api: OpenRouterAPI = None,
    ) -> GeminiAPI | OpenRouterAPI | DeepSeekLLM:
        model_config = ModelConfigurations.get_all_models().get(model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {model_name}")
        providers = {
            Provider.GEMINI: gemini_api,
            Provider.OPENROUTER: openrouter_api,
            Provider.DEEPSEEK: deepseek_api,
        }
        if model_config.provider not in providers:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        api_instance = providers[model_config.provider]
        if api_instance is None:
            raise ValueError(
                f"{model_config.provider.value} API instance is required for model: {model_name}"
            )
        return api_instance

    @classmethod
    def get_available_models(cls) -> Dict[str, ModelConfig]:
        """Get all available model configurations."""
        return ModelConfigurations.get_all_models()
