from typing import Dict, Type
from services.model_handlers import ModelHandler
from services.model_handlers.gemini_handler import GeminiHandler
from services.model_handlers.deepseek_handler import DeepSeekHandler
from services.model_handlers.claude_handler import ClaudeHandler
from services.gemini_api import GeminiAPI


class ModelHandlerFactory:
    """Factory class for creating model handlers."""

    _handlers: Dict[str, ModelHandler] = {}

    @classmethod
    def get_model_handler(
        cls, model_name: str, gemini_api: GeminiAPI = None, claude_api=None
    ) -> ModelHandler:
        """
        Get or create a model handler for the specified model.

        Args:
            model_name: The name of the model to get a handler for.
            gemini_api: An instance of GeminiAPI, required for Gemini model.
            claude_api: An instance of Claude API client, required for Claude model.

        Returns:
            An instance of ModelHandler for the specified model.
        """
        if model_name not in cls._handlers:
            if model_name == "gemini":
                if gemini_api is None:
                    raise ValueError(
                        "GeminiAPI instance is required for Gemini model handler"
                    )
                cls._handlers[model_name] = GeminiHandler(gemini_api)
            elif model_name == "deepseek":
                cls._handlers[model_name] = DeepSeekHandler()
            elif model_name == "claude":
                if claude_api is None:
                    raise ValueError(
                        "Claude API client is required for Claude model handler"
                    )
                cls._handlers[model_name] = ClaudeHandler(claude_api)
            else:
                raise ValueError(f"Unknown model: {model_name}")

        return cls._handlers[model_name]
