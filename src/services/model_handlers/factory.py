from typing import Dict, Type, Optional
from services.model_handlers import ModelHandler
from services.model_handlers.gemini_handler import GeminiHandler
from services.model_handlers.deepseek_handler import DeepSeekHandler
from services.model_handlers.optimus_alpha_handler import OptimusAlphaHandler
from services.model_handlers.openrouter_models import (
    DeepCoderHandler,
    Llama4MaverickHandler,
)
from services.gemini_api import GeminiAPI
from services.openrouter_api import OpenRouterAPI
from services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM


class ModelHandlerFactory:
    """Factory class for creating model handlers."""

    _handlers: Dict[str, ModelHandler] = {}

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
            model_name: The name of the model to get a handler for.
            gemini_api: An instance of GeminiAPI, required for Gemini model.
            deepseek_api: An instance of DeepSeekLLM, required for DeepSeek model.
            openrouter_api: An instance of OpenRouterAPI, required for Quasar Alpha model.

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
                if deepseek_api is None:
                    raise ValueError(
                        "DeepSeekLLM instance is required for DeepSeek model handler"
                    )
                cls._handlers[model_name] = DeepSeekHandler(deepseek_api)
            elif model_name == "optimus-alpha":
                if openrouter_api is None:
                    raise ValueError(
                        "OpenRouterAPI instance is required for Optimus Alpha model handler"
                    )
                cls._handlers[model_name] = OptimusAlphaHandler(openrouter_api)
            elif model_name == "deepcoder":
                if openrouter_api is None:
                    raise ValueError(
                        "OpenRouterAPI instance is required for DeepCoder model handler"
                    )
                cls._handlers[model_name] = DeepCoderHandler(openrouter_api)
            elif model_name == "llama4_maverick":
                if openrouter_api is None:
                    raise ValueError(
                        "OpenRouterAPI instance is required for Llama-4 Maverick model handler"
                    )
                cls._handlers[model_name] = Llama4MaverickHandler(openrouter_api)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            # Ensure handler has implemented format_quoted_message method
            cls._ensure_quoted_message_support(cls._handlers[model_name])

        return cls._handlers[model_name]

    @staticmethod
    def _ensure_quoted_message_support(handler: ModelHandler) -> None:
        """
        Ensure that the handler supports quoted messages by checking if format_quoted_message
        is properly implemented. If not implemented, monkey patch it with a default implementation.

        Args:
            handler: The model handler to check.
        """
        # Check if the handler has a proper implementation of format_quoted_message
        # If not properly implemented, add a default implementation
        if not hasattr(
            handler, "format_quoted_message"
        ) or handler.format_quoted_message.__qualname__.startswith("ModelHandler"):
            # Add default implementation
            def format_quoted_message(
                self, prompt: str, quoted_message: Optional[str]
            ) -> str:
                """Default implementation of format_quoted_message for all model handlers."""
                if quoted_message:
                    return f'The user is replying to this previous message: "{quoted_message}"\n\nUser\'s reply: {prompt}'
                return prompt

            # Monkey patch the method onto the handler instance
            import types

            handler.format_quoted_message = types.MethodType(
                format_quoted_message, handler
            )
