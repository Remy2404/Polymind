"""
Model Configuration System
Defines all available models in a centralized configuration.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum


class Provider(Enum):
    """Supported API providers."""

    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"


@dataclass
class ModelConfig:
    """Configuration for an AI model."""

    model_id: str
    display_name: str
    provider: Provider
    system_message: Optional[str] = None
    indicator_emoji: str = "🤖"
    openrouter_model_key: Optional[str] = None
    max_tokens: int = 32000
    default_temperature: float = 0.7
    supports_images: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_documents: bool = False
    description: str = ""
    type: str = "general_purpose"
    capabilities: List[str] = field(default_factory=list)
    supported_parameters: List[str] = field(default_factory=list)


class ModelConfigurations:
    """Central configuration for all available models."""

    @staticmethod
    def get_all_models() -> Dict[str, ModelConfig]:
        """Get all available model configurations from JSON file, merged with hardcoded models."""
        models = {}
        try:
            models_file = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "models.json"
            )
            if os.path.exists(models_file):
                with open(models_file, "r", encoding="utf-8") as f:
                    models_data = json.load(f)

                # Handle the nested structure with categories
                if isinstance(models_data, dict):
                    for category, model_list in models_data.items():
                        if isinstance(model_list, list):
                            for model_data in model_list:
                                if isinstance(model_data, dict):
                                    model_id = model_data.get("id", "")
                                    if not model_id:
                                        continue
                                    provider = (
                                        ModelConfigurations._determine_provider_from_id(
                                            model_id
                                        )
                                    )
                                    capabilities = ModelConfigurations._extract_capabilities_from_model_data(
                                        model_data
                                    )
                                    model_type = (
                                        ModelConfigurations._determine_model_type(
                                            capabilities
                                        )
                                    )
                                    max_tokens = 32000
                                    description = model_data.get(
                                        "description", ""
                                    ).lower()
                                    if (
                                        "reasoning" in description
                                        or "thinking" in description
                                    ):
                                        max_tokens = 65536
                                    elif (
                                        "code" in description
                                        or "programming" in description
                                    ):
                                        max_tokens = 49152
                                    elif any(
                                        keyword in description
                                        for keyword in [
                                            "small",
                                            "lightweight",
                                            "nano",
                                            "mini",
                                        ]
                                    ):
                                        max_tokens = 16384
                                    elif (
                                        "vision" in description
                                        or "multimodal" in description
                                    ):
                                        max_tokens = 24576
                                    config = ModelConfig(
                                        model_id=model_id,
                                        display_name=model_data.get("name", model_id),
                                        provider=provider,
                                        openrouter_model_key=(
                                            model_id
                                            if provider == Provider.OPENROUTER
                                            else None
                                        ),
                                        description=model_data.get("description", ""),
                                        type=model_type,
                                        capabilities=capabilities,
                                        supported_parameters=model_data.get(
                                            "supported_parameters", []
                                        ),
                                        system_message=ModelConfigurations._generate_system_message(
                                            model_id, model_data.get("name", "")
                                        ),
                                        indicator_emoji=ModelConfigurations._get_indicator_emoji(
                                            provider, model_type
                                        ),
                                        max_tokens=max_tokens,
                                        supports_images="supports_images"
                                        in capabilities,
                                        supports_documents="supports_documents"
                                        in capabilities,
                                    )
                                    models[model_id] = config
                # Handle legacy format (simple list)
                elif isinstance(models_data, list):
                    for model_data in models_data:
                        if isinstance(model_data, dict):
                            model_id = model_data.get("id", "")
                            if not model_id:
                                continue
                            provider = ModelConfigurations._determine_provider_from_id(
                                model_id
                            )
                            capabilities = ModelConfigurations._extract_capabilities_from_model_data(
                                model_data
                            )
                            model_type = ModelConfigurations._determine_model_type(
                                capabilities
                            )
                            max_tokens = 32000
                            description = model_data.get("description", "").lower()
                            if "reasoning" in description or "thinking" in description:
                                max_tokens = 65536
                            elif "code" in description or "programming" in description:
                                max_tokens = 49152
                            elif any(
                                keyword in description
                                for keyword in ["small", "lightweight", "nano", "mini"]
                            ):
                                max_tokens = 16384
                            elif "vision" in description or "multimodal" in description:
                                max_tokens = 24576
                            config = ModelConfig(
                                model_id=model_id,
                                display_name=model_data.get("name", model_id),
                                provider=provider,
                                openrouter_model_key=(
                                    model_id
                                    if provider == Provider.OPENROUTER
                                    else None
                                ),
                                description=model_data.get("description", ""),
                                type=model_type,
                                capabilities=capabilities,
                                supported_parameters=model_data.get(
                                    "supported_parameters", []
                                ),
                                system_message=ModelConfigurations._generate_system_message(
                                    model_id, model_data.get("name", "")
                                ),
                                indicator_emoji=ModelConfigurations._get_indicator_emoji(
                                    provider, model_type
                                ),
                                max_tokens=max_tokens,
                                supports_images="supports_images" in capabilities,
                                supports_documents="supports_documents" in capabilities,
                            )
                            models[model_id] = config
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(
                f"Warning: Failed to load models from JSON: {e}. Using hardcoded models."
            )
            return ModelConfigurations._get_hardcoded_models()
        models.update(ModelConfigurations._get_hardcoded_models())
        return models

    @staticmethod
    def _get_hardcoded_models() -> Dict[str, ModelConfig]:
        """Fallback hardcoded models when JSON loading fails."""
        gemini_config = ModelConfig(
            model_id="gemini",
            display_name="Gemini 2.5 Flash",
            provider=Provider.GEMINI,
            indicator_emoji="✨",
            system_message="You are Gemini, a helpful AI assistant created by Google. Be concise, helpful, and accurate.",
            supports_images=True,
            supports_documents=True,
            supported_parameters=[
                "tools",
                "tool_choice",
                "function_calling",
                "long_context",
            ],
            description="Google's latest multimodal AI model with advanced tool calling capabilities",
            type="multimodal",
            max_tokens=32768,
            capabilities=[
                "supports_images",
                "supports_documents",
                "tool_calling",
                "long_context",
                "general_purpose",
            ],
        )
        return {
            "gemini": gemini_config,
            "deepseek": ModelConfig(
                model_id="deepseek",
                display_name="DeepSeek R1",
                provider=Provider.DEEPSEEK,
                indicator_emoji="🧠",
                system_message="You are DeepSeek, an advanced reasoning AI model that excels at complex problem-solving.",
                description="Advanced reasoning model with strong analytical capabilities",
                type="reasoning",
                max_tokens=65536,
                capabilities=[
                    "reasoning_capable",
                    "long_context",
                    "general_purpose",
                    "tool_calling",
                ],
            ),
        }

    @staticmethod
    def _determine_provider_from_id(model_id: str) -> Provider:
        """Determine provider from model ID."""
        if model_id.startswith("gemini"):
            return Provider.GEMINI
        elif model_id.startswith("google/gemini"):
            return Provider.GEMINI
        elif model_id.startswith("google/gemma"):
            # Gemma models should use OpenRouter, not Gemini API
            return Provider.OPENROUTER
        elif model_id.startswith("google/"):
            # Other Google models (like LaMDA, PaLM, etc.) use OpenRouter
            return Provider.OPENROUTER
        elif model_id.startswith("deepseek/"):
            return Provider.DEEPSEEK
        else:
            return Provider.OPENROUTER

    @staticmethod
    def _extract_capabilities_from_model_data(model_data: Dict[str, Any]) -> List[str]:
        """Extract capabilities from model data (description + supported_parameters)."""
        capabilities = []
        description = model_data.get("description", "")
        description_lower = description.lower()
        supported_params = model_data.get("supported_parameters", [])
        if any(
            param in supported_params
            for param in ["tools", "tool_choice", "function_calling"]
        ):
            capabilities.append("tool_calling")
        if not any("tool_calling" in cap for cap in capabilities):
            explicit_phrases = [
                "tool calling",
                "function calling",
                "tool use",
                "function call",
                "tool calls",
                "function calls",
                "native tool use",
                "supports tools",
            ]
            if any(phrase in description_lower for phrase in explicit_phrases):
                capabilities.append("tool_calling")
        if any(
            param in supported_params for param in ["reasoning", "include_reasoning"]
        ):
            capabilities.append("reasoning_capable")
        elif any(
            keyword in description_lower
            for keyword in ["reasoning", "thinking", "logic", "math"]
        ):
            capabilities.append("reasoning_capable")
        if any(
            keyword in description_lower
            for keyword in ["vision", "image", "visual", "multimodal"]
        ):
            capabilities.append("supports_images")
        if any(
            keyword in description_lower
            for keyword in ["code", "programming", "coding", "developer"]
        ):
            capabilities.append("coding_specialist")
        if any(
            keyword in description_lower
            for keyword in ["multilingual", "language", "translation"]
        ):
            capabilities.append("multilingual_support")
        if any(
            keyword in description_lower
            for keyword in ["long", "context", "128k", "256k", "million"]
        ):
            capabilities.append("long_context")
        if not capabilities:
            capabilities.append("general_purpose")
        return capabilities

    @staticmethod
    def _extract_capabilities_from_description(description: str) -> List[str]:
        """Extract capabilities from model description."""
        capabilities = []
        description_lower = description.lower()
        if any(
            keyword in description_lower
            for keyword in ["tool", "function", "calling", "api"]
        ):
            capabilities.append("tool_calling")
        if any(
            keyword in description_lower
            for keyword in ["reasoning", "thinking", "logic", "math"]
        ):
            capabilities.append("reasoning_capable")
        if any(
            keyword in description_lower
            for keyword in ["vision", "image", "visual", "multimodal"]
        ):
            capabilities.append("supports_images")
        if any(
            keyword in description_lower
            for keyword in ["code", "programming", "coding", "developer"]
        ):
            capabilities.append("coding_specialist")
        if any(
            keyword in description_lower
            for keyword in ["multilingual", "language", "translation"]
        ):
            capabilities.append("multilingual_support")
        if any(
            keyword in description_lower
            for keyword in ["long", "context", "128k", "256k"]
        ):
            capabilities.append("long_context")
        if not capabilities:
            capabilities.append("general_purpose")
        return capabilities

    @staticmethod
    def _determine_model_type(capabilities: List[str]) -> str:
        """Determine model type from capabilities."""
        if "supports_images" in capabilities:
            return "vision"
        elif "coding_specialist" in capabilities:
            return "coding_specialist"
        elif "reasoning_capable" in capabilities:
            return "reasoning"
        elif "multilingual_support" in capabilities:
            return "multilingual"
        else:
            return "general_purpose"

    @staticmethod
    def _generate_system_message(model_id: str, display_name: str) -> str:
        """Generate appropriate system message for the model."""
        if "deepseek" in model_id.lower():
            return f"You are {display_name}, an advanced reasoning AI model that excels at complex problem-solving."
        elif "gemini" in model_id.lower():
            return f"You are {display_name}, a helpful AI assistant created by Google. Be concise, helpful, and accurate."
        elif "qwen" in model_id.lower():
            return f"You are {display_name}, a multilingual AI assistant created by Alibaba Cloud."
        elif "llama" in model_id.lower():
            return f"You are {display_name}, an advanced AI assistant by Meta."
        elif "mistral" in model_id.lower():
            return f"You are {display_name}, a powerful and efficient AI assistant by Mistral AI."
        else:
            return f"You are {display_name}, a helpful AI assistant."

    @staticmethod
    def _get_indicator_emoji(provider: Provider, model_type: str) -> str:
        """Get appropriate indicator emoji based on provider and type."""
        if provider == Provider.GEMINI:
            return "✨"
        elif provider == Provider.DEEPSEEK:
            return "🧠"
        elif model_type == "vision":
            return "👁️"
        elif model_type == "coding_specialist":
            return "💻"
        elif model_type == "reasoning":
            return "🤔"
        else:
            return "🤖"

    @staticmethod
    def get_models_by_provider(provider: Provider) -> Dict[str, ModelConfig]:
        """Get all models for a specific provider."""
        if not isinstance(provider, Provider):
            raise ValueError(
                f"Invalid provider: {provider}. Must be a Provider enum value."
            )
        all_models = ModelConfigurations.get_all_models()
        return {k: v for k, v in all_models.items() if v.provider == provider}

    @staticmethod
    def get_models_with_tool_calls() -> Dict[str, ModelConfig]:
        """Get all models that support tool calls based on logic rather than configuration."""
        all_models = ModelConfigurations.get_all_models()
        return {
            k: v
            for k, v in all_models.items()
            if ModelConfigurations._model_supports_tool_calls_logic(k, v)
        }

    @staticmethod
    def _model_supports_tool_calls_logic(
        model_id: str, model_config: ModelConfig
    ) -> bool:
        """
        Determine if a model supports tool calls based on supported_parameters and provider.
        Following OpenRouter documentation and Gemini's capabilities: check supported_parameters
        and provider-specific logic.
        """
        # Explicitly exclude Gemma models as they don't support tool calling
        if model_id.startswith("google/gemma"):
            return False

        if (
            hasattr(model_config, "supported_parameters")
            and model_config.supported_parameters
        ):
            if "tools" in model_config.supported_parameters:
                return True
        if model_config.provider == Provider.DEEPSEEK:
            return True
        elif model_config.provider == Provider.GEMINI:
            return True
        return False

    @staticmethod
    def get_models_with_tool_calls_by_provider(
        provider: Provider,
    ) -> Dict[str, ModelConfig]:
        """Get all models that support tool calls for a specific provider."""
        if not isinstance(provider, Provider):
            raise ValueError(
                f"Invalid provider: {provider}. Must be a Provider enum value."
            )
        tool_call_models = ModelConfigurations.get_models_with_tool_calls()
        return {k: v for k, v in tool_call_models.items() if v.provider == provider}

    @staticmethod
    def model_supports_tool_calls(model_id: str) -> bool:
        """Check if a specific model supports tool calls."""
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Invalid model_id: must be a non-empty string")
        all_models = ModelConfigurations.get_all_models()
        model = all_models.get(model_id)
        if not model:
            return False
        return ModelConfigurations._model_supports_tool_calls_logic(model_id, model)

    @staticmethod
    def get_model_capabilities(model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive capabilities for a model with intelligent detection.
        Returns a dictionary with capability flags and fallback information.
        """
        if not isinstance(model_id, str) or not model_id.strip():
            return ModelConfigurations._get_default_capabilities()

        all_models = ModelConfigurations.get_all_models()
        model = all_models.get(model_id)

        if not model:
            return ModelConfigurations._get_default_capabilities()

        capabilities = {
            "supports_tools": ModelConfigurations._model_supports_tool_calls_logic(
                model_id, model
            ),
            "supports_system_messages": ModelConfigurations._model_supports_system_messages(
                model_id, model
            ),
            "supports_temperature": "temperature" in model.supported_parameters
            if model.supported_parameters
            else True,
            "supports_max_tokens": "max_tokens" in model.supported_parameters
            if model.supported_parameters
            else True,
            "provider": model.provider.value,
            "model_type": model.type,
            "fallback_model": ModelConfigurations._get_fallback_model(model_id, model),
            "limitations": ModelConfigurations._detect_model_limitations(
                model_id, model
            ),
        }

        return capabilities

    @staticmethod
    def _model_supports_system_messages(
        model_id: str, model_config: ModelConfig
    ) -> bool:
        """
        Determine if a model supports system messages/developer instructions.
        Some models (like Gemma) don't support system messages on certain providers.
        """
        # Known models that don't support system messages
        unsupported_system_message_models = [
            "google/gemma",
        ]

        for unsupported_prefix in unsupported_system_message_models:
            if model_id.startswith(unsupported_prefix):
                return False

        # Provider-specific logic
        if model_config.provider == Provider.OPENROUTER:
            # For OpenRouter, check if it's a Google AI Studio model that might not support system messages
            if model_id.startswith("google/"):
                # Some Google models on OpenRouter may not support system messages
                return False

        # Default to True for most models
        return True

    @staticmethod
    def _detect_model_limitations(
        model_id: str, model_config: ModelConfig
    ) -> List[str]:
        """
        Detect specific limitations of a model based on its configuration and known issues.
        """
        limitations = []

        # Check for tool calling limitations
        if not ModelConfigurations._model_supports_tool_calls_logic(
            model_id, model_config
        ):
            limitations.append("no_tool_calling")

        # Check for system message limitations
        if not ModelConfigurations._model_supports_system_messages(
            model_id, model_config
        ):
            limitations.append("no_system_messages")

        # Check for parameter limitations
        if model_config.supported_parameters:
            all_params = [
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "frequency_penalty",
                "presence_penalty",
            ]
            unsupported_params = [
                param
                for param in all_params
                if param not in model_config.supported_parameters
            ]
            if unsupported_params:
                limitations.append(f"limited_parameters:{','.join(unsupported_params)}")

        return limitations

    @staticmethod
    def _get_fallback_model(model_id: str, model_config: ModelConfig) -> Optional[str]:
        """
        Get an appropriate fallback model for when the current model has limitations.
        """
        # For Gemma models, suggest Gemini as fallback
        if model_id.startswith("google/gemma"):
            return "google/gemini-2.0-flash-exp:free"

        # For other Google models with limitations, suggest DeepSeek
        if (
            model_id.startswith("google/")
            and model_config.provider == Provider.OPENROUTER
        ):
            return "deepseek/deepseek-chat-v3.1:free"

        # Default fallback
        return "deepseek/deepseek-chat-v3.1:free"

    @staticmethod
    def _get_default_capabilities() -> Dict[str, Any]:
        """Get default capabilities for unknown models."""
        return {
            "supports_tools": False,
            "supports_system_messages": True,
            "supports_temperature": True,
            "supports_max_tokens": True,
            "provider": "unknown",
            "model_type": "general_purpose",
            "fallback_model": "deepseek/deepseek-chat-v3.1:free",
            "limitations": ["unknown_model"],
        }

    @staticmethod
    def get_safe_model_config(model_id: str) -> Dict[str, Any]:
        """
        Get a safe configuration for a model that handles all limitations gracefully.
        This is the main method that should be used by API handlers.
        """
        capabilities = ModelConfigurations.get_model_capabilities(model_id)

        safe_config = {
            "model_id": model_id,
            "capabilities": capabilities,
            "use_tools": capabilities["supports_tools"],
            "use_system_message": capabilities["supports_system_messages"],
            "temperature_supported": capabilities["supports_temperature"],
            "max_tokens_supported": capabilities["supports_max_tokens"],
            "fallback_available": capabilities["fallback_model"] is not None,
            "fallback_model": capabilities["fallback_model"],
            "warnings": ModelConfigurations._generate_capability_warnings(capabilities),
        }

        return safe_config

    @staticmethod
    def _generate_capability_warnings(capabilities: Dict[str, Any]) -> List[str]:
        """Generate user-friendly warnings about model limitations."""
        warnings = []

        if not capabilities["supports_tools"]:
            warnings.append(
                "This model doesn't support tool calling. Advanced features may be limited."
            )

        if not capabilities["supports_system_messages"]:
            warnings.append(
                "This model doesn't support system instructions. Responses may be less structured."
            )

        if capabilities["limitations"]:
            for limitation in capabilities["limitations"]:
                if limitation == "no_tool_calling":
                    warnings.append("Tool calling is not available for this model.")
                elif limitation == "no_system_messages":
                    warnings.append("System messages are not supported by this model.")
                elif limitation.startswith("limited_parameters"):
                    params = limitation.split(":")[1]
                    warnings.append(f"Some parameters are not supported: {params}")

        return warnings

    @staticmethod
    def adapt_request_for_model(
        model_id: str, request_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt API request parameters based on model capabilities.
        Removes unsupported parameters and adjusts values as needed.
        """
        capabilities = ModelConfigurations.get_model_capabilities(model_id)

        adapted_params = request_params.copy()

        # Remove unsupported parameters
        if not capabilities["supports_temperature"] and "temperature" in adapted_params:
            del adapted_params["temperature"]

        if not capabilities["supports_max_tokens"] and "max_tokens" in adapted_params:
            del adapted_params["max_tokens"]

        # Handle system message
        if not capabilities["supports_system_messages"]:
            if "messages" in adapted_params:
                # Remove system messages from the messages array
                adapted_params["messages"] = [
                    msg
                    for msg in adapted_params["messages"]
                    if msg.get("role") != "system"
                ]

        # Handle tools
        if not capabilities["supports_tools"]:
            if "tools" in adapted_params:
                del adapted_params["tools"]
            if "tool_choice" in adapted_params:
                del adapted_params["tool_choice"]

        return adapted_params

    @staticmethod
    def get_free_models() -> Dict[str, ModelConfig]:
        """Get all free models (OpenRouter models with :free suffix)."""
        all_models = ModelConfigurations.get_all_models()
        return {
            k: v
            for k, v in all_models.items()
            if v.provider == Provider.OPENROUTER
            and v.openrouter_model_key
            and ":free" in v.openrouter_model_key
        }

    @staticmethod
    def add_openrouter_models(additional_models: List[Dict[str, Any]]) -> None:
        """
        Easily add more OpenRouter models.
        Args:
            additional_models: List of model dictionaries with keys:
                - model_id, display_name, openrouter_model_key, indicator_emoji, etc.
        Raises:
            ValueError: If input validation fails
        """
        if not isinstance(additional_models, list):
            raise ValueError("additional_models must be a list")
        for i, model_data in enumerate(additional_models):
            if not isinstance(model_data, dict):
                raise ValueError(f"Model at index {i} must be a dictionary")
            required_keys = ["model_id", "display_name", "openrouter_model_key"]
            for key in required_keys:
                if key not in model_data:
                    raise ValueError(f"Model at index {i} missing required key: {key}")
                if not isinstance(model_data[key], str) or not model_data[key].strip():
                    raise ValueError(
                        f"Model at index {i} {key} must be a non-empty string"
                    )
        current_models = ModelConfigurations.get_all_models()
        for model_data in additional_models:
            model_config = ModelConfig(
                model_id=model_data["model_id"],
                display_name=model_data["display_name"],
                provider=Provider.OPENROUTER,
                openrouter_model_key=model_data["openrouter_model_key"],
                indicator_emoji=model_data.get("indicator_emoji", "🤖"),
                system_message=model_data.get("system_message"),
                description=model_data.get("description", ""),
            )
            current_models[model_data["model_id"]] = model_config

    @staticmethod
    def get_model_with_fallback(model_id: str) -> str:
        """Get OpenRouter model key with fallback to reliable alternatives"""
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Invalid model_id: must be a non-empty string")

        # If model has provider prefix (e.g., alibaba/, openai/, etc.) with version suffix, use as-is
        # This handles: alibaba/tongyi-deepresearch-30b-a3b:free, meta-llama/llama-3.3-70b:free, etc.
        if "/" in model_id and (":" in model_id or model_id in ["gemini", "deepseek"]):
            return model_id

        model_map = {
            "gemini": "gemini",
            "deepseek": "deepseek",
        }
        fallback_map = {
            "gemini": "gemini",
            "deepseek": "deepseek",
            "deepseek/deepseek-chat-v3.1:free": "deepseek/deepseek-r1:free",
            "meta-llama/llama-4-maverick:free": "meta-llama/llama-3.3-70b-instruct:free",
        }
        if model_id in model_map:
            return model_map[model_id]
        if model_id in fallback_map:
            return fallback_map[model_id]
        if model_id.startswith("mistralai/"):
            return "mistralai/mistral-small-3.2-24b-instruct:free"
        elif model_id.startswith("qwen/"):
            return "qwen/qwen3-8b:free"
        elif model_id.startswith("deepseek/"):
            return "deepseek/deepseek-chat-v3.1:free"
        elif model_id.startswith("google/gemini") or model_id.startswith("gemini"):
            return "google/gemini-2.0-flash-exp:free"
        elif model_id.startswith("google/gemma"):
            # Gemma models should use OpenRouter
            return "google/gemma-3-12b-it:free"
        elif model_id.startswith("google/"):
            # Other Google models use OpenRouter
            return "google/gemini-2.0-flash-exp:free"
        elif model_id.startswith("meta-llama/"):
            return "meta-llama/llama-4-maverick:free"
        elif model_id.startswith("alibaba/"):
            # Alibaba models - return as-is for OpenRouter
            return model_id

        # Default fallback
        return "deepseek/deepseek-chat-v3.1:free"
