"""
🎯 SIMPLIFIED UNIFIED API MANAGEMENT SYSTEM 🎯

This single file replaces:
- factory.py
- gemini_handler.py
- deepseek_handler.py
- model_configs.py
- unified_handler.py

Benefits:
✅ One file to manage all APIs
✅ Easy to add new models/providers
✅ Simplified switching between models
✅ Reduced code complexity
✅ Centralized configuration
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any, Union
from enum import Enum
from dataclasses import dataclass

# Import the actual API classes
from services.gemini_api import GeminiAPI
from services.openrouter_api import OpenRouterAPI
from services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM

logger = logging.getLogger(__name__)


class APIProvider(Enum):
    """Supported API providers"""

    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"


@dataclass
class ModelConfig:
    """Configuration for a single model"""

    model_id: str
    display_name: str
    provider: APIProvider
    emoji: str
    description: str = ""
    system_message: str = ""
    openrouter_key: Optional[str] = None


class SuperSimpleAPIManager:
    """
    🚀 SUPER SIMPLE API MANAGER 🚀

    One class to rule them all! Manages Gemini, OpenRouter, and DeepSeek
    through a single, easy-to-use interface.
    """

    def __init__(
        self,
        gemini_api: Optional[GeminiAPI] = None,
        deepseek_api: Optional[DeepSeekLLM] = None,
        openrouter_api: Optional[OpenRouterAPI] = None,
    ):
        """Initialize with your API instances"""
        self.apis = {
            APIProvider.GEMINI: gemini_api,
            APIProvider.DEEPSEEK: deepseek_api,
            APIProvider.OPENROUTER: openrouter_api,
        }
        self.logger = logging.getLogger(__name__)
        self._setup_models()

    def _setup_models(self):
        """🎨 Configure all your models here - Easy to add new ones!"""
        self.models: Dict[str, ModelConfig] = {
            # 🧠 GEMINI MODELS
            "gemini": ModelConfig(
                model_id="gemini",
                display_name="Gemini 2.0 Flash",
                provider=APIProvider.GEMINI,
                emoji="🧠",
                description="Google's latest multimodal AI",
                system_message="You are Gemini, a friendly AI assistant by Google. Be helpful and concise! 😊",
            ),
            # 🔮 DEEPSEEK MODELS
            "deepseek": ModelConfig(
                model_id="deepseek",
                display_name="DeepSeek R1",
                provider=APIProvider.DEEPSEEK,
                emoji="🔮",
                description="Advanced reasoning AI model",
                system_message="You are DeepSeek, an advanced reasoning AI. Think deeply and provide insights! ✨",
            ),
            # 🌐 OPENROUTER MODELS (Free!)
            "llama4_maverick": ModelConfig(
                model_id="llama4_maverick",
                display_name="Llama 4 Maverick",
                provider=APIProvider.OPENROUTER,
                emoji="🦙",
                description="Meta's latest Llama 4 model",
                openrouter_key="meta-llama/llama-4-maverick:free",
            ),
            "deepcoder": ModelConfig(
                model_id="deepcoder",
                display_name="DeepCoder 14B",
                provider=APIProvider.OPENROUTER,
                emoji="💻",
                description="AI specialized in programming",
                openrouter_key="agentica-org/deepcoder-14b-preview:free",
            ),
            "qwen3-32b": ModelConfig(
                model_id="qwen3-32b",
                display_name="Qwen3 32B",
                provider=APIProvider.OPENROUTER,
                emoji="🎯",
                description="Large parameter Qwen model",
                openrouter_key="qwen/qwen3-32b-a3b:free",
            ),
            "mistral-7b": ModelConfig(
                model_id="mistral-7b",
                display_name="Mistral 7B",
                provider=APIProvider.OPENROUTER,
                emoji="⚡",
                description="High-performance European AI",
                openrouter_key="mistralai/mistral-7b-instruct:free",
            ),
            "gemma-2-9b": ModelConfig(
                model_id="gemma-2-9b",
                display_name="Gemma 2 9B",
                provider=APIProvider.OPENROUTER,
                emoji="💎",
                description="Google's lightweight model",
                openrouter_key="google/gemma-2-9b-it:free",
            ),
            "llama-3.1-8b": ModelConfig(
                model_id="llama-3.1-8b",
                display_name="Llama 3.1 8B",
                provider=APIProvider.OPENROUTER,
                emoji="🌟",
                description="Meta's popular 8B model",
                openrouter_key="meta-llama/llama-3.1-8b-instruct:free",
            ),
            "phi-4-reasoning": ModelConfig(
                model_id="phi-4-reasoning",
                display_name="Phi-4 Reasoning+",
                provider=APIProvider.OPENROUTER,
                emoji="🧮",
                description="Microsoft's reasoning specialist",
                openrouter_key="microsoft/phi-4-reasoning-plus:free",
            ),
            "deepseek-r1-qwen": ModelConfig(
                model_id="deepseek-r1-qwen",
                display_name="DeepSeek R1 Qwen",
                provider=APIProvider.OPENROUTER,
                emoji="🧩",
                description="Latest DeepSeek with Qwen3",
                openrouter_key="deepseek/deepseek-r1-0528-qwen3-8b:free",
            ),
            # 🎯 ADD MORE MODELS HERE - Just copy the pattern above!
        }

    async def chat(
        self,
        model_id: str,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        quoted_message: Optional[str] = None,
    ) -> str:
        """
        🗣️ Chat with any model using this simple method!

        Args:
            model_id: Which model to use (e.g., "gemini", "deepcoder", "llama4_maverick")
            prompt: Your message/question
            context: Previous conversation history (optional)
            temperature: Creativity level (0.0-1.0)
            max_tokens: Maximum response length
            quoted_message: If replying to a specific message

        Returns:
            AI response as string
        """
        try:
            # Get model configuration
            model_config = self.models.get(model_id)
            if not model_config:
                available = list(self.models.keys())
                return (
                    f"❌ Unknown model '{model_id}'. Available: {', '.join(available)}"
                )

            # Get API instance
            api_instance = self.apis.get(model_config.provider)
            if not api_instance:
                return f"❌ {model_config.provider.value} API not available"

            # Add quoted message context if provided
            if quoted_message:
                prompt = f'Replying to: "{quoted_message}"\n\nUser: {prompt}'

            # Route to the right API
            if model_config.provider == APIProvider.GEMINI:
                return await self._call_gemini(api_instance, prompt, context)

            elif model_config.provider == APIProvider.DEEPSEEK:
                return await self._call_deepseek(
                    api_instance, prompt, context, model_config, temperature, max_tokens
                )

            elif model_config.provider == APIProvider.OPENROUTER:
                return await self._call_openrouter(
                    api_instance, prompt, context, model_config, temperature, max_tokens
                )

            return "❌ Unsupported provider"

        except Exception as e:
            self.logger.error(f"Error with {model_id}: {e}")
            return f"❌ Error with {model_config.display_name if model_config else model_id}. Please try again!"

    async def _call_gemini(
        self, api: GeminiAPI, prompt: str, context: Optional[List]
    ) -> str:
        """Call Gemini API"""
        return await api.generate_response(prompt=prompt, context=context)

    async def _call_deepseek(
        self,
        api: DeepSeekLLM,
        prompt: str,
        context: Optional[List],
        model_config: ModelConfig,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call DeepSeek API"""
        messages = []

        # Add system message
        if model_config.system_message:
            messages.append({"role": "system", "content": model_config.system_message})

        # Add conversation history
        if context:
            messages.extend(context)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        return await api.generate_text(
            messages=messages, temperature=temperature, max_tokens=max_tokens
        )

    async def _call_openrouter(
        self,
        api: OpenRouterAPI,
        prompt: str,
        context: Optional[List],
        model_config: ModelConfig,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call OpenRouter API"""
        return await api.generate_response(
            prompt=prompt,
            model=model_config.openrouter_key or model_config.model_id,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # 🎯 SIMPLE HELPER METHODS

    def get_all_models(self) -> Dict[str, ModelConfig]:
        """Get all available models"""
        return self.models

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.models.get(model_id)

    def get_models_by_provider(self, provider: APIProvider) -> Dict[str, ModelConfig]:
        """Get all models for a specific provider"""
        return {k: v for k, v in self.models.items() if v.provider == provider}

    def get_model_display(self, model_id: str) -> str:
        """Get emoji + name for display"""
        config = self.models.get(model_id)
        return f"{config.emoji} {config.display_name}" if config else "❓ Unknown"

    def list_available_models(self) -> str:
        """Get a formatted list of all models"""
        lines = ["🤖 Available AI Models:\n"]

        # Group by provider
        for provider in APIProvider:
            provider_models = self.get_models_by_provider(provider)
            if provider_models:
                lines.append(f"*{provider.value.title()} Models:*")
                for model_id, config in provider_models.items():
                    lines.append(
                        f"• {config.emoji} {config.display_name} - {config.description}"
                    )
                lines.append("")

        return "\n".join(lines)

    def add_model(self, model_config: ModelConfig) -> None:
        """🆕 Add a new model - Super easy!"""
        self.models[model_config.model_id] = model_config
        self.logger.info(f"Added new model: {model_config.display_name}")


# 🎯 CONVENIENCE ALIAS - Use this in your code
UnifiedAPIManager = SuperSimpleAPIManager
