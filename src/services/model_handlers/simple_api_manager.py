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
from src.services.gemini_api import GeminiAPI
from src.services.openrouter_api import OpenRouterAPI
from src.services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM

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
    max_tokens: int = 32000
    type: str = "general_purpose"


# Provider groups for hierarchical model selection
PROVIDER_GROUPS = {
    "🤖 Gemini Models": {
        "provider": APIProvider.GEMINI,
        "description": "Google's Gemini AI models",
        "models": []  # Will be populated dynamically
    },
    "🧠 DeepSeek Models": {
        "provider": APIProvider.DEEPSEEK,
        "description": "DeepSeek reasoning models",
        "models": []  # Will be populated dynamically
    },
    "🔄 OpenRouter Models": {
        "provider": APIProvider.OPENROUTER,
        "description": "Multiple AI models via OpenRouter",
        "models": []  # Will be populated dynamically
    }
}


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
        # Import from the centralized configuration
        from src.services.model_handlers.model_configs import ModelConfigurations, Provider
        
        # Get all models from the configuration
        model_configs = ModelConfigurations.get_all_models()
        
        # Convert to our format
        self.models: Dict[str, ModelConfig] = {}
        
        for model_id, config in model_configs.items():
            # Map Provider enum to APIProvider enum
            api_provider = None
            if config.provider == Provider.GEMINI:
                api_provider = APIProvider.GEMINI
            elif config.provider == Provider.DEEPSEEK:
                api_provider = APIProvider.DEEPSEEK
            elif config.provider == Provider.OPENROUTER:
                api_provider = APIProvider.OPENROUTER
            
            if api_provider:
                self.models[model_id] = ModelConfig(
                    model_id=model_id,
                    display_name=config.display_name,
                    provider=api_provider,
                    emoji=config.indicator_emoji,
                    description=config.description,
                    system_message=config.system_message or "",
                    openrouter_key=config.openrouter_model_key,
                    type=config.type,
                )

    def get_models_by_category(self) -> Dict[str, Dict[str, Any]]:
        """Get models organized by category/provider for hierarchical selection"""
        categories = {
            "gemini": {"name": "🧠 Gemini Models", "emoji": "🧠", "models": {}},
            "deepseek": {"name": "🔮 DeepSeek Models", "emoji": "🔮", "models": {}},
            "meta_llama": {"name": "🦙 Meta Llama Models", "emoji": "🦙", "models": {}},
            "qwen": {"name": "🌟 Qwen Models", "emoji": "🌟", "models": {}},
            "microsoft": {"name": "🔬 Microsoft Models", "emoji": "🔬", "models": {}},
            "mistral": {"name": "🌊 Mistral Models", "emoji": "🌊", "models": {}},
            "gemma": {"name": "💎 Google Gemma", "emoji": "💎", "models": {}},
            "nvidia": {"name": "⚡ NVIDIA Models", "emoji": "⚡", "models": {}},
            "thudm": {"name": "🔥 THUDM Models", "emoji": "🔥", "models": {}},
            "coding": {"name": "💻 Coding Specialists", "emoji": "💻", "models": {}},
            "vision": {"name": "👁️ Vision Models", "emoji": "👁️", "models": {}},
            "reasoning": {"name": "🧠 Reasoning Models", "emoji": "🧠", "models": {}},
            "creative": {"name": "🎭 Creative & Specialized", "emoji": "🎭", "models": {}},
        }        # Categorize models based on their types and providers
        for model_id, config in self.models.items():
            model_name = config.display_name.lower()
            model_type = getattr(config, 'type', 'general_purpose')
            
            # Primary categorization by provider for specific providers
            if config.provider == APIProvider.GEMINI:
                categories["gemini"]["models"][model_id] = config
            elif config.provider == APIProvider.DEEPSEEK:
                categories["deepseek"]["models"][model_id] = config
            # Use type-based categorization for OpenRouter models
            elif model_type == "reasoning" or "deepseek" in model_name or "r1" in model_name:
                categories["reasoning"]["models"][model_id] = config
            elif model_type in ["vision", "multimodal"] or any(x in model_name for x in ["vision", "visual", "vl", "image"]):
                categories["vision"]["models"][model_id] = config
            elif model_type in ["coding_specialist", "mathematical_reasoning"] or any(x in model_name for x in ["code", "coder", "programming", "olympic"]):
                categories["coding"]["models"][model_id] = config
            elif "llama" in model_name:
                categories["meta_llama"]["models"][model_id] = config
            elif "qwen" in model_name:
                categories["qwen"]["models"][model_id] = config
            elif "phi" in model_name or "mai" in model_name:
                categories["microsoft"]["models"][model_id] = config
            elif "mistral" in model_name or "mixtral" in model_name:
                categories["mistral"]["models"][model_id] = config
            elif "gemma" in model_name:
                categories["gemma"]["models"][model_id] = config
            elif "nemotron" in model_name:
                categories["nvidia"]["models"][model_id] = config
            elif "glm" in model_name:
                categories["thudm"]["models"][model_id] = config
            else:
                categories["creative"]["models"][model_id] = config
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v["models"]}

    async def chat(
        self,
        model_id: str,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 32000,
        quoted_message: Optional[str] = None,
    ) -> str:
        """🎯 Universal chat method - works with any model!"""
        
        # Get model config
        model_config = self.models.get(model_id)
        if not model_config:
            return f"❌ Model '{model_id}' not found!"        # Get the appropriate API
        api = self.apis.get(model_config.provider)
        if not api:
            return f"❌ API for {model_config.provider.value} not available!"

        # Use dynamic token allocation if max_tokens is default
        if max_tokens == 32000:  # Default value
            max_tokens = self._determine_optimal_tokens(prompt, model_config)

        try:
            # Add system message to context if provided
            if model_config.system_message and context:
                context = [{"role": "system", "content": model_config.system_message}] + context

            # Determine optimal max_tokens dynamically
            max_tokens = self._determine_optimal_tokens(prompt, model_config)

            # Route to appropriate API
            if model_config.provider == APIProvider.GEMINI:
                return await self._call_gemini(api, prompt, context)
            elif model_config.provider == APIProvider.DEEPSEEK:
                return await self._call_deepseek(api, prompt, context, model_config, temperature, max_tokens)
            elif model_config.provider == APIProvider.OPENROUTER:
                return await self._call_openrouter(api, prompt, context, model_config, temperature, max_tokens)
            else:
                return f"❌ Unsupported provider: {model_config.provider.value}"

        except Exception as e:
            self.logger.error(f"Error with {model_id}: {e}")
            return f"❌ Error: {str(e)}"

    def _determine_optimal_tokens(self, prompt: str, model_config: ModelConfig) -> int:
        """Determine optimal max_tokens based on prompt and model capabilities"""
        prompt_length = len(prompt)
        
        # Long form indicators that suggest need for more tokens
        long_form_indicators = [
            "write a", "generate", "create", "explain in detail", "step by step",
            "tutorial", "guide", "comprehensive", "list", "examples", "detailed",
            "analysis", "comparison", "pros and cons", "advantages", "disadvantages",
            "100", "q&a", "qcm", "questions", "document", "essay", "article"
        ]
        
        is_long_form = any(indicator in prompt.lower() for indicator in long_form_indicators)
        
        # Base token allocation
        if is_long_form or prompt_length > 500:
            return min(32000, model_config.max_tokens)  # Use full capacity for long requests
        elif prompt_length > 200:
            return min(16000, model_config.max_tokens)  # Medium requests
        else:
            return min(8000, model_config.max_tokens)   # Short requests

    async def _call_gemini(
        self, api: GeminiAPI, prompt: str, context: Optional[List]
    ) -> str:
        """Call Gemini API"""
        # Gemini uses its own context format
        return await api.generate_response(prompt, context)

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
        # DeepSeek uses messages format
        messages = context or []
        messages.append({"role": "user", "content": prompt})        
        return await api.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
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
        # Use the specific OpenRouter model key
        model_key = model_config.openrouter_key or model_config.model_id
        
        # OpenRouter API expects prompt and context separately
        return await api.generate_response(
            prompt=prompt,
            context=context,
            model=model_key,
            temperature=temperature,
            max_tokens=max_tokens
        )

    # 🎯 SIMPLE HELPER METHODS

    def get_all_models(self) -> Dict[str, ModelConfig]:
        """Get all available models"""
        return self.models

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get config for a specific model"""
        return self.models.get(model_id)

    def get_models_by_provider(self, provider: APIProvider) -> Dict[str, ModelConfig]:
        """Get all models for a specific provider"""
        return {k: v for k, v in self.models.items() if v.provider == provider}

    def get_model_display(self, model_id: str) -> str:
        """Get display name for a model"""
        config = self.models.get(model_id)
        return f"{config.emoji} {config.display_name}" if config else model_id

    def list_available_models(self) -> str:
        """Get a formatted string of all available models"""
        lines = ["🤖 **Available Models:**\n"]
        
        for provider in APIProvider:
            provider_models = self.get_models_by_provider(provider)
            if provider_models:
                lines.append(f"**{provider.value.title()} Models:**")
                for model_id, config in provider_models.items():
                    lines.append(f"• {config.emoji} {config.display_name}")
                lines.append("")
        
        return "\n".join(lines)

    def add_model(self, model_config: ModelConfig) -> None:
        """Add a new model configuration"""
        self.models[model_config.model_id] = model_config


# 🎯 CONVENIENCE ALIAS - Use this in your code
UnifiedAPIManager = SuperSimpleAPIManager
