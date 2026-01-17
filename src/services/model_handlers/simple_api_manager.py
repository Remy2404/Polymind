"""
🎯 SIMPLIFIED UNIFIED API MANAGEMENT SYSTEM 🎯
This single file replaces:
- factory.py
- gemini_handler.py
-             "reasoning": {"name": "Reasoning Models", "emoji": "🤔", "models": {}},eepseek_handler.py
- model_configs.py
- unified_handler.py
Benefits:
One file to manage all APIs
Easy to add new models/providers
Simplified switching between models
Reduced code complexity
Centralized configuration
"""

import logging
from typing import Dict, Optional, List, Any
from enum import Enum
from dataclasses import dataclass
from src.api.llm.gemini_api import GeminiAPI
from src.api.llm.openrouter_api import OpenRouterAPI
from src.api.llm.deepseek_api import DeepSeekLLM

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
    max_tokens: int = 128000
    type: str = "general_purpose"


PROVIDER_GROUPS = {
    "Gemini Models": {
        "provider": APIProvider.GEMINI,
        "description": "Google's Gemini AI models",
        "models": [],
    },
    "DeepSeek Models": {
        "provider": APIProvider.DEEPSEEK,
        "description": "DeepSeek reasoning models",
        "models": [],
    },
    "OpenRouter Models": {
        "provider": APIProvider.OPENROUTER,
        "description": "Multiple AI models via OpenRouter",
        "models": [],
    },
}


class SuperSimpleAPIManager:
    """
     SUPER SIMPLE API MANAGER 
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
        """ Configure all your models here - Easy to add new ones!"""
        from src.services.model_handlers.model_configs import (
            ModelConfigurations,
            Provider,
        )

        model_configs = ModelConfigurations.get_all_models()
        self.models: Dict[str, ModelConfig] = {}
        for model_id, config in model_configs.items():
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
            "gemini": {"name": "Gemini Models", "emoji": "✨", "models": {}},
            "deepseek": {"name": "DeepSeek Models", "emoji": "🧠", "models": {}},
            "meta_llama": {"name": "Meta Llama Models", "emoji": "🦙", "models": {}},
            "qwen": {"name": "Qwen Models", "emoji": "🌟", "models": {}},
            "microsoft": {"name": "Microsoft Models", "emoji": "🔬", "models": {}},
            "mistral": {"name": "Mistral Models", "emoji": "🌊", "models": {}},
            "gemma": {"name": "💎 Google Gemma", "emoji": "💎", "models": {}},
            "nvidia": {"name": "NVIDIA Models", "emoji": "⚡", "models": {}},
            "thudm": {"name": "THUDM Models", "emoji": "🔥", "models": {}},
            "coding": {"name": "Coding Specialists", "emoji": "💻", "models": {}},
            "vision": {"name": "Vision Models", "emoji": "👁️", "models": {}},
            "reasoning": {"name": "� Reasoning Models", "emoji": "�", "models": {}},
            "creative": {
                "name": "Creative & Specialized",
                "emoji": "🎭",
                "models": {},
            },
        }
        for model_id, config in self.models.items():
            model_name = config.display_name.lower()
            model_type = getattr(config, "type", "general_purpose")
            if config.provider == APIProvider.GEMINI:
                categories["gemini"]["models"][model_id] = config
            elif config.provider == APIProvider.DEEPSEEK:
                categories["deepseek"]["models"][model_id] = config
            elif (
                model_type == "reasoning"
                or "deepseek" in model_name
                or "r1" in model_name
            ):
                categories["reasoning"]["models"][model_id] = config
            elif model_type in ["vision", "multimodal"] or any(
                x in model_name for x in ["vision", "visual", "vl", "image"]
            ):
                categories["vision"]["models"][model_id] = config
            elif model_type in ["coding_specialist", "mathematical_reasoning"] or any(
                x in model_name for x in ["code", "coder", "programming", "olympic"]
            ):
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
        return {k: v for k, v in categories.items() if v["models"]}

    async def chat(
        self,
        model_id: str,
        prompt: str,
        context: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        quoted_message: Optional[str] = None,
    ) -> str:
        """🎯 Universal chat method - works with any model!"""
        model_config = self.models.get(model_id)
        if not model_config:
            return f"❌ Model '{model_id}' not found!"
        api = self.apis.get(model_config.provider)
        if not api:
            return f"❌ API for {model_config.provider.value} not available!"
        try:
            if model_config.system_message and context:
                context = [
                    {"role": "system", "content": model_config.system_message}
                ] + context
            
            if max_tokens is None:
                max_tokens = self._determine_optimal_tokens(prompt, model_config)
            else:
                max_tokens = min(max_tokens, model_config.max_tokens)
            if model_config.provider == APIProvider.GEMINI:
                return await self._call_gemini(api, prompt, context)
            elif model_config.provider == APIProvider.DEEPSEEK:
                return await self._call_deepseek(
                    api, prompt, context, model_config, temperature, max_tokens
                )
            elif model_config.provider == APIProvider.OPENROUTER:
                return await self._call_openrouter(
                    api, prompt, context, model_config, temperature, max_tokens
                )
            else:
                return f"❌ Unsupported provider: {model_config.provider.value}"
        except Exception as e:
            self.logger.error(f"Error with {model_id}: {e}")
            return f"❌ Error: {str(e)}"

    def _determine_optimal_tokens(self, prompt: str, model_config: ModelConfig) -> int:
        """Determine optimal max_tokens based on prompt and model capabilities"""
        from src.services.model_handlers.model_configs import ModelConfigurations
        
        model_configs = ModelConfigurations.get_all_models()
        full_config = model_configs.get(model_config.model_id)
        
        if full_config and hasattr(full_config, 'max_tokens'):
            return full_config.max_tokens
        
        return model_config.max_tokens if hasattr(model_config, 'max_tokens') else 128000

    async def _call_gemini(
        self, api: GeminiAPI, prompt: str, context: Optional[List]
    ) -> str:
        """Call Gemini API"""
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
        messages = context or []
        messages.append({"role": "user", "content": prompt})
        return await api.generate_response(
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
        model_key = model_config.openrouter_key or model_config.model_id
        return await api.generate_response(
            prompt=prompt,
            context=context,
            model=model_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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


UnifiedAPIManager = SuperSimpleAPIManager
