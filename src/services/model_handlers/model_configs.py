"""
Model Configuration System
Defines all available models in a centralized configuration.
"""
from dataclasses import dataclass
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
    max_tokens: int = 4000
    default_temperature: float = 0.7
    supports_images: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_documents: bool = False
    description: str = ""


class ModelConfigurations:
    """Central configuration for all available models."""

    @staticmethod
    def get_all_models() -> Dict[str, ModelConfig]:
        """Get all available model configurations."""
        return {
            # Gemini Models
            "gemini": ModelConfig(
                model_id="gemini",
                display_name="Gemini 2.0 Flash",
                provider=Provider.GEMINI,
                indicator_emoji="✨",
                system_message="You are Gemini, a helpful AI assistant created by Google. Be concise, helpful, and accurate.",
                supports_images=True,
                supports_documents=True,
                description="Google's latest multimodal AI model"
            ),

            # DeepSeek Models  
            "deepseek": ModelConfig(
                model_id="deepseek",
                display_name="DeepSeek R1",
                provider=Provider.DEEPSEEK,
                indicator_emoji="🧠",
                system_message="You are DeepSeek, an advanced reasoning AI model that excels at complex problem-solving.",
                description="Advanced reasoning model with strong analytical capabilities"
            ),

            # OpenRouter Models - Free Models from our updated list
            "deepseek-r1-0528-qwen3-8b": ModelConfig(
                model_id="deepseek-r1-0528-qwen3-8b",
                display_name="DeepSeek R1 Qwen3 8B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-0528-qwen3-8b:free",
                indicator_emoji="🧠",
                system_message="You are DeepSeek R1, an advanced reasoning AI with Qwen3 architecture.",
                description="Latest DeepSeek R1 model with Qwen3 base - Free"
            ),

            "deepseek-r1-zero": ModelConfig(
                model_id="deepseek-r1-zero",
                display_name="DeepSeek R1 Zero",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-zero:free",
                indicator_emoji="🔬",
                system_message="You are DeepSeek R1 Zero, a model trained via large-scale reinforcement learning.",
                description="RL-trained reasoning model - Free"
            ),

            "deepcoder": ModelConfig(
                model_id="deepcoder",
                display_name="DeepCoder 14B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="agentica-org/deepcoder-14b-preview:free",
                indicator_emoji="💻",
                system_message="You are DeepCoder, an AI specialized in programming and software development.",
                description="Code generation specialist - Free"
            ),

            "llama4_maverick": ModelConfig(
                model_id="llama4_maverick",
                display_name="Llama-4 Maverick",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-4-maverick:free",
                indicator_emoji="🦙",
                system_message="You are Llama-4 Maverick, an advanced AI assistant by Meta with enhanced capabilities.",
                description="Meta's latest Llama 4 model - Free"
            ),

            "llama-3.2-11b-vision": ModelConfig(
                model_id="llama-3.2-11b-vision",
                display_name="Llama 3.2 11B Vision",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-3.2-11b-vision-instruct:free",
                indicator_emoji="👁️",
                supports_images=True,
                system_message="You are Llama 3.2 Vision, capable of understanding both text and images.",
                description="Vision-capable Llama model - Free"
            ),

            "qwen3-32b": ModelConfig(
                model_id="qwen3-32b",
                display_name="Qwen3 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-32b-a3b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a multilingual AI assistant created by Alibaba Cloud.",
                description="Large parameter Qwen model - Free"
            ),

            "mistral-small": ModelConfig(
                model_id="mistral-small",
                display_name="Mistral Small 3",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/mistral-small-24b-instruct-2501:free",
                indicator_emoji="🌊",
                system_message="You are Mistral, a high-performance European AI language model.",
                description="Latest Mistral small model - Free"
            ),

            "gemma-2-9b": ModelConfig(
                model_id="gemma-2-9b",
                display_name="Gemma 2 9B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-2-9b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma, a lightweight and efficient AI assistant by Google.",
                description="Google's efficient Gemma model - Free"
            ),

            "phi-4-reasoning": ModelConfig(
                model_id="phi-4-reasoning",
                display_name="Phi-4 Reasoning",
                provider=Provider.OPENROUTER,
                openrouter_model_key="microsoft/phi-4-reasoning-plus:free",
                indicator_emoji="🔬",
                system_message="You are Phi-4, a compact and efficient AI model by Microsoft, specialized in reasoning.",
                description="Microsoft's reasoning-focused model - Free"
            ),

            "olympiccoder-32b": ModelConfig(
                model_id="olympiccoder-32b",
                display_name="OlympicCoder 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="olympiccoder/olympiccoder-32b:free",
                indicator_emoji="🏆",
                system_message="You are OlympicCoder, an AI specialized in competitive programming and complex algorithms.",
                description="Competitive programming specialist - Free"
            ),

            # Additional OpenRouter free models can be easily added here...
        }

    @staticmethod
    def get_models_by_provider(provider: Provider) -> Dict[str, ModelConfig]:
        """Get all models for a specific provider."""
        all_models = ModelConfigurations.get_all_models()
        return {k: v for k, v in all_models.items() if v.provider == provider}

    @staticmethod
    def get_free_models() -> Dict[str, ModelConfig]:
        """Get all free models (OpenRouter models with :free suffix)."""
        all_models = ModelConfigurations.get_all_models()
        return {
            k: v for k, v in all_models.items() 
            if v.provider == Provider.OPENROUTER and 
            v.openrouter_model_key and ":free" in v.openrouter_model_key
        }

    @staticmethod
    def add_openrouter_models(additional_models: List[Dict[str, Any]]) -> None:
        """
        Easily add more OpenRouter models.
        
        Args:
            additional_models: List of model dictionaries with keys:
                - model_id, display_name, openrouter_model_key, indicator_emoji, etc.
        """
        current_models = ModelConfigurations.get_all_models()
        
        for model_data in additional_models:
            model_config = ModelConfig(
                model_id=model_data["model_id"],
                display_name=model_data["display_name"],
                provider=Provider.OPENROUTER,
                openrouter_model_key=model_data["openrouter_model_key"],
                indicator_emoji=model_data.get("indicator_emoji", "🤖"),
                system_message=model_data.get("system_message"),
                description=model_data.get("description", "")
            )
            current_models[model_data["model_id"]] = model_config
