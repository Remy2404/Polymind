"""
Model Configuration System
Defines all available models in a centralized configuration.
"""

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


class ModelConfigurations:
    """Central configuration for all available models."""

    @staticmethod
    def get_all_models() -> Dict[str, ModelConfig]:
        """Get all available model configurations."""
        models = {
            # Gemini Models
            "gemini": ModelConfig(
                model_id="gemini",
                display_name="Gemini 2.0 Flash",
                provider=Provider.GEMINI,
                indicator_emoji="✨",
                system_message="You are Gemini, a helpful AI assistant created by Google. Be concise, helpful, and accurate.",
                supports_images=True,
                supports_documents=True,
                description="Google's latest multimodal AI model",
                type="multimodal",
                capabilities=[
                    "supports_images",
                    "supports_documents",
                    "long_context",
                    "general_purpose",
                ],
            ),
            # DeepSeek Models
            "deepseek": ModelConfig(
                model_id="deepseek",
                display_name="DeepSeek R1",
                provider=Provider.DEEPSEEK,
                indicator_emoji="🧠",
                system_message="You are DeepSeek, an advanced reasoning AI model that excels at complex problem-solving.",
                description="Advanced reasoning model with strong analytical capabilities",
                type="reasoning",
                capabilities=["reasoning_capable", "long_context", "general_purpose"],
            ),
            # === DEEPSEEK MODELS ===
            "deepseek-r1-qwen3-8b": ModelConfig(
                model_id="deepseek-r1-qwen3-8b",
                display_name="DeepSeek R1 Qwen3 8B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-0528-qwen3-8b:free",
                indicator_emoji="🧠",
                system_message="You are DeepSeek R1, an advanced reasoning AI with Qwen3 architecture.",
                description="Latest DeepSeek R1 model with Qwen3 base - Free",
                type="reasoning",
                capabilities=["reasoning_capable", "multilingual_support"],
            ),
            "deepseek-r1-0528": ModelConfig(
                model_id="deepseek-r1-0528",
                display_name="DeepSeek R1 0528",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-0528:free",
                indicator_emoji="🧠",
                system_message="You are DeepSeek R1, an advanced reasoning AI model.",
                description="DeepSeek R1 May 2024 version - Free",
                type="reasoning",
                capabilities=["reasoning_capable"],
            ),
            "deepseek-r1-zero": ModelConfig(
                model_id="deepseek-r1-zero",
                display_name="DeepSeek R1 Zero",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-distill-llama-70b:free",
                indicator_emoji="🔬",
                system_message="You are DeepSeek R1, a model trained via large-scale reinforcement learning.",
                description="RL-trained reasoning model - Free",
                type="reasoning",
                capabilities=["reasoning_capable"],
            ),
            "deepseek-prover-v2": ModelConfig(
                model_id="deepseek-prover-v2",
                display_name="DeepSeek Prover V2",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-prover-v2:free",
                indicator_emoji="🧮",
                system_message="You are DeepSeek Prover, specialized in mathematical reasoning and formal proofs.",
                description="Mathematical proof specialist - Free",
            ),
            "deepseek-v3-base": ModelConfig(
                model_id="deepseek-v3-base",
                display_name="DeepSeek V3 Base",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-v3-base:free",
                indicator_emoji="🧠",
                system_message="You are DeepSeek V3, a powerful foundation model for various AI tasks.",
                description="DeepSeek V3 foundation model - Free",
            ),
            "deepseek-chat-v3": ModelConfig(
                model_id="deepseek-chat-v3",
                display_name="DeepSeek Chat V3",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-chat-v3-0324:free",
                indicator_emoji="💬",
                system_message="You are DeepSeek Chat, optimized for conversational AI interactions.",
                description="Conversational DeepSeek model - Free",
            ),
            "deepseek-r1-distill-qwen-32b": ModelConfig(
                model_id="deepseek-r1-distill-qwen-32b",
                display_name="DeepSeek R1 Distill Qwen 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-distill-qwen-32b:free",
                indicator_emoji="🧠",
                system_message="You are DeepSeek R1 Distilled, combining reasoning capabilities with Qwen architecture.",
                description="Distilled reasoning model 32B - Free",
            ),
            "deepseek-r1-distill-qwen-14b": ModelConfig(
                model_id="deepseek-r1-distill-qwen-14b",
                display_name="DeepSeek R1 Distill Qwen 14B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-distill-qwen-14b:free",
                indicator_emoji="🧠",
                system_message="You are DeepSeek R1 Distilled, combining reasoning capabilities with Qwen architecture.",
                description="Distilled reasoning model 14B - Free",
            ),
            # === LLAMA MAVERICK ===
            "llama4-maverick": ModelConfig(
                model_id="llama4-maverick",
                display_name="Llama 4 Maverick",
                provider=Provider.OPENROUTER,
                openrouter_model_key="152334h/llama4-maverick-16b:free",
                indicator_emoji="🦙",
                system_message="You are Llama-4 Maverick, Meta's latest advanced AI assistant with enhanced capabilities.",
                description="Meta's latest Llama 4 model - Free",
            ),
            "llama4-scout": ModelConfig(
                model_id="llama4-scout",
                display_name="Llama-4 Scout",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-4-scout:free",
                indicator_emoji="🔍",
                system_message="You are Llama-4 Scout, specialized in exploration and analysis tasks.",
                description="Specialized exploration model - Free",
            ),
            "llama-3.3-8b": ModelConfig(
                model_id="llama-3.3-8b",
                display_name="Llama 3.3 8B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-3.3-8b-instruct:free",
                indicator_emoji="🦙",
                system_message="You are Llama 3.3, an advanced AI assistant by Meta.",
                description="Latest Llama 3.3 8B model - Free",
            ),
            # === QWEN MODELS ===
            "qwen3-235b": ModelConfig(
                model_id="qwen3-235b",
                display_name="Qwen3 235B A22B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-235b-a22b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a large-scale multilingual AI assistant created by Alibaba Cloud.",
                description="Massive 235B parameter Qwen model - Free",
            ),
            "qwen3-32b": ModelConfig(
                model_id="qwen3-32b",
                display_name="Qwen3 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-32b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a multilingual AI assistant created by Alibaba Cloud.",
                description="Large 32B parameter Qwen model - Free",
            ),
            "qwen3-30b": ModelConfig(
                model_id="qwen3-30b",
                display_name="Qwen3 30B A3B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-30b-a3b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a multilingual AI assistant created by Alibaba Cloud.",
                description="30B parameter Qwen model - Free",
            ),
            "qwen3-14b": ModelConfig(
                model_id="qwen3-14b",
                display_name="Qwen3 14B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-14b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a multilingual AI assistant created by Alibaba Cloud.",
                description="14B parameter Qwen model - Free",
            ),
            "qwen3-8b": ModelConfig(
                model_id="qwen3-8b",
                display_name="Qwen3 8B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-8b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a multilingual AI assistant created by Alibaba Cloud.",
                description="8B parameter Qwen model - Free",
            ),
            "qwq-32b": ModelConfig(
                model_id="qwq-32b",
                display_name="QwQ 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwq-32b:free",
                indicator_emoji="🤔",
                system_message="You are QwQ, a reasoning-focused AI model that thinks step by step.",
                description="Question-answering reasoning model - Free",
            ),
            "qwen2.5-vl-72b": ModelConfig(
                model_id="qwen2.5-vl-72b",
                display_name="Qwen2.5 VL 72B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen2.5-vl-72b-instruct:free",
                indicator_emoji="👁️",
                supports_images=True,
                system_message="You are Qwen2.5 Vision-Language, capable of understanding text and images.",
                description="Large vision-language model - Free",
            ),
            "qwen2.5-vl-32b": ModelConfig(
                model_id="qwen2.5-vl-32b",
                display_name="Qwen2.5 VL 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen2.5-vl-32b-instruct:free",
                indicator_emoji="👁️",
                supports_images=True,
                system_message="You are Qwen2.5 Vision-Language, capable of understanding text and images.",
                description="Vision-language model 32B - Free",
            ),
            "qwen2.5-vl-3b": ModelConfig(
                model_id="qwen2.5-vl-3b",
                display_name="Qwen2.5 VL 3B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen2.5-vl-3b-instruct:free",
                indicator_emoji="👁️",
                supports_images=True,
                system_message="You are Qwen2.5 Vision-Language, capable of understanding text and images.",
                description="Compact vision-language model - Free",
            ),
            "qwen-2.5-72b-instruct": ModelConfig(
                model_id="qwen-2.5-72b-instruct",
                display_name="Qwen 2.5 72B Instruct",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen-2.5-72b-instruct:free",
                indicator_emoji="🌟",
                system_message="You are Qwen 2.5, the latest series of Qwen large language models.",
                description="Latest 72B Qwen instruction model - Free",
                type="general_purpose",
                capabilities=["multilingual_support", "long_context"],
            ),
            # === MICROSOFT PHI ===
            "phi-4-reasoning-plus": ModelConfig(
                model_id="phi-4-reasoning-plus",
                display_name="Phi 4 Reasoning+",
                provider=Provider.OPENROUTER,
                openrouter_model_key="microsoft/phi-4-reasoning-plus:free",
                indicator_emoji="🔬",
                system_message="You are Phi-4 Reasoning Plus, Microsoft's advanced reasoning model.",
                description="Enhanced reasoning capabilities - Free",
            ),
            "mai-ds-r1": ModelConfig(
                model_id="mai-ds-r1",
                display_name="MAI DS R1",
                provider=Provider.OPENROUTER,
                openrouter_model_key="microsoft/mai-ds-r1:free",
                indicator_emoji="🤖",
                system_message="You are MAI-DS-R1, a post-trained variant of DeepSeek-R1 developed by the Microsoft AI team to improve the model’s responsiveness on previously blocked topics while enhancing its safety profile.",
                description="Microsoft's post-trained DeepSeek-R1 variant - Free",
                type="reasoning",
                capabilities=["reasoning_capable"],
            ),
            "phi-4-reasoning": ModelConfig(
                model_id="phi-4-reasoning",
                display_name="Phi-4 Reasoning",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-coder-v2-lite:free",
                indicator_emoji="💻",
                system_message="You are DeepCoder, an expert AI programming assistant.",
                description="Expert coding assistant from DeepSeek - Free",
                type="coding_specialist",
                capabilities=["coding_specialist"],
            ),
            "olympiccoder-32b": ModelConfig(
                model_id="olympiccoder-32b",
                display_name="Olympic Coder 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="kostaleon/olympicoder-32b:free",
                indicator_emoji="🏆",
                system_message="You are Olympic Coder, a competitive programming AI.",
                description="Competitive programming specialist - Free",
                type="coding_specialist",
                capabilities=["coding_specialist"],
            ),
            "devstral-small": ModelConfig(
                model_id="devstral-small",
                display_name="Devstral Small",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/codestral-mamba-7b:free",
                indicator_emoji="🐹",
                system_message="You are Devstral, a small but powerful coding assistant.",
                description="Small and efficient coding model - Free",
                type="coding_specialist",
                capabilities=["coding_specialist"],
            ),
            # === MATHEMATICAL SPECIALISTS ===
            "qwq-32b": ModelConfig(
                model_id="qwq-32b",
                display_name="QWQ 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwq/qwq-32b:free",
                indicator_emoji="🧮",
                system_message="You are QWQ, a model with strong mathematical abilities.",
                description="Mathematical reasoning model - Free",
                type="mathematical_reasoning",
                capabilities=["mathematical_reasoning"],
            ),
            # === CREATIVE WRITING ===
            "deephermes-3-mistral-24b": ModelConfig(
                model_id="deephermes-3-mistral-24b",
                display_name="DeepHermes 3 Mistral 24B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="cognitivecomputations/deephermes-3-mistral-24b:free",
                indicator_emoji="✍️",
                system_message="You are DeepHermes, a creative writing AI based on Mistral.",
                description="Creative writing specialist - Free",
                type="creative_writing",
                capabilities=["creative_writing"],
            ),
            "qwerky-72b": ModelConfig(
                model_id="qwerky-72b",
                display_name="Qwerky 72B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3-4b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3, Google's lightweight AI assistant.",
                description="Compact Gemma 3 model - Free",
            ),
            "gemma-3-1b": ModelConfig(
                model_id="gemma-3-1b",
                display_name="Gemma 3 1B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="cognitivecomputations/moonlight-16b:free",
                indicator_emoji="🌙",
                system_message="You are Moonlight, an AI that specializes in storytelling and narrative.",
                description="Storytelling and narrative specialist - Free",
                type="creative_writing",
                capabilities=["creative_writing"],
            ),
            # === MULTILINGUAL MODELS ===
            "qwen3-235b": ModelConfig(
                model_id="qwen3-235b",
                display_name="Qwen3 235B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-235b:free",
                indicator_emoji="🌏",
                system_message="You are Qwen3, a powerful multilingual AI.",
                description="Powerful multilingual model - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
            ),
            "shisa-v2-llama3.3-70b": ModelConfig(
                model_id="shisa-v2-llama3.3-70b",
                display_name="Shisa V2 Llama3.3 70B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="shisa/shisa-v2-llama3.3-70b:free",
                indicator_emoji="🦁",
                system_message="You are Shisa, a Japanese-focused multilingual model.",
                description="Japanese-focused multilingual model - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
            ),
            "sarvam-m": ModelConfig(
                model_id="sarvam-m",
                display_name="Sarvam-M",
                provider=Provider.OPENROUTER,
                openrouter_model_key="sarvam.ai/sarvam-m:free",
                indicator_emoji="🇮🇳",
                system_message="You are Sarvam-M, an Indian language AI model.",
                description="Indian language specialist - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
            ),
            "glm-z1-32b": ModelConfig(
                model_id="glm-z1-32b",
                display_name="GLM-Z1 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="zhipu/glm-z1-32b:free",
                indicator_emoji="🇨🇳",
                system_message="You are GLM-Z1, a Chinese-English bilingual model.",
                description="Chinese-English bilingual model - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
            ),
            # === VISION MODELS ===
            "llama-3.2-11b-vision": ModelConfig(
                model_id="llama-3.2-11b-vision",
                display_name="Llama 3.2 11B Vision",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-3.2-11b-vision:free",
                indicator_emoji="👁️",
                system_message="You are Llama 3.2 Vision, a multimodal AI that can understand images.",
                description="Llama 3.2 with vision capabilities - Free",
                supports_images=True,
                type="vision",
                capabilities=["supports_images"],
            ),
            "qwen2.5-vl-72b": ModelConfig(
                model_id="qwen2.5-vl-72b",
                display_name="Qwen2.5 VL 72B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen2.5-vl-72b:free",
                indicator_emoji="👁️",
                system_message="You are Qwen2.5 VL, a powerful vision-language model.",
                description="Powerful vision-language model - Free",
                supports_images=True,
                type="vision",
                capabilities=["supports_images"],
            ),
            "internvl3-14b": ModelConfig(
                model_id="internvl3-14b",
                display_name="InternVL3 14B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="openbmb/internvl3-14b:free",
                indicator_emoji="👁️",
                system_message="You are InternVL3, a vision-language model for complex visual tasks.",
                description="Vision-language model for complex tasks - Free",
                supports_images=True,
                type="vision",
                capabilities=["supports_images"],
            ),
            "kimi-vl-a3b-thinking": ModelConfig(
                model_id="kimi-vl-a3b-thinking",
                display_name="Kimi VL A3B Thinking",
                provider=Provider.OPENROUTER,
                openrouter_model_key="01-ai/kimi-vl-a3b-thinking:free",
                indicator_emoji="🧠",
                system_message="You are Kimi VL, a vision-language model with enhanced reasoning.",
                description="Vision-language model with reasoning - Free",
                supports_images=True,
                type="vision",
                capabilities=["supports_images", "reasoning_capable"],
            ),
            # === MISTRAL MODELS ===
            "mistral-small-3.2-24b-instruct": ModelConfig(
                model_id="mistral-small-3.2-24b-instruct",
                display_name="Mistral Small 3.2 24B Instruct",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/mistral-small-3.2-24b-instruct:free",
                indicator_emoji="🌬️",
                system_message="You are Mistral Small, a powerful and efficient AI assistant by Mistral AI.",
                description="Mistral's latest small, powerful, and efficient model - Free",
                type="general_purpose",
                capabilities=["general_purpose", "multilingual_support"],
            ),
        }
        return models

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
                description=model_data.get("description", ""),
            )
            current_models[model_data["model_id"]] = model_config
