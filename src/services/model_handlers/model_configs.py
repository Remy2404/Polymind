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
            # === OPENROUTER MODELS ===
            # OpenRouter Cypher
            "cypher-alpha": ModelConfig(
                model_id="cypher-alpha",
                display_name="Cypher Alpha",
                provider=Provider.OPENROUTER,
                openrouter_model_key="openrouter/cypher-alpha:free",
                indicator_emoji="🔐",
                system_message="You are Cypher Alpha, an advanced AI model focused on security and analysis.",
                description="Security-focused AI model - Free",
            ),
            # Mistral Models
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
            "mistral-small-3.1-24b-instruct": ModelConfig(
                model_id="mistral-small-3.1-24b-instruct",
                display_name="Mistral Small 3.1 24B Instruct",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/mistral-small-3.1-24b-instruct:free",
                indicator_emoji="🌬️",
                system_message="You are Mistral Small 3.1, a powerful and efficient AI assistant.",
                description="Mistral Small 3.1 model - Free",
                type="general_purpose",
                capabilities=["general_purpose", "multilingual_support"],
            ),
            "mistral-small-24b-instruct-2501": ModelConfig(
                model_id="mistral-small-24b-instruct-2501",
                display_name="Mistral Small 24B Instruct 2501",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/mistral-small-24b-instruct-2501:free",
                indicator_emoji="🌬️",
                system_message="You are Mistral Small 2501, the latest efficient AI assistant.",
                description="Latest Mistral Small 2501 model - Free",
                type="general_purpose",
                capabilities=["general_purpose", "multilingual_support"],
            ),
            "mistral-nemo": ModelConfig(
                model_id="mistral-nemo",
                display_name="Mistral Nemo",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/mistral-nemo:free",
                indicator_emoji="🌬️",
                system_message="You are Mistral Nemo, a compact and efficient AI assistant.",
                description="Compact Mistral model - Free",
            ),
            "mistral-7b-instruct": ModelConfig(
                model_id="mistral-7b-instruct",
                display_name="Mistral 7B Instruct",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/mistral-7b-instruct:free",
                indicator_emoji="🌬️",
                system_message="You are Mistral 7B, a powerful and compact AI assistant.",
                description="Classic Mistral 7B model - Free",
            ),
            "devstral-small": ModelConfig(
                model_id="devstral-small",
                display_name="Devstral Small",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/devstral-small:free",
                indicator_emoji="🐹",
                system_message="You are Devstral, a small but powerful coding assistant.",
                description="Small and efficient coding model - Free",
                type="coding_specialist",
                capabilities=["coding_specialist"],
            ),
            # Moonshot AI Models
            "kimi": ModelConfig(
                model_id="kimi",
                display_name="Kimi",
                provider=Provider.OPENROUTER,
                openrouter_model_key="moonshotai/kimi-k2:free",
                indicator_emoji="🌙",
                system_message="You are Kimi, a helpful AI assistant.",
                description="General purpose AI model by Moonshot AI - Free",
                type="general_purpose",
            ),
            "kimi-dev-72b": ModelConfig(
                model_id="kimi-dev-72b",
                display_name="Kimi Dev 72B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="moonshotai/kimi-k2:free",
                indicator_emoji="🌙",
                system_message="You are Kimi Dev, a powerful development-focused AI model.",
                description="Development-focused AI model by Moonshot AI - Free",
                type="coding_specialist",
                capabilities=["coding_specialist"],
            ),
            "kimi-vl-a3b-thinking": ModelConfig(
                model_id="kimi-vl-a3b-thinking",
                display_name="Kimi VL A3B Thinking",
                provider=Provider.OPENROUTER,
                openrouter_model_key="moonshotai/kimi-vl-a3b-thinking:free",
                indicator_emoji="🧠",
                system_message="You are Kimi VL, a vision-language model with enhanced reasoning.",
                description="Vision-language model with reasoning - Free",
                supports_images=True,
                type="vision",
                capabilities=["supports_images", "reasoning_capable"],
            ),
            # DeepSeek Models
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
            "deepseek-r1": ModelConfig(
                model_id="deepseek-r1",
                display_name="DeepSeek R1",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1:free",
                indicator_emoji="🧠",
                system_message="You are DeepSeek R1, an advanced reasoning AI model.",
                description="Latest DeepSeek R1 model - Free",
                type="reasoning",
                capabilities=["reasoning_capable"],
            ),
            "deepseek-chat": ModelConfig(
                model_id="deepseek-chat",
                display_name="DeepSeek Chat",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-chat:free",
                indicator_emoji="💬",
                system_message="You are DeepSeek Chat, optimized for conversational AI interactions.",
                description="Conversational DeepSeek model - Free",
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
                openrouter_model_key="deepseek/deepseek-chat:free",
                indicator_emoji="💬",
                system_message="You are DeepSeek Chat, optimized for conversational AI interactions.",
                description="Conversational DeepSeek model - Free",
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
            "deepseek-r1-distill-llama-70b": ModelConfig(
                model_id="deepseek-r1-distill-llama-70b",
                display_name="DeepSeek R1 Distill Llama 70B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/r1-distill-llama-70b:free",
                indicator_emoji="🔬",
                system_message="You are DeepSeek R1, a model trained via large-scale reinforcement learning.",
                description="RL-trained reasoning model - Free",
                type="reasoning",
                capabilities=["reasoning_capable"],
            ),
            # Sarvam AI
            "sarvam-m": ModelConfig(
                model_id="sarvam-m",
                display_name="Sarvam-M",
                provider=Provider.OPENROUTER,
                openrouter_model_key="sarvamai/sarvam-m:free",
                indicator_emoji="🇮🇳",
                system_message="You are Sarvam-M, an Indian language AI model.",
                description="Indian language specialist - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
            ),
            # Google Models
            "gemma-3n-e4b": ModelConfig(
                model_id="gemma-3n-e4b",
                display_name="Gemma 3N E4B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3n-e4b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3N, Google's efficient AI assistant.",
                description="Efficient Gemma 3N model - Free",
            ),
            "gemma-3-27b": ModelConfig(
                model_id="gemma-3-27b",
                display_name="Gemma 3 27B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3-27b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3, Google's advanced lightweight AI assistant.",
                description="Large Gemma 3 model - Free",
            ),
            "gemma-3-12b": ModelConfig(
                model_id="gemma-3-12b",
                display_name="Gemma 3 12B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3-12b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3, Google's lightweight AI assistant.",
                description="Medium Gemma 3 model - Free",
            ),
            "gemma-3-4b": ModelConfig(
                model_id="gemma-3-4b",
                display_name="Gemma 3 4B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3-4b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3, Google's lightweight AI assistant.",
                description="Compact Gemma 3 model - Free",
            ),
            "gemma-2-9b": ModelConfig(
                model_id="gemma-2-9b",
                display_name="Gemma 2 9B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-2-9b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 2, Google's lightweight AI assistant.",
                description="Gemma 2 9B model - Free",
            ),
            "gemini-2.5-pro-exp": ModelConfig(
                model_id="gemini-2.5-pro-exp",
                display_name="Gemini 2.5 Pro Experimental",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemini-2.5-pro-exp-03-25",
                indicator_emoji="✨",
                system_message="You are Gemini 2.5 Pro Experimental, Google's latest experimental AI model.",
                description="Google's experimental Gemini 2.5 Pro - Paid",
                supports_images=True,
                type="multimodal",
                capabilities=["supports_images", "general_purpose"],
            ),
            "gemini-2.0-flash-exp": ModelConfig(
                model_id="gemini-2.0-flash-exp",
                display_name="Gemini 2.0 Flash Experimental",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemini-2.0-flash-exp:free",
                indicator_emoji="⚡",
                system_message="You are Gemini 2.0 Flash Experimental, Google's fast experimental AI model.",
                description="Google's experimental Gemini 2.0 Flash - Free",
                supports_images=True,
                type="multimodal",
                capabilities=["supports_images", "general_purpose"],
            ),
            # Qwen Models
            "qwen3-30b": ModelConfig(
                model_id="qwen3-30b",
                display_name="Qwen3 30B A3B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-30b-a3b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a multilingual AI assistant created by Alibaba Cloud.",
                description="30B parameter Qwen model - Free",
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
            "qwen3-14b": ModelConfig(
                model_id="qwen3-14b",
                display_name="Qwen3 14B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-14b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a multilingual AI assistant created by Alibaba Cloud.",
                description="14B parameter Qwen model - Free",
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
            "qwen3-235b": ModelConfig(
                model_id="qwen3-235b",
                display_name="Qwen3 235B A22B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen3-235b-a22b:free",
                indicator_emoji="🌟",
                system_message="You are Qwen3, a large-scale multilingual AI assistant created by Alibaba Cloud.",
                description="Massive 235B parameter Qwen model - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
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
            "qwen2.5-vl-72b": ModelConfig(
                model_id="qwen2.5-vl-72b",
                display_name="Qwen2.5 VL 72B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen2.5-vl-72b-instruct:free",
                indicator_emoji="👁️",
                supports_images=True,
                system_message="You are Qwen2.5 Vision-Language, capable of understanding text and images.",
                description="Large vision-language model - Free",
                type="vision",
                capabilities=["supports_images"],
            ),
            "qwen-2.5-coder-32b": ModelConfig(
                model_id="qwen-2.5-coder-32b",
                display_name="Qwen 2.5 Coder 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwen-2.5-coder-32b-instruct:free",
                indicator_emoji="💻",
                system_message="You are Qwen 2.5 Coder, specialized in programming and code generation.",
                description="Specialized coding model - Free",
                type="coding_specialist",
                capabilities=["coding_specialist"],
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
            "qwq-32b": ModelConfig(
                model_id="qwq-32b",
                display_name="QwQ 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="qwen/qwq-32b:free",
                indicator_emoji="🤔",
                system_message="You are QwQ, a reasoning-focused AI model that thinks step by step.",
                description="Question-answering reasoning model - Free",
                type="mathematical_reasoning",
                capabilities=["mathematical_reasoning"],
            ),
            # TNG Tech
            "deepseek-r1t-chimera": ModelConfig(
                model_id="deepseek-r1t-chimera",
                display_name="DeepSeek R1T Chimera",
                provider=Provider.OPENROUTER,
                openrouter_model_key="tngtech/deepseek-r1t-chimera:free",
                indicator_emoji="🧬",
                system_message="You are DeepSeek R1T Chimera, an enhanced reasoning model.",
                description="Enhanced DeepSeek reasoning variant - Free",
                type="reasoning",
                capabilities=["reasoning_capable"],
            ),
            # Microsoft
            "mai-ds-r1": ModelConfig(
                model_id="mai-ds-r1",
                display_name="MAI DS R1",
                provider=Provider.OPENROUTER,
                openrouter_model_key="microsoft/mai-ds-r1:free",
                indicator_emoji="🤖",
                system_message="You are MAI-DS-R1, a post-trained variant of DeepSeek-R1 developed by the Microsoft AI team to improve the model's responsiveness on previously blocked topics while enhancing its safety profile.",
                description="Microsoft's post-trained DeepSeek-R1 variant - Free",
                type="reasoning",
                capabilities=["reasoning_capable"],
            ),
            # THUDM
            "glm-z1-32b": ModelConfig(
                model_id="glm-z1-32b",
                display_name="GLM-Z1 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="thudm/glm-z1-32b:free",
                indicator_emoji="🇨🇳",
                system_message="You are GLM-Z1, a Chinese-English bilingual model.",
                description="Chinese-English bilingual model - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
            ),
            "glm-4-32b": ModelConfig(
                model_id="glm-4-32b",
                display_name="GLM-4 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="thudm/glm-4-32b:free",
                indicator_emoji="🇨🇳",
                system_message="You are GLM-4, a powerful Chinese-English bilingual model.",
                description="Powerful Chinese-English bilingual model - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
            ),
            # Shisa AI
            "shisa-v2-llama3.3-70b": ModelConfig(
                model_id="shisa-v2-llama3.3-70b",
                display_name="Shisa V2 Llama3.3 70B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="shisa-ai/shisa-v2-llama3.3-70b:free",
                indicator_emoji="🦁",
                system_message="You are Shisa, a Japanese-focused multilingual model.",
                description="Japanese-focused multilingual model - Free",
                type="multilingual",
                capabilities=["multilingual_support"],
            ),
            # ARLIAI
            "qwq-32b-arliai": ModelConfig(
                model_id="qwq-32b-arliai",
                display_name="QwQ 32B ARLIAI",
                provider=Provider.OPENROUTER,
                openrouter_model_key="arliai/qwq-32b-arliai-rpr-v1:free",
                indicator_emoji="🤔",
                system_message="You are QwQ ARLIAI, an enhanced reasoning-focused AI model.",
                description="Enhanced QwQ reasoning model by ARLIAI - Free",
                type="mathematical_reasoning",
                capabilities=["mathematical_reasoning"],
            ),
            # Agentica
            "deepcoder-14b": ModelConfig(
                model_id="deepcoder-14b",
                display_name="DeepCoder 14B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="agentica-org/deepcoder-14b-preview:free",
                indicator_emoji="💻",
                system_message="You are DeepCoder, an expert AI programming assistant.",
                description="Expert coding assistant - Free",
                type="coding_specialist",
                capabilities=["coding_specialist"],
            ),
            # NVIDIA
            "llama-3.3-nemotron-super-49b": ModelConfig(
                model_id="llama-3.3-nemotron-super-49b",
                display_name="Llama 3.3 Nemotron Super 49B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="nvidia/llama-3.3-nemotron-super-49b-v1:free",
                indicator_emoji="🚀",
                system_message="You are Llama 3.3 Nemotron Super, NVIDIA's enhanced Llama model.",
                description="NVIDIA's enhanced Llama 3.3 model - Free",
            ),
            "llama-3.1-nemotron-ultra-253b": ModelConfig(
                model_id="llama-3.1-nemotron-ultra-253b",
                display_name="Llama 3.1 Nemotron Ultra 253B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
                indicator_emoji="🚀",
                system_message="You are Llama 3.1 Nemotron Ultra, NVIDIA's largest enhanced Llama model.",
                description="NVIDIA's massive 253B enhanced Llama model - Free",
            ),
            # Meta Llama
            "llama4-maverick": ModelConfig(
                model_id="llama4-maverick",
                display_name="Llama 4 Maverick",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-4-maverick:free",
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
            "llama-3.3-70b": ModelConfig(
                model_id="llama-3.3-70b",
                display_name="Llama 3.3 70B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-3.3-70b-instruct:free",
                indicator_emoji="🦙",
                system_message="You are Llama 3.3, an advanced AI assistant by Meta.",
                description="Latest Llama 3.3 70B model - Free",
            ),
            "llama-3.2-11b-vision": ModelConfig(
                model_id="llama-3.2-11b-vision",
                display_name="Llama 3.2 11B Vision",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-3.2-11b-vision-instruct:free",
                indicator_emoji="👁️",
                system_message="You are Llama 3.2 Vision, a multimodal AI that can understand images.",
                description="Llama 3.2 with vision capabilities - Free",
                supports_images=True,
                type="vision",
                capabilities=["supports_images"],
            ),
            # Featherless
            "qwerky-72b": ModelConfig(
                model_id="qwerky-72b",
                display_name="Qwerky 72B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="featherless/qwerky-72b:free",
                indicator_emoji="🎭",
                system_message="You are Qwerky, a quirky and creative AI assistant.",
                description="Quirky creative AI model - Free",
                type="creative_writing",
                capabilities=["creative_writing"],
            ),
            # Reka AI
            "reka-flash-3": ModelConfig(
                model_id="reka-flash-3",
                display_name="Reka Flash 3",
                provider=Provider.OPENROUTER,
                openrouter_model_key="rekaai/reka-flash-3:free",
                indicator_emoji="⚡",
                system_message="You are Reka Flash 3, a fast and efficient AI assistant.",
                description="Fast and efficient Reka model - Free",
            ),
            # Nous Research
            "deephermes-3-llama-8b": ModelConfig(
                model_id="deephermes-3-llama-8b",
                display_name="DeepHermes 3 Llama 8B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="nousresearch/deephermes-3-llama-3-8b-preview:free",
                indicator_emoji="✍️",
                system_message="You are DeepHermes 3, a creative writing AI based on Llama.",
                description="Creative writing specialist based on Llama - Free",
                type="creative_writing",
                capabilities=["creative_writing"],
            ),
            # Cognitive Computations
            "dolphin3-r1-mistral-24b": ModelConfig(
                model_id="dolphin3-r1-mistral-24b",
                display_name="Dolphin 3.0 R1 Mistral 24B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
                indicator_emoji="🐬",
                system_message="You are Dolphin 3.0 R1, an enhanced uncensored AI assistant.",
                description="Enhanced uncensored conversational AI - Free",
                type="creative_writing",
                capabilities=["creative_writing"],
            ),
            "dolphin3-mistral-24b": ModelConfig(
                model_id="dolphin3-mistral-24b",
                display_name="Dolphin 3.0 Mistral 24B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="cognitivecomputations/dolphin3.0-mistral-24b:free",
                indicator_emoji="🐬",
                system_message="You are Dolphin 3.0, an uncensored AI assistant based on Mistral.",
                description="Uncensored conversational AI - Free",
                type="creative_writing",
                capabilities=["creative_writing"],
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

    @staticmethod
    def get_model_with_fallback(model_id: str) -> str:
        """Get OpenRouter model key with fallback to reliable alternatives"""
        model_map = {
            "gemini": "gemini",
            "deepseek": "deepseek",
            "cypher-alpha": "openrouter/cypher-alpha:free",
            "mistral-small-3.2-24b-instruct": "mistralai/mistral-small-3.2-24b-instruct:free",
            "mistral-small-3.1-24b-instruct": "mistralai/mistral-small-3.1-24b-instruct:free",
            "mistral-small-24b-instruct-2501": "mistralai/mistral-small-24b-instruct-2501:free",
            "mistral-nemo": "mistralai/mistral-nemo:free",
            "mistral-7b-instruct": "mistralai/mistral-7b-instruct:free",
            "devstral-small": "mistralai/devstral-small:free",
            "kimi": "moonshotai/kimi-k2:free",
            "kimi-dev-72b": "moonshotai/kimi-k2:free",
            "kimi-vl-a3b-thinking": "moonshotai/kimi-vl-a3b-thinking:free",
            "deepseek-r1-qwen3-8b": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "deepseek-r1-0528": "deepseek/deepseek-r1-0528:free",
            "deepseek-r1": "deepseek/deepseek-r1:free",
            "deepseek-chat": "deepseek/deepseek-chat:free",
            "deepseek-v3-base": "deepseek/deepseek-v3-base:free",
            "deepseek-chat-v3": "deepseek/deepseek-chat:free",
            "deepseek-r1-distill-qwen-14b": "deepseek/deepseek-r1-distill-qwen-14b:free",
            "deepseek-r1-distill-llama-70b": "deepseek/r1-distill-llama-70b:free",
            "sarvam-m": "sarvamai/sarvam-m:free",
            "gemma-3n-e4b": "google/gemma-3n-e4b-it:free",
            "gemma-3-27b": "google/gemma-3-27b-it:free",
            "gemma-3-12b": "google/gemma-3-12b-it:free",
            "gemma-3-4b": "google/gemma-3-4b-it:free",
            "gemma-2-9b": "google/gemma-2-9b-it:free",
            "gemini-2.5-pro-exp": "google/gemini-2.5-pro-exp-03-25",
            "gemini-2.0-flash-exp": "google/gemini-2.0-flash-exp:free",
            "qwen3-30b": "qwen/qwen3-30b-a3b:free",
            "qwen3-8b": "qwen/qwen3-8b:free",
            "qwen3-14b": "qwen/qwen3-14b:free",
            "qwen3-32b": "qwen/qwen3-32b:free",
            "qwen3-235b": "qwen/qwen3-235b-a22b:free",
            "qwen2.5-vl-32b": "qwen/qwen2.5-vl-32b-instruct:free",
            "qwen2.5-vl-72b": "qwen/qwen2.5-vl-72b-instruct:free",
            "qwen-2.5-coder-32b": "qwen/qwen-2.5-coder-32b-instruct:free",
            "qwen-2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct:free",
            "qwq-32b": "qwen/qwq-32b:free",
            "deepseek-r1t-chimera": "tngtech/deepseek-r1t-chimera:free",
            "mai-ds-r1": "microsoft/mai-ds-r1:free",
            "glm-z1-32b": "thudm/glm-z1-32b:free",
            "glm-4-32b": "thudm/glm-4-32b:free",
            "shisa-v2-llama3.3-70b": "shisa-ai/shisa-v2-llama3.3-70b:free",
            "qwq-32b-arliai": "arliai/qwq-32b-arliai-rpr-v1:free",
            "deepcoder-14b": "agentica-org/deepcoder-14b-preview:free",
            "llama-3.3-nemotron-super-49b": "nvidia/llama-3.3-nemotron-super-49b-v1:free",
            "llama-3.1-nemotron-ultra-253b": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
            "llama4-maverick": "meta-llama/llama-4-maverick:free",
            "llama4-scout": "meta-llama/llama-4-scout:free",
            "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
            "llama-3.2-11b-vision": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "qwerky-72b": "featherless/qwerky-72b:free",
            "reka-flash-3": "rekaai/reka-flash-3:free",
            "deephermes-3-llama-8b": "nousresearch/deephermes-3-llama-3-8b-preview:free",
            "dolphin3-r1-mistral-24b": "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            "dolphin3-mistral-24b": "cognitivecomputations/dolphin3.0-mistral-24b:free",
        }

        # Fallback mapping for specific models
        fallback_map = {
            "moonshotai/kimi-dev-72b:free": "deepseek/deepseek-chat:free",  # Add fallback for kimi-dev-72b
        }

        # First, try to get the model from the primary map
        if model_id in model_map:
            return model_map[model_id]

        # If not found, check the fallback map
        if model_id in fallback_map:
            return fallback_map[model_id]

        # If still not found, return the model_id as a last resort
        return model_id
