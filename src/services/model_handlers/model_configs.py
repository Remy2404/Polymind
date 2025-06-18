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
                description="Google's latest multimodal AI model",
            ),
            # DeepSeek Models
            "deepseek": ModelConfig(
                model_id="deepseek",
                display_name="DeepSeek R1",
                provider=Provider.DEEPSEEK,
                indicator_emoji="🧠",
                system_message="You are DeepSeek, an advanced reasoning AI model that excels at complex problem-solving.",
                description="Advanced reasoning model with strong analytical capabilities",
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
            ),
            "deepseek-r1-0528": ModelConfig(
                model_id="deepseek-r1-0528",
                display_name="DeepSeek R1 0528",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-0528:free",
                indicator_emoji="🧠",
                system_message="You are DeepSeek R1, an advanced reasoning AI model.",
                description="DeepSeek R1 May 2024 version - Free",
            ),            "deepseek-r1-zero": ModelConfig(
                model_id="deepseek-r1-zero",
                display_name="DeepSeek R1 Zero",
                provider=Provider.OPENROUTER,
                openrouter_model_key="deepseek/deepseek-r1-distill-llama-70b:free",
                indicator_emoji="🔬",
                system_message="You are DeepSeek R1, a model trained via large-scale reinforcement learning.",
                description="RL-trained reasoning model - Free",
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
            # === META LLAMA MODELS ===
            "llama4-maverick": ModelConfig(
                model_id="llama4-maverick",
                display_name="Llama-4 Maverick",
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
            "llama-3.3-8b": ModelConfig(
                model_id="llama-3.3-8b",
                display_name="Llama 3.3 8B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-3.3-8b-instruct:free",
                indicator_emoji="🦙",
                system_message="You are Llama 3.3, an advanced AI assistant by Meta.",
                description="Latest Llama 3.3 8B model - Free",
            ),
            "llama-3.2-11b-vision": ModelConfig(
                model_id="llama-3.2-11b-vision",
                display_name="Llama 3.2 11B Vision",
                provider=Provider.OPENROUTER,
                openrouter_model_key="meta-llama/llama-3.2-11b-vision-instruct:free",
                indicator_emoji="👁️",
                supports_images=True,
                system_message="You are Llama 3.2 Vision, capable of understanding both text and images.",
                description="Vision-capable Llama model - Free",
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
            # === MICROSOFT MODELS ===
            "phi-4-reasoning-plus": ModelConfig(
                model_id="phi-4-reasoning-plus",
                display_name="Phi-4 Reasoning Plus",
                provider=Provider.OPENROUTER,
                openrouter_model_key="microsoft/phi-4-reasoning-plus:free",
                indicator_emoji="🔬",
                system_message="You are Phi-4 Reasoning Plus, Microsoft's advanced reasoning model.",
                description="Enhanced reasoning capabilities - Free",
            ),
            "phi-4-reasoning": ModelConfig(
                model_id="phi-4-reasoning",
                display_name="Phi-4 Reasoning",
                provider=Provider.OPENROUTER,
                openrouter_model_key="microsoft/phi-4-reasoning:free",
                indicator_emoji="🔬",
                system_message="You are Phi-4, specialized in logical reasoning and problem-solving.",
                description="Microsoft's reasoning model - Free",
            ),
            "mai-ds-r1": ModelConfig(
                model_id="mai-ds-r1",
                display_name="MAI DS R1",
                provider=Provider.OPENROUTER,
                openrouter_model_key="microsoft/mai-ds-r1:free",
                indicator_emoji="🤖",
                system_message="You are MAI DS R1, Microsoft's data science reasoning model.",
                description="Data science focused model - Free",
            ),
            # === MISTRAL MODELS ===
            "mistral-small-3.1": ModelConfig(
                model_id="mistral-small-3.1",
                display_name="Mistral Small 3.1 24B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/mistral-small-3.1-24b-instruct:free",
                indicator_emoji="🌊",
                system_message="You are Mistral Small 3.1, a high-performance European AI language model.",
                description="Latest Mistral small model - Free",
            ),
            "mistral-small-24b": ModelConfig(
                model_id="mistral-small-24b",
                display_name="Mistral Small 24B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/mistral-small-24b-instruct-2501:free",
                indicator_emoji="🌊",
                system_message="You are Mistral Small, a high-performance European AI language model.",
                description="Mistral 24B parameter model - Free",
            ),
            "devstral-small": ModelConfig(
                model_id="devstral-small",
                display_name="Devstral Small",
                provider=Provider.OPENROUTER,
                openrouter_model_key="mistralai/devstral-small:free",
                indicator_emoji="💻",
                system_message="You are Devstral, Mistral's coding-focused AI assistant.",
                description="Development-focused Mistral model - Free",
            ),
            # === GOOGLE GEMMA MODELS ===
            "gemma-3-27b": ModelConfig(
                model_id="gemma-3-27b",
                display_name="Gemma 3 27B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3-27b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3, Google's latest efficient AI assistant.",
                description="Large Gemma 3 model - Free",
            ),
            "gemma-3-12b": ModelConfig(
                model_id="gemma-3-12b",
                display_name="Gemma 3 12B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3-12b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3, Google's efficient AI assistant.",
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
            "gemma-3-1b": ModelConfig(
                model_id="gemma-3-1b",
                display_name="Gemma 3 1B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3-1b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3, Google's ultra-lightweight AI assistant.",
                description="Ultra-compact Gemma 3 model - Free",
            ),
            "gemma-3n-e4b": ModelConfig(
                model_id="gemma-3n-e4b",
                display_name="Gemma 3N E4B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="google/gemma-3n-e4b-it:free",
                indicator_emoji="💎",
                system_message="You are Gemma 3N, Google's next-generation efficient AI assistant.",
                description="Next-gen Gemma model - Free",
            ),
            # === NVIDIA MODELS ===
            "llama-3.3-nemotron-super-49b": ModelConfig(
                model_id="llama-3.3-nemotron-super-49b",
                display_name="Llama 3.3 Nemotron Super 49B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="nvidia/llama-3.3-nemotron-super-49b-v1:free",
                indicator_emoji="⚡",
                system_message="You are Llama Nemotron Super, NVIDIA's enhanced AI assistant.",
                description="NVIDIA's super-enhanced model - Free",
            ),
            "llama-3.1-nemotron-ultra-253b": ModelConfig(
                model_id="llama-3.1-nemotron-ultra-253b",
                display_name="Llama 3.1 Nemotron Ultra 253B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
                indicator_emoji="⚡",
                system_message="You are Llama Nemotron Ultra, NVIDIA's largest AI assistant.",
                description="Massive 253B NVIDIA model - Free",
            ),
            # === THUDM MODELS ===
            "glm-z1-32b": ModelConfig(
                model_id="glm-z1-32b",
                display_name="GLM Z1 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="thudm/glm-z1-32b:free",
                indicator_emoji="🔥",
                system_message="You are GLM Z1, an advanced Chinese language model with strong reasoning capabilities.",
                description="Advanced GLM reasoning model - Free",
            ),
            "glm-4-32b": ModelConfig(
                model_id="glm-4-32b",
                display_name="GLM-4 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="thudm/glm-4-32b:free",
                indicator_emoji="🔥",
                system_message="You are GLM-4, a powerful Chinese language model with multilingual capabilities.",
                description="GLM-4 large model - Free",
            ),
            # === CODING SPECIALISTS ===
            "deepcoder": ModelConfig(
                model_id="deepcoder",
                display_name="DeepCoder 14B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="agentica-org/deepcoder-14b-preview:free",
                indicator_emoji="💻",
                system_message="You are DeepCoder, an AI specialized in programming and software development.",
                description="Code generation specialist - Free",
            ),
            "olympiccoder-32b": ModelConfig(
                model_id="olympiccoder-32b",
                display_name="OlympicCoder 32B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="open-r1/olympiccoder-32b:free",
                indicator_emoji="🏆",
                system_message="You are OlympicCoder, specialized in competitive programming and complex algorithms.",
                description="Competitive programming specialist - Free",
            ),
            # === CREATIVE & SPECIALIZED MODELS ===
            "shisa-v2-llama3.3-70b": ModelConfig(
                model_id="shisa-v2-llama3.3-70b",
                display_name="Shisa V2 Llama3.3 70B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="shisa-ai/shisa-v2-llama3.3-70b:free",
                indicator_emoji="🗾",
                system_message="You are Shisa V2, a Japanese-English bilingual AI assistant.",
                description="Japanese-English bilingual model - Free",
            ),
            "deephermes-3-mistral-24b": ModelConfig(
                model_id="deephermes-3-mistral-24b",
                display_name="DeepHermes 3 Mistral 24B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="nousresearch/deephermes-3-mistral-24b-preview:free",
                indicator_emoji="📜",
                system_message="You are DeepHermes 3, a versatile AI assistant with enhanced conversational abilities.",
                description="Enhanced conversational model - Free",
            ),
            "deephermes-3-llama-8b": ModelConfig(
                model_id="deephermes-3-llama-8b",
                display_name="DeepHermes 3 Llama 8B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="nousresearch/deephermes-3-llama-3-8b-preview:free",
                indicator_emoji="📜",
                system_message="You are DeepHermes 3, a versatile AI assistant optimized for helpful conversations.",
                description="Conversational AI specialist - Free",
            ),
            "dolphin3.0-r1-mistral-24b": ModelConfig(
                model_id="dolphin3.0-r1-mistral-24b",
                display_name="Dolphin 3.0 R1 Mistral 24B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
                indicator_emoji="🐬",
                system_message="You are Dolphin 3.0, an uncensored AI assistant focused on helpful responses.",
                description="Uncensored conversational model - Free",
            ),
            "dolphin3.0-mistral-24b": ModelConfig(
                model_id="dolphin3.0-mistral-24b",
                display_name="Dolphin 3.0 Mistral 24B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="cognitivecomputations/dolphin3.0-mistral-24b:free",
                indicator_emoji="🐬",
                system_message="You are Dolphin 3.0, an uncensored AI assistant designed for helpful conversations.",
                description="Uncensored AI model - Free",
            ),
            # === VISION MODELS ===
            "internvl3-14b": ModelConfig(
                model_id="internvl3-14b",
                display_name="InternVL3 14B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="opengvlab/internvl3-14b:free",
                indicator_emoji="👁️",
                supports_images=True,
                system_message="You are InternVL3, a vision-language model capable of understanding images and text.",
                description="Advanced vision-language model - Free",
            ),
            "internvl3-2b": ModelConfig(
                model_id="internvl3-2b",
                display_name="InternVL3 2B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="opengvlab/internvl3-2b:free",
                indicator_emoji="👁️",
                supports_images=True,
                system_message="You are InternVL3, a compact vision-language model for image understanding.",
                description="Compact vision model - Free",
            ),
            "kimi-vl-a3b-thinking": ModelConfig(
                model_id="kimi-vl-a3b-thinking",
                display_name="Kimi VL A3B Thinking",
                provider=Provider.OPENROUTER,
                openrouter_model_key="moonshotai/kimi-vl-a3b-thinking:free",
                indicator_emoji="🌙",
                supports_images=True,
                system_message="You are Kimi VL, a thinking-capable vision-language model.",
                description="Vision model with reasoning - Free",
            ),
            # === SPECIALIZED MODELS ===
            "sarvam-m": ModelConfig(
                model_id="sarvam-m",
                display_name="Sarvam M",
                provider=Provider.OPENROUTER,
                openrouter_model_key="sarvamai/sarvam-m:free",
                indicator_emoji="🕉️",
                system_message="You are Sarvam M, specialized in Indian languages and cultural contexts.",
                description="Indian language specialist - Free",
            ),
            "qwerky-72b": ModelConfig(
                model_id="qwerky-72b",
                display_name="Qwerky 72B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="featherless/qwerky-72b:free",
                indicator_emoji="🤪",
                system_message="You are Qwerky, a creative and quirky AI with unique personality.",
                description="Creative personality model - Free",
            ),
            "reka-flash-3": ModelConfig(
                model_id="reka-flash-3",
                display_name="Reka Flash 3",
                provider=Provider.OPENROUTER,
                openrouter_model_key="rekaai/reka-flash-3:free",
                indicator_emoji="⚡",
                system_message="You are Reka Flash 3, a fast and efficient AI assistant.",
                description="Fast response model - Free",
            ),
            "moonlight-16b": ModelConfig(
                model_id="moonlight-16b",
                display_name="Moonlight 16B A3B",
                provider=Provider.OPENROUTER,
                openrouter_model_key="moonshotai/moonlight-16b-a3b-instruct:free",
                indicator_emoji="🌙",
                system_message="You are Moonlight, an AI assistant optimized for thoughtful conversations.",
                description="Thoughtful conversation model - Free",
            ),
            "deepseek-r1t-chimera": ModelConfig(
                model_id="deepseek-r1t-chimera",
                display_name="DeepSeek R1T Chimera",
                provider=Provider.OPENROUTER,
                openrouter_model_key="tngtech/deepseek-r1t-chimera:free",
                indicator_emoji="🧬",
                system_message="You are DeepSeek R1T Chimera, a hybrid reasoning model with enhanced capabilities.",
                description="Hybrid reasoning model - Free",
            ),
            "qwq-32b-arliai": ModelConfig(
                model_id="qwq-32b-arliai",
                display_name="QwQ 32B ArliAI",
                provider=Provider.OPENROUTER,
                openrouter_model_key="arliai/qwq-32b-arliai-rpr-v1:free",
                indicator_emoji="🤔",
                system_message="You are QwQ ArliAI, enhanced for step-by-step reasoning and problem solving.",
                description="Enhanced reasoning model - Free",
            ),
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
