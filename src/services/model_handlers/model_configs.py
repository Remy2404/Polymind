import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum
from src.services.model_handlers.system_instructions import UNIFIED_SYSTEM_INSTRUCTION

class Provider(Enum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    DEEPSEEK = "deepseek"

@dataclass
class ModelConfig:
    model_id: str
    display_name: str
    provider: Provider
    system_message: str = UNIFIED_SYSTEM_INSTRUCTION
    indicator_emoji: str = "🤖"
    openrouter_model_key: Optional[str] = None
    max_tokens: int = 48000
    default_temperature: float = 0.7
    supports_images: bool = False
    supports_documents: bool = False
    description: str = ""
    type: str = "general_purpose"
    capabilities: List[str] = field(default_factory=list)
    supported_parameters: List[str] = field(default_factory=list)
    has_streaming_tool_conflict: bool = False

class ModelConfigurations:
    @staticmethod
    def get_all_models() -> Dict[str, ModelConfig]:
        models = {}
        try:
            models_file = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models.json")
            if os.path.exists(models_file):
                with open(models_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Unified parsing for both list and dict formats
                raw_list = []
                if isinstance(data, dict):
                    for model_list in data.values():
                        if isinstance(model_list, list): raw_list.extend(model_list)
                elif isinstance(data, list):
                    raw_list = data

                for m_data in raw_list:
                    if not m_data.get("id"): continue
                    config = ModelConfigurations._create_config_from_data(m_data)
                    models[config.model_id] = config

        except Exception as e:
            print(f"Warning: Loading models failed: {e}. Using fallbacks.")
        
        models.update(ModelConfigurations._get_hardcoded_models())
        return models

    @staticmethod
    def _create_config_from_data(data: Dict[str, Any]) -> ModelConfig:
        mid = data.get("id", "")
        provider = ModelConfigurations._determine_provider_from_id(mid)
        caps = ModelConfigurations._extract_capabilities(data)
        
        return ModelConfig(
            model_id=mid,
            display_name=data.get("name", mid),
            provider=provider,
            openrouter_model_key=mid if provider == Provider.OPENROUTER else None,
            description=data.get("description", ""),
            type=ModelConfigurations._determine_model_type(caps),
            capabilities=caps,
            supported_parameters=data.get("supported_parameters", []),
            has_streaming_tool_conflict=ModelConfigurations._check_conflict(mid, data),
            supports_images="supports_images" in caps,
            supports_documents="supports_documents" in caps,
            indicator_emoji=ModelConfigurations._get_indicator_emoji(provider, "")
        )

    @staticmethod
    def _extract_capabilities(data: Dict[str, Any]) -> List[str]:
        """Unified capability extraction from description and parameters."""
        caps = []
        desc = data.get("description", "").lower()
        params = data.get("supported_parameters", [])
        
        # Tool Calling detection
        if any(p in params for p in ["tools", "tool_choice"]) or "tool" in desc:
            caps.append("tool_calling")
        # Reasoning detection
        if "reasoning" in desc or "thinking" in desc:
            caps.append("reasoning_capable")
        # Multimodal detection
        if any(kw in desc for kw in ["vision", "image", "multimodal"]):
            caps.append("supports_images")
        if "pdf" in desc or "document" in desc:
            caps.append("supports_documents")
            
        return caps if caps else ["general_purpose"]

    @staticmethod
    def _determine_provider_from_id(mid: str) -> Provider:
        mid_l = mid.lower()
        if "gemini" in mid_l: return Provider.GEMINI
        if "deepseek" in mid_l and "/" not in mid_l: return Provider.DEEPSEEK
        return Provider.OPENROUTER

    @staticmethod
    def _check_conflict(mid: str, data: Dict[str, Any]) -> bool:
        """Simplified conflict check."""
        desc = data.get("description", "").lower()
        return "free" in mid.lower() and "llama" in mid.lower() or "limited" in desc

    @staticmethod
    def _determine_model_type(caps: List[str]) -> str:
        if "supports_images" in caps: return "vision"
        if "reasoning_capable" in caps: return "reasoning"
        return "general_purpose"

    @staticmethod
    def _get_indicator_emoji(provider: Provider, m_type: str) -> str:
        mapping = {Provider.GEMINI: "✨", Provider.DEEPSEEK: "🧠"}
        return mapping.get(provider, "🤖")

    @staticmethod
    def _get_hardcoded_models() -> Dict[str, ModelConfig]:
        return {
            "gemini": ModelConfig("gemini", "Gemini Flash", Provider.GEMINI, supports_images=True),
            "deepseek": ModelConfig("deepseek", "DeepSeek R1", Provider.DEEPSEEK, type="reasoning")
        }
