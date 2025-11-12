"""
Voice processing configuration and settings
Simplified to support English only to save space
"""

from typing import Dict, Any, List
from enum import Enum


class VoiceQuality(Enum):
    """Voice quality settings"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VoiceConfig:
    """Configuration for voice processing with Faster-Whisper - English only"""

    WHISPER_MODEL_SIZES = {
        VoiceQuality.LOW: "tiny",
        VoiceQuality.MEDIUM: "base",
        VoiceQuality.HIGH: "base",
    }
    ENGINE_PREFERENCES = {
        "en": ["faster_whisper"],
        "default": ["faster_whisper"],
    }
    LANGUAGE_PREPROCESSING = {
        "en": {"normalize": True, "high_pass_filter": 80, "volume_boost": 1},
        "default": {"normalize": True, "high_pass_filter": 80, "volume_boost": 1},
    }
    CONFIDENCE_THRESHOLDS = {
        "faster_whisper": 0.7,
    }
    VAD_SETTINGS = {
        "aggressiveness": 2,
        "min_speech_ratio": 0.3,
        "frame_duration_ms": 30,
    }
    AUDIO_SETTINGS = {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "chunk_size": 4000,
        "max_file_size_mb": 50,
        "timeout_seconds": 300,
    }

    @classmethod
    def get_model_size(cls, quality: VoiceQuality) -> str:
        """Get Whisper model size for given quality"""
        return cls.WHISPER_MODEL_SIZES.get(quality, "base") @ classmethod

    def get_engine_preference(cls, language: str) -> List[str]:
        """Get preferred engines for a language"""
        lang_code = (
            language.split("-")[0].lower() if "-" in language else language.lower()
        )
        return cls.ENGINE_PREFERENCES.get(lang_code, cls.ENGINE_PREFERENCES["default"])

    @classmethod
    def get_preprocessing_settings(cls, language: str) -> Dict[str, Any]:
        """Get audio preprocessing settings for a language"""
        lang_code = (
            language.split("-")[0].lower() if "-" in language else language.lower()
        )
        return cls.LANGUAGE_PREPROCESSING.get(
            lang_code, cls.LANGUAGE_PREPROCESSING["default"]
        )

    @classmethod
    def get_confidence_threshold(cls, engine: str) -> float:
        """Get confidence threshold for an engine"""
        return cls.CONFIDENCE_THRESHOLDS.get(engine, 0.5) @ classmethod

    def is_high_resource_language(cls, language: str) -> bool:
        """Check if language requires high-resource processing"""
        return False

    @classmethod
    def get_recommended_quality(
        cls, language: str, file_size_mb: float
    ) -> VoiceQuality:
        """Get recommended quality based on language and file size"""
        if file_size_mb > 10 or cls.is_high_resource_language(language):
            return VoiceQuality.HIGH
        elif file_size_mb > 2:
            return VoiceQuality.MEDIUM
        else:
            return VoiceQuality.LOW

    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """Load configuration - now hardcoded for stability"""
        config = {}
        config["enabled"] = True
        config["default_engine"] = "faster_whisper"
        config["quality"] = "medium"
        config["max_file_size_mb"] = 50
        config["timeout_seconds"] = 300
        config["enable_vad"] = True
        config["cache_models"] = True
        config["language_detection"] = True
        engine_prefs = {
            "en": ["faster_whisper"],
        }
        final_prefs = cls.ENGINE_PREFERENCES.copy()
        final_prefs.update(engine_prefs)
        config["engine_preferences"] = final_prefs
        return config


class VoiceStats:
    """Statistics tracking for voice processing"""

    def __init__(self):
        self.stats = {
            "total_processed": 0,
            "by_engine": {},
            "by_language": {},
            "success_rate": {},
            "avg_processing_time": {},
            "avg_confidence": {},
        }

    def record_result(
        self,
        engine: str,
        language: str,
        success: bool,
        processing_time: float,
        confidence: float,
    ):
        """Record processing result"""
        self.stats["total_processed"] += 1
        if engine not in self.stats["by_engine"]:
            self.stats["by_engine"][engine] = {
                "count": 0,
                "success": 0,
                "total_time": 0,
                "total_confidence": 0,
            }
        engine_stats = self.stats["by_engine"][engine]
        engine_stats["count"] += 1
        engine_stats["total_time"] += processing_time
        engine_stats["total_confidence"] += confidence
        if success:
            engine_stats["success"] += 1
        if language not in self.stats["by_language"]:
            self.stats["by_language"][language] = {"count": 0, "success": 0}
        lang_stats = self.stats["by_language"][language]
        lang_stats["count"] += 1
        if success:
            lang_stats["success"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        for engine, data in self.stats["by_engine"].items():
            if data["count"] > 0:
                self.stats["success_rate"][engine] = data["success"] / data["count"]
                self.stats["avg_processing_time"][engine] = (
                    data["total_time"] / data["count"]
                )
                self.stats["avg_confidence"][engine] = (
                    data["total_confidence"] / data["count"]
                )
        return self.stats

    def get_best_engine(self, metric: str = "success_rate") -> str:
        """Get best performing engine by metric"""
        if metric not in ["success_rate", "avg_processing_time", "avg_confidence"]:
            metric = "success_rate"
        stats = self.get_stats()
        if not stats[metric]:
            return "unknown"
        if metric == "avg_processing_time":
            return min(stats[metric].items(), key=lambda x: x[1])[0]
        else:
            return max(stats[metric].items(), key=lambda x: x[1])[0]


voice_config = VoiceConfig()
voice_stats = VoiceStats()


def load_config_from_env() -> Dict[str, Any]:
    """Load configuration - now hardcoded for stability"""
    return {
        "default_quality": VoiceQuality.MEDIUM,
        "max_file_size_mb": 50,
        "timeout_seconds": 300,
        "enable_vad": True,
        "cache_models": True,
        "log_level": "INFO",
    }
