"""
Voice processing configuration and settings
"""

from typing import Dict, Any, List
from enum import Enum
import os


class VoiceQuality(Enum):
    """Voice quality settings"""

    LOW = "low"  # Fast processing, lower accuracy
    MEDIUM = "medium"  # Balanced speed and accuracy
    HIGH = "high"  # Best accuracy, slower processing


class VoiceConfig:
    """Configuration for voice processing with Faster-Whisper"""

    # Model sizes for Faster-Whisper
    WHISPER_MODEL_SIZES = {
        VoiceQuality.LOW: "tiny",
        VoiceQuality.MEDIUM: "base",
        VoiceQuality.HIGH: "large-v3",
    }

    # All languages now use Faster-Whisper
    ENGINE_PREFERENCES = {
        "en": ["faster_whisper"],
        "es": ["faster_whisper"],
        "fr": ["faster_whisper"],
        "de": ["faster_whisper"],
        "zh": ["faster_whisper"],
        "ja": ["faster_whisper"],
        "ko": ["faster_whisper"],
        "ru": ["faster_whisper"],
        "ar": ["faster_whisper"],
        "hi": ["faster_whisper"],
        "default": ["faster_whisper"],
    }

    # Language-specific audio preprocessing
    LANGUAGE_PREPROCESSING = {
        "zh": {
            "normalize": True,
            "high_pass_filter": 100,
            "low_pass_filter": 8000,
            "volume_boost": 2,
        },
        "ja": {"normalize": True, "high_pass_filter": 80, "volume_boost": 1},
        "ko": {"normalize": True, "high_pass_filter": 120, "volume_boost": 2},
        "ar": {"normalize": True, "high_pass_filter": 150, "volume_boost": 3},
        "hi": {"normalize": True, "high_pass_filter": 100, "volume_boost": 2},
        "th": {"normalize": True, "high_pass_filter": 200, "volume_boost": 3},  # Thai
        "km": {"normalize": True, "high_pass_filter": 300, "volume_boost": 3},  # Khmer
        "default": {"normalize": True, "high_pass_filter": 80, "volume_boost": 0},
    }    # Confidence threshold for Faster-Whisper
    CONFIDENCE_THRESHOLDS = {
        "faster_whisper": 0.7,
    }

    # VAD (Voice Activity Detection) settings
    VAD_SETTINGS = {
        "aggressiveness": 2,  # 0-3, higher = more aggressive
        "min_speech_ratio": 0.3,  # Minimum ratio of speech frames
        "frame_duration_ms": 30,  # Frame duration in milliseconds
    }    # Audio processing settings
    AUDIO_SETTINGS = {
        "sample_rate": 16000,  # 16kHz is optimal for speech recognition
        "channels": 1,  # Mono
        "bit_depth": 16,  # 16-bit
        "chunk_size": 4000,  # Frames per chunk for processing
        "max_file_size_mb": 50,  # Maximum audio file size
        "timeout_seconds": 300,  # Maximum processing time
    }

    @classmethod
    def get_model_size(cls, quality: VoiceQuality) -> str:
        """Get Whisper model size for given quality"""
        return cls.WHISPER_MODEL_SIZES.get(quality, "base")

    @classmethod
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
        return cls.CONFIDENCE_THRESHOLDS.get(engine, 0.5)

    @classmethod
    def is_high_resource_language(cls, language: str) -> bool:
        """Check if language requires high-resource processing"""
        high_resource_langs = ["zh", "ja", "ko", "ar", "hi", "th", "km", "vi"]
        lang_code = (
            language.split("-")[0].lower() if "-" in language else language.lower()
        )
        return lang_code in high_resource_langs

    @classmethod
    def get_recommended_quality(
        cls, language: str, file_size_mb: float
    ) -> VoiceQuality:
        """Get recommended quality based on language and file size"""
        # For large files or high-resource languages, use higher quality
        if file_size_mb > 10 or cls.is_high_resource_language(language):
            return VoiceQuality.HIGH
        elif file_size_mb > 2:
            return VoiceQuality.MEDIUM
        else:
            return VoiceQuality.LOW

    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}

        # Basic voice settings
        config["enabled"] = (
            os.getenv("VOICE_PROCESSING_ENABLED", "true").lower() == "true"
        )
        config["default_engine"] = os.getenv("VOICE_DEFAULT_ENGINE", "auto")
        config["quality"] = os.getenv("VOICE_QUALITY", "medium")
        config["max_file_size_mb"] = int(os.getenv("VOICE_MAX_FILE_SIZE_MB", "50"))
        config["timeout_seconds"] = int(os.getenv("VOICE_TIMEOUT_SECONDS", "300"))
        config["enable_vad"] = os.getenv("VOICE_ENABLE_VAD", "true").lower() == "true"
        config["cache_models"] = (
            os.getenv("VOICE_CACHE_MODELS", "true").lower() == "true"
        )
        config["language_detection"] = (
            os.getenv("VOICE_LANGUAGE_DETECTION", "true").lower() == "true"
        )

        # Engine preferences from environment
        engine_prefs = {}
        for lang_code in ["en", "es", "fr", "zh", "ja", "ko", "ru", "ar"]:
            env_var = f"VOICE_ENGINE_PREFERENCES_{lang_code.upper()}"
            pref_string = os.getenv(env_var)
            if pref_string:
                engine_prefs[lang_code] = [
                    engine.strip() for engine in pref_string.split(",")
                ]

        # Merge with defaults
        final_prefs = cls.ENGINE_PREFERENCES.copy()
        final_prefs.update(engine_prefs)
        config["engine_preferences"] = final_prefs

        return config

    # ...existing code...


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

        # Update by engine
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

        # Update by language
        if language not in self.stats["by_language"]:
            self.stats["by_language"][language] = {"count": 0, "success": 0}

        lang_stats = self.stats["by_language"][language]
        lang_stats["count"] += 1
        if success:
            lang_stats["success"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        # Calculate derived stats
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
            # For processing time, lower is better
            return min(stats[metric].items(), key=lambda x: x[1])[0]
        else:
            # For success rate and confidence, higher is better
            return max(stats[metric].items(), key=lambda x: x[1])[0]


# Global configuration instance
voice_config = VoiceConfig()
voice_stats = VoiceStats()


# Environment-based configuration
def load_config_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    return {
        "default_quality": VoiceQuality(os.getenv("VOICE_QUALITY", "medium")),
        "max_file_size_mb": int(os.getenv("VOICE_MAX_FILE_SIZE_MB", "50")),
        "timeout_seconds": int(os.getenv("VOICE_TIMEOUT_SECONDS", "300")),
        "enable_vad": os.getenv("VOICE_ENABLE_VAD", "true").lower() == "true",
        "cache_models": os.getenv("VOICE_CACHE_MODELS", "true").lower() == "true",
        "log_level": os.getenv("VOICE_LOG_LEVEL", "INFO"),
    }
