"""
Voice processing configuration and settings
"""

from typing import Dict, Any, List
from enum import Enum
import os
import re


class VoiceQuality(Enum):
    """Voice quality settings"""

    LOW = "low"  # Fast processing, lower accuracy
    MEDIUM = "medium"  # Balanced speed and accuracy
    HIGH = "high"  # Best accuracy, slower processing


class VoiceConfig:
    """Configuration for voice processing with Faster-Whisper"""    # Model sizes for Faster-Whisper
    WHISPER_MODEL_SIZES = {
        VoiceQuality.LOW: "tiny",
        VoiceQuality.MEDIUM: "base",
        VoiceQuality.HIGH: "large-v3",
    }

    # Special model sizes for specific languages
    LANGUAGE_SPECIFIC_MODELS = {
        "km": "large-v3",  # Khmer requires larger model for better accuracy
        "kh": "large-v3",  # Alternative Khmer code
        "th": "large-v3",  # Thai also benefits from larger model
        "vi": "base",      # Vietnamese works well with base
        "zh": "large-v3",  # Chinese requires larger model
        "ja": "large-v3",  # Japanese requires larger model
        "ko": "large-v3",  # Korean requires larger model
        "ar": "large-v3",  # Arabic requires larger model
    }# All languages now use Faster-Whisper
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
        "km": ["faster_whisper"],  # Khmer
        "kh": ["faster_whisper"],  # Alternative Khmer code
        "th": ["faster_whisper"],  # Thai
        "vi": ["faster_whisper"],  # Vietnamese
        "default": ["faster_whisper"],
    }    # Language-specific audio preprocessing - Enhanced for Khmer
    LANGUAGE_PREPROCESSING = {
        "zh": {
            "normalize": True,
            "high_pass_filter": 100,
            "low_pass_filter": 8000,
            "volume_boost": 2,
        },
        "ja": {"normalize": True, "high_pass_filter": 80, "volume_boost": 1},
        "ko": {"normalize": True, "high_pass_filter": 120, "volume_boost": 2},        "ar": {"normalize": True, "high_pass_filter": 150, "volume_boost": 3},
        "hi": {"normalize": True, "high_pass_filter": 100, "volume_boost": 2},
        "km": {
            "normalize": True, 
            "high_pass_filter": 300, 
            "low_pass_filter": 7000,  # Add low-pass to focus on speech frequencies
            "volume_boost": 4,  # Increased boost for Khmer
            "noise_reduction": True,  # Enable noise reduction for Khmer
            "sample_rate": 16000,  # Ensure optimal sample rate
        },
        "kh": {
            "normalize": True, 
            "high_pass_filter": 300, 
            "low_pass_filter": 7000,
            "volume_boost": 4,
            "noise_reduction": True,
            "sample_rate": 16000,
        },  # Alternative Khmer
        "th": {"normalize": True, "high_pass_filter": 200, "volume_boost": 2},  # Thai
        "vi": {"normalize": True, "high_pass_filter": 150, "volume_boost": 2},  # Vietnamese
        "default": {"normalize": True, "high_pass_filter": 80, "volume_boost": 0},
    }    # Confidence threshold for Faster-Whisper
    CONFIDENCE_THRESHOLDS = {
        "faster_whisper": 0.7,
        "faster_whisper_khmer": 0.3,  # Much lower threshold for Khmer due to detection challenges
        "faster_whisper_khmer_strict": 0.6,  # Strict threshold for high-confidence Khmer
    }

    # VAD (Voice Activity Detection) settings
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
        return cls.WHISPER_MODEL_SIZES.get(quality, "base")    @classmethod
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
        high_resource_langs = ["zh", "ja", "ko", "ar", "hi", "th", "km", "kh", "vi"]
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
            "en": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_EN=faster_whisper
            "es": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_ES=faster_whisper
            "fr": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_FR=faster_whisper
            "zh": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_ZH=faster_whisper
            "ja": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_JA=faster_whisper
            "ko": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_KO=faster_whisper
            "ru": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_RU=faster_whisper
            "ar": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_AR=faster_whisper
            # Additional common languages
            "km": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_KM=faster_whisper (Khmer)
            "kh": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_KH=faster_whisper (Alt Khmer)
            "vi": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_VI=faster_whisper (Vietnamese)
            "de": ["faster_whisper"],  # VOICE_ENGINE_PREFERENCES_DE=faster_whisper (German)
        }

        # Merge with class defaults
        final_prefs = cls.ENGINE_PREFERENCES.copy()
        final_prefs.update(engine_prefs)
        config["engine_preferences"] = final_prefs

        return config

    @classmethod
    def is_likely_false_english_for_khmer(cls, text: str, confidence: float = 0.0) -> bool:
        """
        Detect if an English transcription is likely a false positive for Khmer audio
        
        Args:
            text: The transcribed text
            confidence: The confidence score
            
        Returns:
            bool: True if this is likely a false positive English transcription
        """
        if not text or not text.strip():
            return False
            
        text_lower = text.lower().strip()
        
        # Common false positive patterns for Khmer -> English
        false_positive_patterns = [
            # Repetitive patterns
            r'\b(\w+)\s+\1\b',  # "so so", "slide slide", "on on"
            r'\b(\w+)\s+(\w+)\s+\1\s+\2\b',  # "so slide so slide"
            
            # Common false positive words for Khmer
            r'\b(so|slide|on|it|the|and|to|of|in|that|is|for|with|as|by|at|be|or|an|are|was|but|not|have|from|they|we|she|he|his|her|him|them|their|this|that|will|would|could|should|may|might|can|than|then|now|how|what|when|where|why|who|which|more|most|some|any|no|yes|very|just|only|even|also|still|again|back|here|there|up|down|out|off|over|under|through|before|after|between|during|while|until|since|because|if|unless|although|though|however|therefore|thus|so|yet|but|and|or|nor|for|yet|so)\b',
        ]
        
        # Check for repetitive patterns
        for pattern in false_positive_patterns[:2]:  # Check repetitive patterns
            if re.search(pattern, text_lower):
                return True
        
        # Check for too many common English words (suggests false positive)
        words = text_lower.split()
        if len(words) > 0:
            common_word_pattern = false_positive_patterns[2]
            common_words = [word for word in words if re.match(common_word_pattern, word)]
            
            # If more than 70% of words are common English words, likely false positive
            if len(common_words) / len(words) > 0.7:
                return True
        
        # Check for very short repetitive phrases
        if len(text_lower) < 50 and len(set(words)) < len(words) * 0.5:
            return True
            
        # Low confidence with suspicious patterns
        if confidence < 0.5:
            suspicious_phrases = [
                "so slide on it",
                "so slide",
                "slide on it", 
                "on it on it",
                "so so",
                "the the",
                "and and",
                "to to"
            ]
            
            for phrase in suspicious_phrases:
                if phrase in text_lower:
                    return True
        
        return False
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


# Hardcoded configuration (no longer dependent on environment)
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
