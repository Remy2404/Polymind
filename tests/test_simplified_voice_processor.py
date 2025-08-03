"""
Test suite for the simplified voice processor (Faster-Whisper only)
"""

import pytest
import tempfile
import os
from unittest.mock import patch
from src.services.media.voice_processor import (
    VoiceProcessor,
    SpeechEngine,
    create_voice_processor,
    get_supported_languages,
)
from src.services.media.voice_config import VoiceConfig, VoiceQuality


class TestSimplifiedVoiceProcessor:
    """Test cases for simplified VoiceProcessor (Faster-Whisper only)"""

    @pytest.fixture
    async def processor(self):
        """Create a test voice processor"""
        return await create_voice_processor(engine=SpeechEngine.FASTER_WHISPER)

    def test_engine_availability_check(self):
        """Test engine availability detection"""
        processor = VoiceProcessor()
        available = processor._check_available_engines()

        # Only Faster-Whisper should be available
        assert available["faster_whisper"] is True

        # Other engines should not be present
        assert "whisper" not in available
        assert "vosk" not in available
        assert "google" not in available

    def test_engine_recommendation(self):
        """Test engine recommendation logic"""
        processor = VoiceProcessor()

        # All languages should use Faster-Whisper
        en_engine = processor.get_recommended_engine("en")
        assert en_engine == SpeechEngine.FASTER_WHISPER

        # Non-English should also use Faster-Whisper
        es_engine = processor.get_recommended_engine("es")
        assert es_engine == SpeechEngine.FASTER_WHISPER

    def test_supported_languages(self):
        """Test supported languages function"""
        languages = get_supported_languages()

        # Check that only Faster-Whisper is represented
        assert "faster_whisper" in languages

        # Check that languages are lists
        for engine, langs in languages.items():
            assert isinstance(langs, list)
            assert len(langs) > 0
            assert "en" in langs  # English should be supported

    @pytest.mark.asyncio
    async def test_engine_info(self, processor):
        """Test engine information retrieval"""
        info = processor.get_engine_info()

        assert "available_engines" in info
        assert "default_engine" in info
        assert "recommended_engines" in info
        assert "features" in info

        # Check features structure (simplified for Faster-Whisper only)
        features = info["features"]
        if "faster_whisper" in features:
            engine_features = features["faster_whisper"]
            assert "multilingual" in engine_features
            assert "timestamps" in engine_features
            assert "offline" in engine_features
            assert "accuracy" in engine_features

    @pytest.mark.asyncio
    async def test_transcription(self, processor):
        """Test basic transcription functionality"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Mock the transcription process
            with patch.object(
                processor, "_transcribe_with_faster_whisper"
            ) as mock_transcribe:
                mock_transcribe.return_value = (
                    "Hello world",
                    "en",
                    {"confidence": 0.9},
                )

                result = await processor.transcribe(
                    temp_path, engine=SpeechEngine.FASTER_WHISPER
                )

                assert result[0] == "Hello world"
                assert result[1] == "en"
                assert result[2]["confidence"] == 0.9

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_processor_creation(self):
        """Test processor creation with different configurations"""
        # Test default creation
        processor = await create_voice_processor()
        assert processor.default_engine == SpeechEngine.FASTER_WHISPER

        # Test explicit engine specification
        processor = await create_voice_processor(engine=SpeechEngine.FASTER_WHISPER)
        assert processor.default_engine == SpeechEngine.FASTER_WHISPER

    def test_voice_config_integration(self):
        """Test integration with VoiceConfig"""
        config = VoiceConfig()

        # Test quality settings
        assert config.quality in [
            VoiceQuality.FAST,
            VoiceQuality.BALANCED,
            VoiceQuality.HIGH,
        ]

        # Test language detection
        assert config.detect_language is not None

        # Test engine selection
        assert (
            config.engine == SpeechEngine.FASTER_WHISPER
            or config.engine == SpeechEngine.AUTO
        )


class TestVoiceConfig:
    """Test cases for VoiceConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = VoiceConfig()

        assert config.engine in [SpeechEngine.FASTER_WHISPER, SpeechEngine.AUTO]
        assert config.quality in [
            VoiceQuality.FAST,
            VoiceQuality.BALANCED,
            VoiceQuality.HIGH,
        ]
        assert isinstance(config.detect_language, bool)
        assert isinstance(config.return_timestamps, bool)

    def test_config_validation(self):
        """Test configuration validation"""
        config = VoiceConfig()
        # Test that only valid engines are accepted
        assert config.engine in [SpeechEngine.FASTER_WHISPER, SpeechEngine.AUTO]

        # Test quality settings
        assert config.quality in [
            VoiceQuality.FAST,
            VoiceQuality.BALANCED,
            VoiceQuality.HIGH,
        ]


if __name__ == "__main__":
    pytest.main([__file__])
