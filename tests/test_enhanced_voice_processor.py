"""
Test suite for the simplified voice processor (Faster-Whisper only)
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from src.services.media.voice_processor import (
    VoiceProcessor,
    SpeechEngine,
    create_voice_processor,
    get_supported_languages,
)
from src.services.media.voice_config import VoiceConfig, VoiceQuality


class TestVoiceProcessor:
    """Test cases for VoiceProcessor"""

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

    @pytest.mark.asyncio
    async def test_file_conversion(self, processor):
        """Test audio file conversion"""
        # Create a temporary audio file (mock)
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Mock the conversion process
            with (
                patch("pydub.AudioSegment.from_file") as mock_from_file,
                patch("pydub.AudioSegment.export") as mock_export,
            ):

                mock_audio = Mock()
                mock_audio.high_pass_filter.return_value = mock_audio
                mock_audio.normalize.return_value = mock_audio
                mock_audio.__add__.return_value = mock_audio
                mock_from_file.return_value = mock_audio

                # Test conversion
                output_path = temp_path.replace(".ogg", ".wav")
                await processor._convert_to_wav(temp_path, output_path)

                # Verify methods were called
                mock_from_file.assert_called_once_with(temp_path)
                mock_export.assert_called_once()

        finally:
            # Cleanup
            for path in [temp_path, temp_path.replace(".ogg", ".wav")]:
                if os.path.exists(path):
                    os.unlink(path)

    @pytest.mark.asyncio
    async def test_google_transcription(self, processor):
        """Test Google Speech Recognition"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Mock the speech recognition
            with patch("speech_recognition.Recognizer") as mock_recognizer_class:
                mock_recognizer = Mock()
                mock_recognizer.adjust_for_ambient_noise = Mock()
                mock_recognizer.record.return_value = Mock()
                mock_recognizer.recognize_google.return_value = "Hello world"
                mock_recognizer_class.return_value = mock_recognizer

                with patch("speech_recognition.AudioFile"):
                    text, lang, metadata = await processor._transcribe_google(
                        temp_path, "en-US"
                    )

                    assert text == "Hello world"
                    assert lang == "en-US"
                    assert metadata["engine"] == "google"
                    assert metadata["has_speech"] is True

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_transcribe_with_fallback(self, processor):
        """Test transcription with engine fallback"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Mock VAD to return True (speech detected)
            with patch.object(
                processor, "_apply_voice_activity_detection", return_value=True
            ):
                # Mock Google transcription
                with patch.object(
                    processor,
                    "_transcribe_google",
                    return_value=(
                        "Test transcription",
                        "en-US",
                        {"confidence": 0.8, "engine": "google"},
                    ),
                ):

                    text, lang, metadata = await processor.transcribe(
                        temp_path, language="en-US"
                    )

                    assert text == "Test transcription"
                    assert lang == "en-US"
                    assert metadata["engine"] == "google"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_voice_config(self):
        """Test voice configuration"""
        # Test model size mapping
        assert VoiceConfig.get_model_size(VoiceQuality.LOW) == "tiny"
        assert VoiceConfig.get_model_size(VoiceQuality.MEDIUM) == "base"
        assert VoiceConfig.get_model_size(VoiceQuality.HIGH) == "large-v3"

        # Test engine preferences
        en_prefs = VoiceConfig.get_engine_preference("en")
        assert "faster_whisper" in en_prefs or "whisper" in en_prefs

        # Test preprocessing settings
        zh_settings = VoiceConfig.get_preprocessing_settings("zh")
        assert "normalize" in zh_settings
        assert zh_settings["high_pass_filter"] > 0

        # Test high-resource language detection
        assert VoiceConfig.is_high_resource_language("zh") is True
        assert VoiceConfig.is_high_resource_language("en") is False

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
    async def test_multiple_engines_comparison(self, processor):
        """Test comparing multiple engines (simplified to single engine)"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Mock Faster-Whisper result
            mock_results = {
                SpeechEngine.FASTER_WHISPER: (
                    "Faster-Whisper result",
                    "en-US",
                    {"confidence": 0.9, "engine": "faster_whisper"},
                ),
            }

            with patch.object(processor, "transcribe") as mock_transcribe:

                async def mock_transcribe_side_effect(
                    audio_path, language=None, engine=None
                ):
                    return mock_results.get(
                        engine,
                        (
                            "",
                            language,
                            {"error": "not available", "engine": engine.value},
                        ),
                    )

                mock_transcribe.side_effect = mock_transcribe_side_effect

                results = await processor.transcribe_with_multiple_engines(
                    temp_path, engines=[SpeechEngine.FASTER_WHISPER]
                )

                assert len(results) == 1
                assert "faster_whisper" in results

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_best_transcription_selection(self, processor):
        """Test best transcription selection logic"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Mock transcription that returns high confidence result
            with patch.object(processor, "transcribe") as mock_transcribe:
                mock_transcribe.return_value = (
                    "High confidence result",
                    "en-US",
                    {"confidence": 0.9, "engine": "whisper"},
                )

                text, lang, metadata = await processor.get_best_transcription(
                    temp_path, confidence_threshold=0.7
                )

                assert text == "High confidence result"
                assert metadata["confidence"] == 0.9

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_vad_functionality(self, processor):
        """Test Voice Activity Detection"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # If VAD is not available, should return True
            if not processor.vad_available:
                result = await processor._apply_voice_activity_detection(temp_path)
                assert result is True
            else:
                # Mock VAD detection
                with patch("soundfile.read", return_value=(Mock(), 16000)):
                    with patch.object(processor.vad, "is_speech", return_value=True):
                        result = await processor._apply_voice_activity_detection(
                            temp_path
                        )
                        assert isinstance(result, bool)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_create_voice_processor_factory(self):
        """Test the factory function"""

        async def test_factory():
            processor = await create_voice_processor(
                engine=SpeechEngine.GOOGLE, enable_logging=True
            )
            assert isinstance(processor, VoiceProcessor)
            assert processor.default_engine == SpeechEngine.GOOGLE

        asyncio.run(test_factory())


class TestVoiceProcessorIntegration:
    """Integration tests for voice processor"""

    @pytest.mark.asyncio
    async def test_full_pipeline_simulation(self):
        """Test the full voice processing pipeline"""
        processor = await create_voice_processor()

        # Mock a complete pipeline
        with patch.object(processor, "download_and_convert") as mock_download:
            with patch.object(processor, "transcribe") as mock_transcribe:

                # Mock file paths
                mock_download.return_value = ("temp.ogg", "temp.wav")
                mock_transcribe.return_value = (
                    "This is a test transcription",
                    "en-US",
                    {
                        "confidence": 0.85,
                        "engine": "faster_whisper",
                        "has_speech": True,
                    },
                )

                # Simulate Telegram voice file
                mock_voice_file = Mock()
                mock_voice_file.download_to_drive = AsyncMock()

                # Process the voice file
                ogg_path, wav_path = await processor.download_and_convert(
                    mock_voice_file, "user123"
                )

                text, lang, metadata = await processor.transcribe(wav_path)

                # Verify results
                assert text == "This is a test transcription"
                assert lang == "en-US"
                assert metadata["confidence"] > 0.8
                assert metadata["has_speech"] is True

    @pytest.mark.asyncio
    async def test_error_handling_pipeline(self):
        """Test error handling in the pipeline"""
        processor = await create_voice_processor()

        # Test with non-existent file
        text, lang, metadata = await processor.transcribe(
            "non_existent_file.wav", language="en-US"
        )

        # Should handle error gracefully
        assert text == ""
        assert "error" in metadata

    def test_multilingual_support(self):
        """Test multilingual support configuration"""
        processor = VoiceProcessor()

        # Test various languages
        test_languages = ["en", "es", "fr", "de", "zh", "ja", "ko", "ar", "hi"]

        for lang in test_languages:
            engine = processor.get_recommended_engine(lang)
            assert engine in [
                SpeechEngine.WHISPER,
                SpeechEngine.FASTER_WHISPER,
                SpeechEngine.VOSK,
                SpeechEngine.GOOGLE,
            ]

            # Check if preprocessing settings exist
            settings = VoiceConfig.get_preprocessing_settings(lang)
            assert "normalize" in settings
            assert "high_pass_filter" in settings


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
