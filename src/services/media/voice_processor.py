import os
import logging
import asyncio
import tempfile
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum
from pydub import AudioSegment
import uuid

# Import Faster-Whisper (only engine we'll use)
try:
    from faster_whisper import WhisperModel

    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    raise ImportError(
        "faster_whisper is required but not installed. Please install it with: pip install faster-whisper"
    )


class SpeechEngine(Enum):
    """Available speech recognition engines"""

    FASTER_WHISPER = "faster_whisper"
    AUTO = "auto"


class VoiceProcessor:
    """
    Enhanced voice message processing and transcription with multiple engines.
    Supports OpenAI Whisper, Faster-Whisper, Vosk, and Google Speech Recognition.
    """

    def __init__(self, default_engine: SpeechEngine = SpeechEngine.FASTER_WHISPER):
        """Initialize the voice processor with Faster-Whisper engine"""
        self.logger = logging.getLogger(__name__)
        self.temp_dir = tempfile.gettempdir()
        self.default_engine = default_engine

        # Model cache (only Faster-Whisper)
        self._faster_whisper_model = None

        # Initialize available engines
        self.available_engines = self._check_available_engines()
        self.logger.info(
            f"Available speech engines: {list(self.available_engines.keys())}"
        )

        # VAD is handled by Faster-Whisper internally
        self.vad_available = False

    def _check_available_engines(self) -> Dict[str, bool]:
        """Check which speech engines are available"""
        return {
            SpeechEngine.FASTER_WHISPER.value: FASTER_WHISPER_AVAILABLE,
        }

    def get_recommended_engine(self, language: str = "en") -> SpeechEngine:
        """Get recommended engine based on language and availability"""
        # Always use Faster-Whisper as it's our only engine
        if FASTER_WHISPER_AVAILABLE:
            return SpeechEngine.FASTER_WHISPER
        else:
            raise RuntimeError(
                "Faster-Whisper is not available. Please install it with: pip install faster-whisper"
            )

    async def _load_faster_whisper_model(
        self, model_size: str = "base"
    ) -> Optional[WhisperModel]:
        """Load Faster-Whisper model"""
        if not FASTER_WHISPER_AVAILABLE:
            return None

        if self._faster_whisper_model is None:
            try:
                self.logger.info(f"Loading Faster-Whisper model: {model_size}")
                # Use CPU for compatibility
                device = "cpu"
                compute_type = "int8"

                self._faster_whisper_model = WhisperModel(
                    model_size, device=device, compute_type=compute_type
                )
                self.logger.info(f"Faster-Whisper model loaded on {device}")
            except Exception as e:
                self.logger.error(f"Failed to load Faster-Whisper model: {e}")
                return None
        return self._faster_whisper_model

    async def download_and_convert(self, voice_file, user_id: str) -> Tuple[str, str]:
        """
        Download voice file and convert to WAV format for speech recognition

        Args:
            voice_file: Telegram file object
            user_id: User ID for file naming

        Returns:
            Tuple[str, str]: Path to original file and converted WAV file
        """
        try:
            # Generate a unique filename to avoid conflicts
            file_unique_id = str(uuid.uuid4())[:8]
            ogg_file_path = os.path.join(
                self.temp_dir, f"voice_{user_id}_{file_unique_id}.ogg"
            )
            wav_file_path = os.path.join(
                self.temp_dir, f"voice_{user_id}_{file_unique_id}.wav"
            )

            # Download the voice file
            await voice_file.download_to_drive(ogg_file_path)

            # Convert OGG to WAV
            await self._convert_to_wav(ogg_file_path, wav_file_path)

            return ogg_file_path, wav_file_path

        except Exception as e:
            self.logger.error(f"Error downloading/converting voice: {str(e)}")
            raise ValueError(f"Failed to process voice file: {str(e)}")

    async def _convert_to_wav(self, input_path: str, output_path: str) -> None:
        """
        Convert an audio file to WAV format optimized for speech recognition

        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file
        """
        try:

            def _do_convert():
                # Load the audio file
                audio = AudioSegment.from_file(input_path)

                # Apply standard English preprocessing
                audio = audio.normalize()  # Normalize volume
                audio = audio.high_pass_filter(80)  # Remove very low frequencies

                # Export as WAV with speech recognition friendly settings
                sample_rate = 16000

                audio.export(
                    output_path,
                    format="wav",
                    parameters=[
                        "-ar",
                        str(sample_rate),
                        "-ac",
                        "1",
                    ],  # Sample rate, mono
                )

            # Run conversion in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do_convert)

        except Exception as e:
            self.logger.error(f"Audio conversion error: {str(e)}")
            raise ValueError(f"Failed to convert audio: {str(e)}")

    async def _apply_voice_activity_detection(self, audio_path: str) -> bool:
        """
        Apply voice activity detection to determine if audio contains speech
        Since we removed webrtcvad, we'll rely on Faster-Whisper's built-in VAD

        Args:
            audio_path: Path to audio file

        Returns:
            bool: Always True (let Faster-Whisper handle VAD internally)
        """
        # Faster-Whisper has built-in VAD, so we don't need external VAD
        return True

    async def transcribe(
        self,
        audio_file_path: str,
        language: str = "en-US",
        engine: Optional[SpeechEngine] = None,
        model_size: str = "base",
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Transcribe a voice file to text using Faster-Whisper

        Args:
            audio_file_path: Path to WAV file
            language: Language code for recognition (only English supported)
            engine: Specific engine to use (if None, uses default/auto)
            model_size: Model size for Whisper engines

        Returns:
            Tuple[str, str, Dict]: Transcribed text, detected language, and metadata
        """
        try:
            # VAD is handled by Faster-Whisper's built-in vad_filter
            # No need for external VAD check

            # Determine engine to use
            if engine is None:
                if self.default_engine == SpeechEngine.AUTO:
                    engine = self.get_recommended_engine(language)
                else:
                    engine = self.default_engine

            # Ensure the engine is available
            if not self.available_engines.get(engine.value, False):
                raise RuntimeError(
                    "Faster-Whisper engine not available. Please install it with: pip install faster-whisper"
                )

            self.logger.info(f"Using speech engine: {engine.value}")

            # Transcribe using Faster-Whisper (only engine available)
            result = await self._transcribe_faster_whisper(
                audio_file_path, language, model_size
            )

            # Clean up temp files in background
            asyncio.create_task(self._cleanup_files(audio_file_path))

            return result

        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            # Attempt to clean up even on error
            asyncio.create_task(self._cleanup_files(audio_file_path))
            return (
                "",
                language,
                {"error": str(e), "engine": engine.value if engine else "unknown"},
            )

    async def _transcribe_faster_whisper(
        self, audio_file_path: str, language: str, model_size: str = "base"
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Transcribe using Faster-Whisper"""
        # Convert language code first
        lang_code = language.split("-")[0] if "-" in language else language

        # Always use standard processing for English
        model = await self._load_faster_whisper_model(model_size)
        if model is None:
            raise ValueError("Faster-Whisper model not available")

        def _do_transcribe():
            try:
                self.logger.info("🔍 STANDARD TRANSCRIPTION:")
                self.logger.info(f"  → Input language: {language}")
                self.logger.info(f"  → Processed lang_code: {lang_code}")
                self.logger.info(f"  → Audio file: {audio_file_path}")
                self.logger.info(f"  → Model size: {model_size}")

                # Standard English processing
                segments, info = model.transcribe(
                    audio_file_path,
                    language=lang_code if lang_code != "auto" else None,
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,  # Built-in VAD
                )

                # Collect segments
                segments_list = list(segments)
                text = " ".join([segment.text.strip() for segment in segments_list])

                self.logger.info(
                    f"  → Result - Detected: {info.language}, Confidence: {info.language_probability:.3f}"
                )
                self.logger.info(f"  → Text: {text[:100]}...")

                # Final logging
                self.logger.info("🎯 FINAL STANDARD TRANSCRIPTION RESULT:")
                self.logger.info(f"  → Final language: {info.language}")
                self.logger.info(
                    f"  → Final confidence: {info.language_probability:.3f}"
                )
                self.logger.info(f"  → Final text length: {len(text)} chars")
                self.logger.info(f"  → Segments count: {len(segments_list)}")

                return (
                    text,
                    info.language,
                    {
                        "confidence": info.language_probability,
                        "engine": "faster_whisper",
                        "model_size": model_size,
                        "segments": [
                            {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text,
                                "words": (
                                    [
                                        {
                                            "start": w.start,
                                            "end": w.end,
                                            "word": w.word,
                                            "probability": w.probability,
                                        }
                                        for w in seg.words
                                    ]
                                    if hasattr(seg, "words") and seg.words
                                    else []
                                ),
                            }
                            for seg in segments_list
                        ],
                        "has_speech": True,
                        "requested_language": language,  # Add this for debugging
                        "detected_language": info.language,  # Add this for debugging
                    },
                )
            except Exception as e:
                self.logger.error(f"Faster-Whisper transcription error: {e}")
                return "", language, {"error": str(e), "engine": "faster_whisper"}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _do_transcribe)

    async def transcribe_with_multiple_engines(
        self,
        audio_file_path: str,
        language: str = "en-US",
        engines: Optional[List[SpeechEngine]] = None,
    ) -> Dict[str, Tuple[str, str, Dict[str, Any]]]:
        """
        Transcribe using multiple model sizes for comparison

        Args:
            audio_file_path: Path to audio file
            language: Language code
            engines: List of model sizes to use (if None, uses different sizes)

        Returns:
            Dict mapping model sizes to transcription results
        """
        if engines is None:
            # Test different model sizes instead of different engines
            model_sizes = ["tiny", "base", "small", "medium"]
        else:
            model_sizes = ["base"]  # Default to base model

        results = {}

        for model_size in model_sizes:
            try:
                result = await self.transcribe(
                    audio_file_path,
                    language=language,
                    engine=SpeechEngine.FASTER_WHISPER,
                    model_size=model_size,
                )
                results[f"faster_whisper_{model_size}"] = result
            except Exception as e:
                self.logger.error(f"Error with model size {model_size}: {e}")
                results[f"faster_whisper_{model_size}"] = (
                    "",
                    language,
                    {
                        "error": str(e),
                        "engine": "faster_whisper",
                        "model_size": model_size,
                    },
                )

        return results

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available engines"""
        return {
            "available_engines": self.available_engines,
            "default_engine": self.default_engine.value,
            "recommended_engines": {
                "english": "faster_whisper",
            },
            "features": {
                "faster_whisper": {
                    "multilingual": False,  # English-only to save space
                    "timestamps": True,
                    "offline": True,
                    "accuracy": "very_high",
                    "speed": "fast",
                }
            },
        }

    async def benchmark_engines(
        self, audio_file_path: str, language: str = "en-US"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different model sizes on the same audio file

        Args:
            audio_file_path: Path to test audio file
            language: Language code

        Returns:
            Dict with performance metrics for each model size
        """
        import time

        results = {}
        model_sizes = ["tiny", "base", "small", "medium", "large-v3"]

        for model_size in model_sizes:
            start_time = time.time()

            try:
                text, detected_lang, metadata = await self.transcribe(
                    audio_file_path,
                    language=language,
                    engine=SpeechEngine.FASTER_WHISPER,
                    model_size=model_size,
                )

                end_time = time.time()
                processing_time = end_time - start_time

                results[f"faster_whisper_{model_size}"] = {
                    "text": text,
                    "detected_language": detected_lang,
                    "processing_time": processing_time,
                    "confidence": metadata.get("confidence", 0.0),
                    "success": bool(text.strip()),
                    "model_size": model_size,
                    "metadata": metadata,
                }

            except Exception as e:
                results[f"faster_whisper_{model_size}"] = {
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "success": False,
                    "model_size": model_size,
                }

        return results

    async def get_best_transcription(
        self,
        audio_file_path: str,
        language: str = "en-US",
        confidence_threshold: float = 0.7,
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Get the best transcription by trying different model sizes
        English only for space optimization
        """
        self.logger.info(f"🎯 Getting best transcription for language: {language}")

        best_result = ("", language, {"confidence": 0.0, "engine": "faster_whisper"})

        # For English, base model is usually sufficient
        model_sizes = ["base", "tiny"]

        for model_size in model_sizes:
            try:
                self.logger.info(f"🧪 Trying model size: {model_size}")

                result = await self._transcribe_faster_whisper(
                    audio_file_path, language=language, model_size=model_size
                )

                text, detected_lang, metadata = result
                confidence = metadata.get("confidence", 0.0)

                self.logger.info(f"📊 Model {model_size} result:")
                self.logger.info(f"  → Text length: {len(text)} chars")
                self.logger.info(f"  → Detected language: {detected_lang}")
                self.logger.info(f"  → Confidence: {confidence:.3f}")

                # If we get a good result, return it
                if text.strip() and confidence >= confidence_threshold:
                    self.logger.info(
                        f"✅ Model {model_size} met confidence threshold ({confidence:.3f} >= {confidence_threshold})"
                    )
                    return text, detected_lang, metadata

                # Keep track of the best result so far
                if confidence > best_result[2]["confidence"]:
                    best_result = (text, detected_lang, metadata)
                    self.logger.info(
                        f"📈 New best result with model {model_size}: confidence {confidence:.3f}"
                    )

            except Exception as e:
                self.logger.warning(f"⚠️ Model size {model_size} failed: {e}")
                continue

        # If no result met the threshold, return the best one
        final_confidence = best_result[2].get("confidence", 0.0)
        self.logger.info(f"🏁 Returning best result: confidence {final_confidence:.3f}")

        return best_result

    # Function removed to save space - English only version

    async def _cleanup_files(self, *file_paths) -> None:
        """
        Clean up temporary files

        Args:
            file_paths: Paths to files that should be deleted
        """
        for path in file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp file {path}: {str(e)}")


# Utility functions for voice processing
async def create_voice_processor(
    engine: SpeechEngine = SpeechEngine.FASTER_WHISPER, enable_logging: bool = True
) -> VoiceProcessor:
    """
    Factory function to create a properly configured voice processor

    Args:
        engine: Default speech engine to use (only FASTER_WHISPER supported)
        enable_logging: Whether to enable debug logging

    Returns:
        Configured VoiceProcessor instance
    """
    if enable_logging:
        logging.basicConfig(level=logging.INFO)

    processor = VoiceProcessor(default_engine=engine)

    # Log available engines
    info = processor.get_engine_info()
    available = [name for name, avail in info["available_engines"].items() if avail]
    processor.logger.info(f"Voice processor initialized with engines: {available}")

    return processor


def get_supported_languages() -> Dict[str, List[str]]:
    """
    Get supported languages for Faster-Whisper engine
    English-only to save space

    Returns:
        Dict mapping engine name to supported language codes
    """
    return {"faster_whisper": ["en"]}
