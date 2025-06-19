import io
import os
import logging
import asyncio
import tempfile
from typing import Tuple, Optional, Union, Dict, Any, List
from enum import Enum
from pydub import AudioSegment
import uuid
import json
import soundfile as sf
import numpy as np

# Import Faster-Whisper (only engine we'll use)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    raise ImportError("faster_whisper is required but not installed. Please install it with: pip install faster-whisper")


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
        self.logger.info(f"Available speech engines: {list(self.available_engines.keys())}")
        
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
            raise RuntimeError("Faster-Whisper is not available. Please install it with: pip install faster-whisper")
    
    async def _load_faster_whisper_model(self, model_size: str = "base") -> Optional[WhisperModel]:
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
                    model_size, 
                    device=device, 
                    compute_type=compute_type
                )
                self.logger.info(f"Faster-Whisper model loaded on {device}")
            except Exception as e:
                self.logger.error(f"Failed to load Faster-Whisper model: {e}")
                return None
        return self._faster_whisper_model

    async def download_and_convert(
        self, voice_file, user_id: str, is_khmer: bool = False
    ) -> Tuple[str, str]:
        """
        Download voice file and convert to WAV format for speech recognition

        Args:
            voice_file: Telegram file object
            user_id: User ID for file naming
            is_khmer: Flag for Khmer language processing

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
            await self._convert_to_wav(ogg_file_path, wav_file_path, is_khmer)

            return ogg_file_path, wav_file_path

        except Exception as e:
            self.logger.error(f"Error downloading/converting voice: {str(e)}")
            raise ValueError(f"Failed to process voice file: {str(e)}")

    async def _convert_to_wav(
        self, input_path: str, output_path: str, is_khmer: bool = False
    ) -> None:
        """
        Convert an audio file to WAV format optimized for speech recognition

        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file
            is_khmer: Flag for Khmer language processing
        """
        try:

            def _do_convert():
                # Load the audio file
                audio = AudioSegment.from_file(input_path)

                # Apply preprocessing based on language
                if is_khmer:
                    # Enhanced preprocessing for Khmer speech
                    from src.services.media.voice_config import VoiceConfig
                    khmer_settings = VoiceConfig.get_preprocessing_settings("km")
                    
                    # Apply normalization
                    if khmer_settings.get("normalize", True):
                        audio = audio.normalize()
                    
                    # Apply high-pass filter to remove low-frequency noise
                    high_pass = khmer_settings.get("high_pass_filter", 300)
                    audio = audio.high_pass_filter(high_pass)
                    
                    # Apply low-pass filter to focus on speech frequencies
                    low_pass = khmer_settings.get("low_pass_filter", 7000)
                    audio = audio.low_pass_filter(low_pass)
                    
                    # Volume boost for Khmer
                    volume_boost = khmer_settings.get("volume_boost", 4)
                    audio = audio + volume_boost
                    
                    # Noise reduction (simple approach)
                    if khmer_settings.get("noise_reduction", True):
                        # Simple noise gate - reduce very quiet parts
                        audio = audio.compress_dynamic_range(threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
                        
                    self.logger.info(f"🇰🇭 Applied Khmer-specific audio preprocessing")
                else:
                    # General preprocessing for better recognition
                    audio = audio.normalize()  # Normalize volume
                    audio = audio.high_pass_filter(80)  # Remove very low frequencies

                # Export as WAV with speech recognition friendly settings
                sample_rate = 16000
                if is_khmer:
                    # Ensure optimal sample rate for Khmer
                    from src.services.media.voice_config import VoiceConfig
                    khmer_settings = VoiceConfig.get_preprocessing_settings("km")
                    sample_rate = khmer_settings.get("sample_rate", 16000)
                
                audio.export(
                    output_path,
                    format="wav",
                    parameters=["-ar", str(sample_rate), "-ac", "1"],  # Sample rate, mono
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
        is_khmer: bool = False,
        engine: Optional[SpeechEngine] = None,
        model_size: str = "base"
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Transcribe a voice file to text using multiple engines

        Args:
            audio_file_path: Path to WAV file
            language: Language code for recognition
            is_khmer: Flag for Khmer language processing
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
                raise RuntimeError(f"Faster-Whisper engine not available. Please install it with: pip install faster-whisper")
            
            self.logger.info(f"Using speech engine: {engine.value}")
            
            # Transcribe using Faster-Whisper (only engine available)
            result = await self._transcribe_faster_whisper(audio_file_path, language, model_size)
            
            # Clean up temp files in background
            asyncio.create_task(self._cleanup_files(audio_file_path))
            
            return result

        except Exception as e:
            self.logger.error(f"Transcription error: {str(e)}")
            # Attempt to clean up even on error
            asyncio.create_task(self._cleanup_files(audio_file_path))
            return "", language, {"error": str(e), "engine": engine.value if engine else "unknown"}
    
    async def _transcribe_faster_whisper(
        self, audio_file_path: str, language: str, model_size: str = "base"
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Transcribe using Faster-Whisper with enhanced Khmer support"""
        # Convert language code first
        lang_code = language.split('-')[0] if '-' in language else language
        
        # For Khmer, use the enhanced processing method
        if lang_code in ["km", "kh"]:
            self.logger.info(f"🇰🇭 KHMER DETECTED - Using enhanced multi-strategy processing")
            return await self._transcribe_khmer_enhanced(audio_file_path, language)
        
        # Standard processing for non-Khmer languages
        model = await self._load_faster_whisper_model(model_size)
        if model is None:
            raise ValueError("Faster-Whisper model not available")
        
        def _do_transcribe():
            try:
                self.logger.info(f"🔍 STANDARD TRANSCRIPTION:")
                self.logger.info(f"  → Input language: {language}")
                self.logger.info(f"  → Processed lang_code: {lang_code}")
                self.logger.info(f"  → Audio file: {audio_file_path}")
                self.logger.info(f"  → Model size: {model_size}")
                
                # Standard processing for non-Khmer languages
                segments, info = model.transcribe(
                    audio_file_path,
                    language=lang_code if lang_code != "auto" else None,
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True  # Built-in VAD
                )
                
                # Collect segments
                segments_list = list(segments)
                text = " ".join([segment.text.strip() for segment in segments_list])
                
                self.logger.info(f"  → Result - Detected: {info.language}, Confidence: {info.language_probability:.3f}")
                self.logger.info(f"  → Text: {text[:100]}...")
                
                # Final logging
                self.logger.info(f"🎯 FINAL STANDARD TRANSCRIPTION RESULT:")
                self.logger.info(f"  → Final language: {info.language}")
                self.logger.info(f"  → Final confidence: {info.language_probability:.3f}")
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
                                "words": [
                                    {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                                    for w in seg.words
                                ] if hasattr(seg, 'words') and seg.words else []
                            }
                            for seg in segments_list
                        ],
                        "has_speech": True,
                        "requested_language": language,  # Add this for debugging
                        "detected_language": info.language,  # Add this for debugging
                    }
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
        engines: Optional[List[SpeechEngine]] = None
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
                    model_size=model_size
                )
                results[f"faster_whisper_{model_size}"] = result
            except Exception as e:
                self.logger.error(f"Error with model size {model_size}: {e}")
                results[f"faster_whisper_{model_size}"] = ("", language, {"error": str(e), "engine": "faster_whisper", "model_size": model_size})
        
        return results

    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about available engines"""
        return {
            "available_engines": self.available_engines,
            "default_engine": self.default_engine.value,
            "recommended_engines": {
                "english": "faster_whisper",
                "multilingual": "faster_whisper",
            },
            "features": {
                "faster_whisper": {
                    "multilingual": True,
                    "timestamps": True,
                    "offline": True,
                    "accuracy": "very_high",
                    "speed": "fast"
                }
            }
        }
    
    async def benchmark_engines(
        self, 
        audio_file_path: str, 
        language: str = "en-US"
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
                    model_size=model_size
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
                    "metadata": metadata
                }
                
            except Exception as e:
                results[f"faster_whisper_{model_size}"] = {
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                    "success": False,
                    "model_size": model_size
                }
        
        return results
    
    async def get_best_transcription(
        self, 
        audio_file_path: str, 
        language: str = "en-US",
        confidence_threshold: float = 0.7
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Get the best transcription by trying different model sizes
        Enhanced for Khmer language support with multiple strategies
        """
        lang_code = language.split('-')[0] if '-' in language else language
        
        # Special handling for Khmer
        if lang_code in ["km", "kh"]:
            self.logger.info(f"🇰🇭 KHMER TRANSCRIPTION REQUESTED - Using enhanced processing")
            return await self._transcribe_khmer_enhanced(audio_file_path, language, confidence_threshold)
        
        # Standard processing for other languages
        self.logger.info(f"🎯 Getting best transcription for language: {language}")
        
        best_result = ("", language, {"confidence": 0.0, "engine": "faster_whisper"})
        
        # Determine optimal model sizes based on language
        if lang_code in ["zh", "ja", "ko", "ar", "hi", "th", "vi"]:
            # For other high-resource languages, also prefer larger models
            model_sizes = ["large-v3", "base", "tiny"]
            self.logger.info(f"🌍 High-resource language ({lang_code}) - using larger models")
        else:
            # For common languages like English, base model is usually sufficient
            model_sizes = ["base", "large-v3", "tiny"]
        
        for model_size in model_sizes:
            try:
                self.logger.info(f"🧪 Trying model size: {model_size}")
                
                result = await self._transcribe_faster_whisper(
                    audio_file_path,
                    language=language,
                    model_size=model_size
                )
                
                text, detected_lang, metadata = result
                confidence = metadata.get("confidence", 0.0)
                
                self.logger.info(f"📊 Model {model_size} result:")
                self.logger.info(f"  → Text length: {len(text)} chars")
                self.logger.info(f"  → Detected language: {detected_lang}")
                self.logger.info(f"  → Confidence: {confidence:.3f}")
                
                # If we get a good result, return it
                if text.strip() and confidence >= confidence_threshold:
                    self.logger.info(f"✅ Model {model_size} met confidence threshold ({confidence:.3f} >= {confidence_threshold})")
                    return text, detected_lang, metadata
                
                # Keep track of the best result so far
                if confidence > best_result[2]["confidence"]:
                    best_result = (text, detected_lang, metadata)
                    self.logger.info(f"📈 New best result with model {model_size}: confidence {confidence:.3f}")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Model size {model_size} failed: {e}")
                continue
        
        # If no result met the threshold, return the best one
        final_confidence = best_result[2].get("confidence", 0.0)
        self.logger.info(f"🏁 Returning best result: confidence {final_confidence:.3f}")
        
        return best_result
    
    async def _transcribe_khmer_enhanced(
        self, 
        audio_file_path: str, 
        language: str = "km-KH",
        confidence_threshold: float = 0.3  # Lower threshold for Khmer
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Enhanced Khmer transcription with multiple fallback strategies
        Based on official Faster-Whisper documentation
        """
        from src.services.media.voice_config import VoiceConfig
        
        self.logger.info(f"🇰🇭 ENHANCED KHMER TRANSCRIPTION STARTING")
        self.logger.info(f"  → Audio file: {audio_file_path}")
        self.logger.info(f"  → Target language: {language}")
        
        results = []
        
        # Strategy 1: Force Khmer with large-v3 model (official approach)
        try:
            self.logger.info("🧪 Strategy 1: Forcing Khmer with large-v3 model")
            model = await self._load_faster_whisper_model("large-v3")
            if model:
                def _transcribe_forced():
                    # Use official parameters from documentation
                    segments, info = model.transcribe(
                        audio_file_path,
                        language="km",  # Simple language code as per docs
                        beam_size=5,
                        word_timestamps=True,
                        vad_filter=True,
                        condition_on_previous_text=False,  # Important for language detection
                        temperature=0.0  # More deterministic for Khmer
                    )
                    # Convert generator to list immediately as per docs
                    segments = list(segments)
                    return segments, info
                
                loop = asyncio.get_event_loop()
                segments_list, info = await loop.run_in_executor(None, _transcribe_forced)
                
                text = " ".join([segment.text.strip() for segment in segments_list])
                
                result1 = {
                    "text": text,
                    "language": info.language,
                    "confidence": info.language_probability,
                    "strategy": "forced_km_large_v3",
                    "segments": len(segments_list)
                }
                results.append(result1)
                
                self.logger.info(f"  → Strategy 1 Result:")
                self.logger.info(f"    - Detected: {info.language}")
                self.logger.info(f"    - Confidence: {info.language_probability:.3f}")
                self.logger.info(f"    - Text preview: {text[:100]}...")
        except Exception as e:
            self.logger.warning(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Force Khmer with higher temperature for more diversity
        try:
            self.logger.info("🧪 Strategy 2: Forcing Khmer with temperature sampling")
            model = await self._load_faster_whisper_model("large-v3")
            if model:
                def _transcribe_temp():
                    # Use temperature sampling for more diversity
                    segments, info = model.transcribe(
                        audio_file_path,
                        language="km",
                        beam_size=5,
                        word_timestamps=True,
                        vad_filter=True,
                        condition_on_previous_text=False,
                        temperature=0.2  # Slight temperature for diversity
                    )
                    segments = list(segments)
                    return segments, info
                
                loop = asyncio.get_event_loop()
                segments_list, info = await loop.run_in_executor(None, _transcribe_temp)
                
                text = " ".join([segment.text.strip() for segment in segments_list])
                
                result2 = {
                    "text": text,
                    "language": info.language,
                    "confidence": info.language_probability,
                    "strategy": "forced_km_large_v3_temp",
                    "segments": len(segments_list)
                }
                results.append(result2)
                
                self.logger.info(f"  → Strategy 2 Result:")
                self.logger.info(f"    - Detected: {info.language}")
                self.logger.info(f"    - Confidence: {info.language_probability:.3f}")
                self.logger.info(f"    - Text preview: {text[:100]}...")
        except Exception as e:
            self.logger.warning(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Auto-detect with large model (no language forcing)
        try:
            self.logger.info("🧪 Strategy 3: Auto-detection with large-v3 model")
            model = await self._load_faster_whisper_model("large-v3")
            if model:
                def _transcribe_auto():
                    # Auto-detection approach from documentation
                    segments, info = model.transcribe(
                        audio_file_path,
                        beam_size=5,  # No language parameter = auto-detect
                        word_timestamps=True,
                        vad_filter=True,
                        condition_on_previous_text=False,
                        temperature=0.0
                    )
                    segments = list(segments)
                    return segments, info
                
                loop = asyncio.get_event_loop()
                segments_list, info = await loop.run_in_executor(None, _transcribe_auto)
                
                text = " ".join([segment.text.strip() for segment in segments_list])
                
                # Check if auto-detected English is likely a false positive for Khmer
                is_false_positive = VoiceConfig.is_likely_false_english_for_khmer(text, info.language_probability)
                
                result3 = {
                    "text": text,
                    "language": info.language,
                    "confidence": info.language_probability,
                    "strategy": "auto_detect_large_v3",
                    "segments": len(segments_list),
                    "is_false_positive": is_false_positive
                }
                results.append(result3)
                
                self.logger.info(f"  → Strategy 3 Result:")
                self.logger.info(f"    - Detected: {info.language}")
                self.logger.info(f"    - Confidence: {info.language_probability:.3f}")
                self.logger.info(f"    - False positive check: {is_false_positive}")
                self.logger.info(f"    - Text preview: {text[:100]}...")
        except Exception as e:
            self.logger.warning(f"Strategy 3 failed: {e}")
        
        # Strategy 4: Base model with Khmer forced (fallback)
        try:
            self.logger.info("🧪 Strategy 4: Base model fallback with forced Khmer")
            model = await self._load_faster_whisper_model("base")
            if model:
                def _transcribe_base():
                    segments, info = model.transcribe(
                        audio_file_path,
                        language="km",
                        beam_size=3,  # Lower for speed
                        condition_on_previous_text=False,
                        vad_filter=True,
                        temperature=0.0
                    )
                    segments = list(segments)
                    return segments, info
                
                loop = asyncio.get_event_loop()
                segments_list, info = await loop.run_in_executor(None, _transcribe_base)
                
                text = " ".join([segment.text.strip() for segment in segments_list])
                
                result4 = {
                    "text": text,
                    "language": info.language,
                    "confidence": info.language_probability,
                    "strategy": "forced_km_base",
                    "segments": len(segments_list)
                }
                results.append(result4)
                
                self.logger.info(f"  → Strategy 4 Result:")
                self.logger.info(f"    - Detected: {info.language}")
                self.logger.info(f"    - Confidence: {info.language_probability:.3f}")
                self.logger.info(f"    - Text preview: {text[:100]}...")
        except Exception as e:
            self.logger.warning(f"Strategy 4 failed: {e}")

        # Analyze results and choose the best one
        self.logger.info(f"🔍 ANALYZING {len(results)} TRANSCRIPTION RESULTS:")
        
        best_result = None # Initialize best_result to None
        
        candidate_results = []
        for i, result in enumerate(results):
            self.logger.info(f"  Result {i+1} ({result['strategy']}):")
            self.logger.info(f"    - Language: {result['language']}")
            self.logger.info(f"    - Confidence: {result['confidence']:.3f}")
            self.logger.info(f"    - Text length: {len(result['text'])}")
            self.logger.info(f"    - Has text: {bool(result['text'].strip())}")
            
            # Filter out results with obvious false positives
            if result.get('is_false_positive', False):
                self.logger.warning(f"    ⚠️ Skipping likely false positive result")
                continue
            
            if result['text'].strip(): # Only consider if there's text
                candidate_results.append(result)

        if not candidate_results:
            self.logger.error(f"❌ ALL KHMER TRANSCRIPTION STRATEGIES YIELDED EMPTY TEXT OR FALSE POSITIVES.")
            return "", "km", {"error": "All transcription attempts yielded empty text or false positives", "confidence": 0.0, "strategy": "all_failed_empty", "requested_language": language, "detected_language": "km", "language_mismatch": False, "khmer_detected": False, "khmer_expected": True, "all_attempts": len(results)}

        # Enhanced selection logic with false positive detection
        # 1. Prefer Khmer results that meet the confidence threshold
        # 2. If none, prefer any Khmer result (best among them)
        # 3. If still no Khmer results, check if non-Khmer results are false positives
        # 4. Only accept high-confidence non-Khmer results that aren't false positives

        strong_khmer_results = [r for r in candidate_results if r['language'] == 'km' and r['confidence'] >= confidence_threshold]
        if strong_khmer_results:
            best_result = max(strong_khmer_results, key=lambda x: (x['confidence'], len(x['text'])))
            self.logger.info(f"✅ Selected strong Khmer result: {best_result['strategy']} (conf: {best_result['confidence']:.3f})")
        else:
            all_khmer_results = [r for r in candidate_results if r['language'] == 'km']
            if all_khmer_results:
                best_result = max(all_khmer_results, key=lambda x: (x['confidence'], len(x['text'])))
                self.logger.info(f"⚠️ Selected weaker Khmer result (below threshold {confidence_threshold}): {best_result['strategy']} (conf: {best_result['confidence']:.3f})")
            else:
                non_khmer_results = [r for r in candidate_results if r['language'] != 'km']
                if non_khmer_results:
                    # Check each non-Khmer result for false positives
                    valid_non_khmer = []
                    for result in non_khmer_results:
                        is_false_positive = VoiceConfig.is_likely_false_english_for_khmer(result['text'], result['confidence'])
                        if not is_false_positive and result['confidence'] > 0.8:
                            valid_non_khmer.append(result)
                        else:
                            self.logger.warning(f"⚠️ Rejecting non-Khmer result as likely false positive: '{result['text'][:50]}...'")
                    
                    if valid_non_khmer:
                        best_result = max(valid_non_khmer, key=lambda x: (x['confidence'], len(x['text'])))
                        self.logger.warning(f"⚠️ NO KHMER DETECTED! Using validated high-confidence non-Khmer: {best_result['strategy']} (lang: {best_result['language']}, conf: {best_result['confidence']:.3f})")
                    else:
                        self.logger.error(f"❌ No valid results found - all non-Khmer results appear to be false positives")
                        return "", "km", {"error": "No confident transcription found", "confidence": 0.0, "strategy": "no_confident_result", "requested_language": language, "detected_language": "unknown", "language_mismatch": True, "khmer_detected": False, "khmer_expected": True, "all_attempts": len(results)}
                else:
                    # This case should ideally be caught by `if not candidate_results`
                    self.logger.error(f"❌ UNEXPECTED: No candidate results left after filtering.")
                    return "", "km", {"error": "All transcription attempts failed or produced no usable text", "confidence": 0.0, "strategy": "all_failed_no_candidates", "requested_language": language, "detected_language": "km", "language_mismatch": False, "khmer_detected": False, "khmer_expected": True, "all_attempts": len(results)}

        if not best_result: # Safeguard if no result was selected
            self.logger.error(f"❌ UNEXPECTED: No best_result selected after evaluation logic.")
            return "", "km", {"error": "Internal selection error in Khmer transcription", "confidence": 0.0, "strategy": "selection_error", "requested_language": language, "detected_language": "unknown", "language_mismatch": True, "khmer_detected": False, "khmer_expected": True, "all_attempts": len(results)}
        
        # Augment the chosen best_result with additional metadata before returning
        best_result['all_attempts_tried'] = len(results) # 'results' is the list of all strategy results from this function
        best_result['requested_language_for_khmer_module'] = language # The 'km-KH' (or similar) passed to this function
        
        # Prepare final result with enhanced metadata
        final_text = best_result['text']
        final_language = best_result['language']
        final_metadata = {
            "confidence": best_result['confidence'],
            "engine": "faster_whisper_enhanced",
            "strategy": best_result['strategy'],
            "requested_language": language,
            "detected_language": final_language,
            "all_attempts": len(results),
            "khmer_detected": final_language == "km",
            "khmer_expected": True,
            "language_mismatch": final_language != "km",
            "segments": [
                {
                    "start": 0,
                    "end": 0,
                    "text": final_text,
                    "words": []
                }
            ],
            "has_speech": bool(final_text.strip())
        }
        
        self.logger.info(f"🎯 FINAL KHMER TRANSCRIPTION RESULT:")
        self.logger.info(f"  → Text length: {len(final_text)} chars")
        self.logger.info(f"  → Detected language: {final_language}")
        self.logger.info(f"  → Confidence: {best_result['confidence']:.3f}")
        self.logger.info(f"  → Strategy used: {best_result['strategy']}")
        self.logger.info(f"  → Khmer detected: {final_language == 'km'}")
        self.logger.info(f"  → Language mismatch: {final_language != 'km'}")
        
        return final_text, final_language, final_metadata

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
    engine: SpeechEngine = SpeechEngine.FASTER_WHISPER,
    enable_logging: bool = True
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
    
    Returns:
        Dict mapping engine name to supported language codes
    """
    return {
        "faster_whisper": [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", 
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", 
            "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", 
            "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", 
            "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", 
            "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", 
            "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", 
            "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    }
