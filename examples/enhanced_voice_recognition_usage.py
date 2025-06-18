"""
Simplified Voice Processor Usage Examples
Demonstrates speech recognition using Faster-Whisper engine
"""

import asyncio
import logging
from pathlib import Path
from src.services.media.voice_processor import (
    VoiceProcessor,
    SpeechEngine,
    create_voice_processor,
    get_supported_languages,
)
from src.services.media.voice_config import VoiceQuality, VoiceConfig


async def basic_usage_example():
    """Basic usage with Faster-Whisper engine"""
    print("=== Basic Voice Recognition Example (Faster-Whisper) ===")

    # Create voice processor with Faster-Whisper
    processor = await create_voice_processor(engine=SpeechEngine.FASTER_WHISPER)

    # Show engine info
    info = processor.get_engine_info()
    print(f"Available engines: {list(info['available_engines'].keys())}")
    print(f"Default engine: {info['default_engine']}")

    # Example: Process a voice file (you would replace with actual file path)
    # audio_file = "path/to/your/audio.wav"
    # text, language, metadata = await processor.transcribe(audio_file)
    # print(f"Transcribed: '{text}' (Language: {language}, Engine: {metadata['engine']})")


async def quality_comparison():
    """Compare results from different quality settings"""
    print("\n=== Quality Settings Comparison Example ===")

    # Example audio file path (replace with actual path)
    audio_file = "test_audio.wav"  # You would use a real file

    if Path(audio_file).exists():
        print("Comparison Results with different quality settings:")

        for quality in [VoiceQuality.FAST, VoiceQuality.BALANCED, VoiceQuality.HIGH]:
            config = VoiceConfig()
            config.quality = quality

            processor = await create_voice_processor(
                engine=SpeechEngine.FASTER_WHISPER, config=config
            )
            text, lang, metadata = await processor.transcribe(audio_file)
            confidence = metadata.get("confidence", 0.0)
            print(f"{quality.value:15}: '{text}' (confidence: {confidence:.2f})")
    else:
        print("No test audio file available for comparison")


async def language_specific_example():
    """Examples for different languages with Faster-Whisper"""
    print("\n=== Language-Specific Processing (Faster-Whisper) ===")

    processor = await create_voice_processor(engine=SpeechEngine.FASTER_WHISPER)

    # Show supported languages
    languages = get_supported_languages()
    print("Supported languages:")
    for engine, langs in languages.items():
        print(f"{engine}: {len(langs)} languages")

    # Language-specific configurations
    test_languages = ["en-US", "es-ES", "zh-CN", "ja-JP", "ar-SA"]

    for lang in test_languages:
        engine = processor.get_recommended_engine(lang)
        print(f"{lang:6}: Engine={engine.value}")
        # Note: Preprocessing settings may vary based on implementation


async def quality_settings_example():
    """Demonstrate different quality settings"""
    print("\n=== Quality Settings Example ===")

    # Create processors with different quality settings
    for quality in [VoiceQuality.LOW, VoiceQuality.MEDIUM, VoiceQuality.HIGH]:
        model_size = VoiceConfig.get_model_size(quality)
        print(f"{quality.value:6} quality: Model size = {model_size}")

        # You would create processor and set model size accordingly
        # processor = await create_voice_processor()
        # result = await processor.transcribe(audio_file, model_size=model_size)


async def advanced_features_example():
    """Demonstrate advanced features"""
    print("\n=== Advanced Features Example ===")

    processor = await create_voice_processor()

    # Example: Get best transcription (tries multiple engines)
    audio_file = "test_audio.wav"  # Replace with actual file

    if Path(audio_file).exists():
        # Get the best transcription with confidence threshold
        text, lang, metadata = await processor.get_best_transcription(
            audio_file, language="en-US", confidence_threshold=0.7
        )

        print(f"Best result: '{text}'")
        print(f"Engine used: {metadata.get('engine')}")
        print(f"Confidence: {metadata.get('confidence', 0):.2f}")

        # Benchmark engines
        benchmark = await processor.benchmark_engines(audio_file)
        print("\nBenchmark results:")
        for engine, metrics in benchmark.items():
            if metrics.get("success", False):
                print(
                    f"{engine:15}: {metrics['processing_time']:.2f}s, "
                    f"confidence: {metrics.get('confidence', 0):.2f}"
                )


async def telegram_integration_example():
    """Example for Telegram bot integration"""
    print("\n=== Telegram Integration Example ===")

    processor = await create_voice_processor(engine=SpeechEngine.FASTER_WHISPER)

    # Simulate Telegram voice message processing
    async def process_telegram_voice(voice_file, user_id, language_hint="en-US"):
        """Process a Telegram voice message"""
        try:
            # Download and convert the voice file
            ogg_path, wav_path = await processor.download_and_convert(
                voice_file, user_id, is_khmer=(language_hint == "km-KH")
            )

            # Transcribe with the best available engine
            text, detected_lang, metadata = await processor.get_best_transcription(
                wav_path, language=language_hint
            )

            if text.strip():
                # Send transcription back to user
                response = f"🎤 **Voice Message Transcribed:**\n\n{text}"

                # Add metadata if useful
                engine = metadata.get("engine", "unknown")
                confidence = metadata.get("confidence", 0.0)

                if confidence > 0:
                    response += f"\n\n_Engine: {engine}, Confidence: {confidence:.1%}_"

                return response
            else:
                return "❌ Sorry, I couldn't understand the voice message. Please try speaking more clearly."

        except Exception as e:
            logging.error(f"Voice processing error: {e}")
            return "❌ Sorry, there was an error processing your voice message."

    print("Telegram integration ready!")
    print("Use process_telegram_voice(voice_file, user_id, language_hint) in your bot")


async def performance_optimization_example():
    """Performance optimization techniques"""
    print("\n=== Performance Optimization ===")

    # For high-volume applications
    processor = await create_voice_processor(engine=SpeechEngine.FASTER_WHISPER)

    # Pre-load models to reduce first-time latency
    print("Pre-loading models...")
    await processor._load_faster_whisper_model("base")

    # Use appropriate quality based on use case
    def get_optimal_settings(file_size_mb, language, priority="speed"):
        """Get optimal settings based on requirements"""
        if priority == "speed":
            quality = VoiceQuality.LOW if file_size_mb < 5 else VoiceQuality.MEDIUM
            engine = SpeechEngine.FASTER_WHISPER
        elif priority == "accuracy":
            quality = VoiceQuality.HIGH
            engine = SpeechEngine.WHISPER
        else:  # balanced
            quality = VoiceConfig.get_recommended_quality(language, file_size_mb)
            engine = processor.get_recommended_engine(language)

        return {
            "quality": quality,
            "engine": engine,
            "model_size": VoiceConfig.get_model_size(quality),
        }

    # Example optimization scenarios
    scenarios = [
        {"file_size": 1.5, "language": "en", "priority": "speed"},
        {"file_size": 8.0, "language": "zh", "priority": "accuracy"},
        {"file_size": 3.0, "language": "es", "priority": "balanced"},
    ]

    for scenario in scenarios:
        settings = get_optimal_settings(
            scenario["file_size"], scenario["language"], scenario["priority"]
        )
        print(
            f"File: {scenario['file_size']}MB, Lang: {scenario['language']}, "
            f"Priority: {scenario['priority']} => {settings}"
        )


async def error_handling_example():
    """Demonstrate robust error handling"""
    print("\n=== Error Handling and Fallbacks ===")

    processor = await create_voice_processor()

    async def robust_transcribe(audio_file, language="en-US"):
        """Robust transcription with multiple fallbacks"""
        try:
            # Try the best transcription first
            text, lang, metadata = await processor.get_best_transcription(
                audio_file, language=language, confidence_threshold=0.6
            )

            if text.strip():
                return {
                    "success": True,
                    "text": text,
                    "language": lang,
                    "engine": metadata.get("engine"),
                    "confidence": metadata.get("confidence", 0.0),
                }

            # If no good result, try all engines
            results = await processor.transcribe_with_multiple_engines(
                audio_file, language
            )

            # Find the best result among all engines
            best_result = None
            best_score = 0

            for engine, (text, lang, metadata) in results.items():
                if text.strip():
                    score = len(text) * metadata.get("confidence", 0.1)
                    if score > best_score:
                        best_score = score
                        best_result = {
                            "success": True,
                            "text": text,
                            "language": lang,
                            "engine": engine,
                            "confidence": metadata.get("confidence", 0.0),
                            "fallback": True,
                        }

            return best_result or {
                "success": False,
                "error": "No engine could transcribe the audio",
                "attempted_engines": list(results.keys()),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "fallback_available": True}

    print("Robust transcription function ready!")
    print("It tries multiple engines and provides detailed error information")


async def main():
    """Run all examples"""
    print("🎤 Simplified Voice Processor Examples (Faster-Whisper)")
    print("=" * 60)

    await basic_usage_example()
    await quality_comparison()
    await language_specific_example()
    await quality_settings_example()
    await advanced_features_example()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("\nTo use in your Telegram bot:")
    print("1. Install the required packages: pip install -r requirements.txt")
    print("2. Import: from src.services.media.voice_processor import VoiceProcessor")
    print("3. Create processor: processor = await create_voice_processor()")
    print(
        "4. Process voice: text, lang, metadata = await processor.transcribe(audio_file)"
    )


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run examples
    asyncio.run(main())
