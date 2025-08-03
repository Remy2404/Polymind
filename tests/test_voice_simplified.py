#!/usr/bin/env python3
"""
Test script for the simplified voice processor
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import asyncio
from src.services.media.voice_processor import (
    VoiceProcessor,
    SpeechEngine,
    create_voice_processor,
    get_supported_languages,
)


async def main():
    print("🎤 Testing Simplified Voice Processor")
    print("=" * 50)

    try:
        # Test imports
        print("✅ Imports successful!")
        print(f"📋 Available engines: {[e.value for e in SpeechEngine]}")

        # Test supported languages
        languages = get_supported_languages()
        print(
            f"🌍 Supported languages: {len(languages['faster_whisper'])} languages for Faster-Whisper"
        )

        # Test processor creation
        print("\n🔧 Creating voice processor...")
        processor = await create_voice_processor()

        # Test engine info
        info = processor.get_engine_info()
        print(f"📊 Engine availability: {info['available_engines']}")
        print(f"🎯 Default engine: {info['default_engine']}")
        print(f"⭐ Recommended for English: {info['recommended_engines']['english']}")

        # Test features
        features = info["features"]["faster_whisper"]
        print(
            f"🎛️ Features: Multilingual={features['multilingual']}, "
            f"Timestamps={features['timestamps']}, "
            f"Offline={features['offline']}, "
            f"Speed={features['speed']}"
        )

        print("\n✅ All tests passed!")
        print("🚀 Voice processor is ready for use with Faster-Whisper only!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
