import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.media.voice_config import (
    VoiceConfig,
    load_config_from_env,
    voice_config,
)


def test_hardcoded_config():
    """Test that voice configuration is properly hardcoded"""

    print("🎯 Testing Hardcoded Voice Configuration")
    print("=" * 50)
    # Test the VoiceConfig.from_env() method
    config_from_env = VoiceConfig.from_env()

    print("\n📋 VoiceConfig.from_env() Results:")
    print("-" * 30)

    expected_values = {
        "enabled": True,
        "default_engine": "faster_whisper",
        "quality": "medium",
        "max_file_size_mb": 50,
        "timeout_seconds": 300,
        "enable_vad": True,
        "cache_models": True,
        "language_detection": True,
    }

    all_correct = True

    for key, expected in expected_values.items():
        actual = config_from_env.get(key)
        status = "✅" if actual == expected else "❌"

        if actual != expected:
            all_correct = False

        print(f"{status} {key}: {actual} (expected: {expected})")

    # Test engine preferences
    print("\n🌍 Engine Preferences:")
    print("-" * 20)

    engine_prefs = config_from_env.get("engine_preferences", {})
    expected_languages = ["en", "es", "fr", "zh", "ja", "ko", "ru", "ar"]

    for lang in expected_languages:
        prefs = engine_prefs.get(lang, [])
        expected_engine = ["faster_whisper"]
        status = "✅" if prefs == expected_engine else "❌"

        if prefs != expected_engine:
            all_correct = False

        print(f"{status} {lang}: {prefs}")

    # Test load_config_from_env() function
    print("\n📋 load_config_from_env() Results:")
    print("-" * 30)

    direct_config = load_config_from_env()

    direct_expected = {
        "max_file_size_mb": 50,
        "timeout_seconds": 300,
        "enable_vad": True,
        "cache_models": True,
        "log_level": "INFO",
    }

    for key, expected in direct_expected.items():
        actual = direct_config.get(key)
        status = "✅" if actual == expected else "❌"

        if actual != expected:
            all_correct = False

        print(f"{status} {key}: {actual} (expected: {expected})")

    # Test default quality enum
    quality_obj = direct_config.get("default_quality")
    expected_quality = "medium"
    actual_quality = (
        quality_obj.value if hasattr(quality_obj, "value") else str(quality_obj)
    )
    status = "✅" if actual_quality == expected_quality else "❌"

    if actual_quality != expected_quality:
        all_correct = False

    print(f"{status} default_quality: {actual_quality} (expected: {expected_quality})")

    print("\n" + "=" * 50)

    if all_correct:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Voice configuration is properly hardcoded")
        print("✅ No dependency on .env file")
        print("✅ All settings match requirements")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Configuration may not be properly hardcoded")

    print("\n💡 Benefits of Hardcoded Configuration:")
    print("   • No dependency on .env file")
    print("   • Configuration is guaranteed to be consistent")
    print("   • Faster-Whisper is always enabled and configured")
    print("   • Voice processing will work even if .env is missing")

    return all_correct


def test_without_env():
    """Test that configuration works even without .env variables"""

    print("\n🧪 Testing Configuration Without .env Variables")
    print("=" * 55)

    # Temporarily clear relevant environment variables
    env_vars_to_clear = [
        "VOICE_PROCESSING_ENABLED",
        "VOICE_DEFAULT_ENGINE",
        "VOICE_QUALITY",
        "VOICE_MAX_FILE_SIZE_MB",
        "VOICE_TIMEOUT_SECONDS",
        "VOICE_ENABLE_VAD",
        "VOICE_CACHE_MODELS",
        "VOICE_LANGUAGE_DETECTION",
    ]

    # Also clear language preferences
    lang_vars = [
        f"VOICE_ENGINE_PREFERENCES_{lang.upper()}"
        for lang in ["EN", "ES", "FR", "ZH", "JA", "KO", "RU", "AR"]
    ]
    env_vars_to_clear.extend(lang_vars)

    original_values = {}
    for var in env_vars_to_clear:
        original_values[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    try:
        # Test configuration loading without env vars
        config = VoiceConfig.from_env()

        # Key tests
        tests = [
            ("Voice processing enabled", config.get("enabled") == True),
            ("Uses Faster-Whisper", config.get("default_engine") == "faster_whisper"),
            ("Medium quality", config.get("quality") == "medium"),
            ("50MB file limit", config.get("max_file_size_mb") == 50),
            ("300s timeout", config.get("timeout_seconds") == 300),
            ("VAD enabled", config.get("enable_vad") == True),
            ("Model caching enabled", config.get("cache_models") == True),
            ("Language detection enabled", config.get("language_detection") == True),
        ]

        all_passed = True
        for test_name, result in tests:
            status = "✅" if result else "❌"
            if not result:
                all_passed = False
            print(f"{status} {test_name}")

        if all_passed:
            print("\n🎉 SUCCESS! Configuration works without .env variables")
        else:
            print("\n❌ FAILED! Some settings still depend on .env")

    finally:
        # Restore original environment variables
        for var, value in original_values.items():
            if value is not None:
                os.environ[var] = value

    return all_passed


if __name__ == "__main__":
    print("🔧 Voice Configuration Hardcoding Test")
    print("=" * 60)

    # Run tests
    test1_passed = test_hardcoded_config()
    test2_passed = test_without_env()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! Voice configuration is properly hardcoded.")
    else:
        print("❌ Some tests failed. Check the configuration.")
    print("=" * 60)
