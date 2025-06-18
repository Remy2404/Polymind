"""
Simple test to verify the new Gemini 2.0 Flash implementation works
"""

import sys
import os
import asyncio
import io

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def test_imports():
    """Test that all imports work correctly"""
    try:
        from src.services.gemini_api import (
            GeminiAPI,
            MediaType,
            MediaInput,
            ProcessingResult,
            create_image_input,
            create_document_input,
            create_text_input,
        )

        print("✅ Gemini API imports successful")

        from src.services.multimodal_processor import TelegramMultimodalProcessor

        print("✅ Multimodal processor import successful")

        from src.utils.docgen.document_processor import DocumentProcessor

        print("✅ Document processor import successful")

        from src.services.rate_limiter import RateLimiter

        print("✅ Rate limiter import successful")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_media_creation():
    """Test creating media inputs"""
    try:
        from src.services.gemini_api import (
            create_image_input,
            create_document_input,
            create_text_input,
        )

        # Test document input
        doc_content = "This is a test document"
        doc_data = io.BytesIO(doc_content.encode("utf-8"))
        doc_input = create_document_input(doc_data, "test.txt")
        print(
            f"✅ Document input created: {doc_input.type.value}, {doc_input.mime_type}"
        )

        # Test text input
        text_input = create_text_input("This is test text")
        print(f"✅ Text input created: {text_input.type.value}, {text_input.mime_type}")

        return True

    except Exception as e:
        print(f"❌ Media creation error: {e}")
        return False


async def test_basic_functionality():
    """Test basic functionality (without actual API calls)"""
    try:
        from src.services.gemini_api import GeminiAPI
        from src.services.rate_limiter import RateLimiter
        from src.utils.docgen.document_processor import DocumentProcessor

        # Test initialization
        rate_limiter = RateLimiter(requests_per_minute=60)
        gemini_api = GeminiAPI(rate_limiter)
        print("✅ GeminiAPI initialized successfully")

        # Test document processor initialization
        doc_processor = DocumentProcessor(gemini_api)
        print("✅ DocumentProcessor initialized successfully")

        # Test helper methods
        is_supported = doc_processor.is_supported_document("test.pdf")
        print(f"✅ Document support check: PDF supported = {is_supported}")

        doc_info = doc_processor.get_document_info("example.py")
        print(f"✅ Document info: {doc_info}")

        return True

    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing New Gemini 2.0 Flash Implementation")
    print("=" * 50)

    all_passed = True

    # Test imports
    print("\n📦 Testing Imports...")
    if not test_imports():
        all_passed = False

    # Test media creation
    print("\n🎬 Testing Media Creation...")
    if not test_media_creation():
        all_passed = False

    # Test basic functionality
    print("\n⚙️ Testing Basic Functionality...")
    try:
        if not asyncio.run(test_basic_functionality()):
            all_passed = False
    except Exception as e:
        print(f"❌ Async test error: {e}")
        all_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ The new implementation is ready to use!")
        print("\nNext steps:")
        print("1. Update your .env file with GEMINI_API_KEY")
        print("2. Test with real Telegram messages")
        print("3. Enjoy combined multimodal processing! 🚀")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
