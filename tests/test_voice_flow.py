import asyncio
import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("."))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_voice_message_flow():
    """Test voice message processing with corrected OpenRouter API"""

    logger.info("🎤 Testing voice message processing flow...")

    try:
        # Import the MessageHandlers class
        from src.handlers.message_handlers import MessageHandlers
        from src.services.rate_limiter import RateLimiter

        # Create a mock context for testing
        class MockUpdate:
            def __init__(self):
                self.message = MockMessage()

        class MockMessage:
            def __init__(self):
                self.from_user = MockUser()
                self.voice = MockVoice()
                self.chat = MockChat()

        class MockUser:
            def __init__(self):
                self.id = 123456
                self.first_name = "TestUser"

        class MockChat:
            def __init__(self):
                self.id = 123456
                self.type = "private"

        class MockVoice:
            def __init__(self):
                self.duration = 5

        class MockContext:
            def __init__(self):
                self.bot = MockBot()

        class MockBot:
            def __init__(self):
                pass

            async def get_file(self, file_id):
                class MockFile:
                    def __init__(self):
                        self.file_path = "test_audio.ogg"

                return MockFile()

            async def send_message(self, chat_id, text, parse_mode=None):
                logger.info(f"Mock bot sending message to {chat_id}: {text[:100]}...")

        # Create rate limiter and handlers
        rate_limiter = RateLimiter()
        handlers = MessageHandlers(rate_limiter=rate_limiter)

        # Test model availability check
        logger.info("Testing model availability...")

        # Check if OpenRouter models are properly loaded
        openrouter_model = handlers.get_model_by_key("llama-3.3-8b")
        if openrouter_model:
            logger.info(f"✅ Found OpenRouter model: {openrouter_model}")
        else:
            logger.error("❌ OpenRouter model not found")

        # Test direct OpenRouter API call with new max_tokens
        logger.info("Testing direct OpenRouter API call...")

        if hasattr(handlers, "openrouter_api") and handlers.openrouter_api:
            response = await handlers.openrouter_api.generate_response(
                prompt="Test voice message transcription: 'Hello, this is a test voice message.'",
                model="llama-3.3-8b",
            )

            if response:
                logger.info(f"✅ OpenRouter API test successful!")
                logger.info(f"Response length: {len(response)} characters")
                logger.info(f"Response preview: {response[:150]}...")
            else:
                logger.error("❌ OpenRouter API test failed - no response")
        else:
            logger.warning("⚠️ OpenRouter API not available in handlers")

        logger.info("🎉 Voice message flow test completed!")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


async def main():
    """Main test function"""
    logger.info("🚀 Starting voice message flow test...")
    await test_voice_message_flow()
    logger.info("🏁 Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
