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


async def test_shared_memory_context():
    """Test that voice and text messages use the same conversation context"""

    logger.info("🔍 Testing shared memory context between voice and text messages...")

    try:
        # Import required classes
        from src.handlers.message_handlers import MessageHandlers
        from src.handlers.text_handlers import TextHandler
        from src.services.rate_limiter import RateLimiter

        # Mock some dependencies
        logger.info("Creating mock dependencies...")
        rate_limiter = RateLimiter()

        # Mock managers (simplified for testing)
        class MockGeminiAPI:
            pass

        class MockUserDataManager:
            async def initialize_user(self, user_id):
                pass

            async def get_user_settings(self, user_id):
                return {"active_model": "deepseek-r1-0528"}

            async def get_user_preference(self, user_id, pref, default):
                return default

        class MockTelegramLogger:
            def log_message(self, message, user_id):
                logger.info(f"TelegramLogger: {message}")

        # Create instances
        gemini_api = MockGeminiAPI()
        user_data_manager = MockUserDataManager()
        telegram_logger = MockTelegramLogger()

        # Create text handler
        text_handler = TextHandler(
            gemini_api=gemini_api,
            user_data_manager=user_data_manager,
        )

        # Create message handlers with text handler
        message_handlers = MessageHandlers(
            gemini_api=gemini_api,
            user_data_manager=user_data_manager,
            telegram_logger=telegram_logger,
            text_handler=text_handler,
        )

        # Test: Check if both handlers reference the same conversation manager
        logger.info("🔍 Checking conversation manager instances...")

        # Check text handler conversation manager
        if hasattr(text_handler, "conversation_manager"):
            logger.info("✅ TextHandler has conversation_manager")
            text_conv_manager = text_handler.conversation_manager
        else:
            logger.error("❌ TextHandler missing conversation_manager")
            return

        # Check if voice handler would use the same conversation manager
        if hasattr(message_handlers.text_handler, "conversation_manager"):
            voice_conv_manager = message_handlers.text_handler.conversation_manager
            logger.info(
                "✅ Voice handler can access TextHandler's conversation_manager"
            )
        else:
            logger.error(
                "❌ Voice handler cannot access TextHandler's conversation_manager"
            )
            return

        # Test if they're the same instance
        if text_conv_manager is voice_conv_manager:
            logger.info(
                "✅ SUCCESS: Voice and text handlers share the SAME ConversationManager instance!"
            )
        else:
            logger.error(
                "❌ FAILURE: Voice and text handlers use DIFFERENT ConversationManager instances!"
            )

        # Test memory manager sharing
        if hasattr(text_conv_manager, "memory_manager") and hasattr(
            voice_conv_manager, "memory_manager"
        ):
            if text_conv_manager.memory_manager is voice_conv_manager.memory_manager:
                logger.info("✅ SUCCESS: Both handlers share the same MemoryManager!")
            else:
                logger.error(
                    "❌ FAILURE: Handlers use different MemoryManager instances!"
                )

        logger.info("✅ Shared memory context test completed!")

    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()


async def main():
    """Main test function"""
    logger.info("🚀 Starting shared memory context test...")
    await test_shared_memory_context()
    logger.info("🏁 Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
