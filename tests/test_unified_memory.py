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


async def test_unified_memory_context():
    """Test that voice and text messages share the same memory context"""

    logger.info("🧠 Testing unified memory context for voice and text messages...")

    try:
        # Import required modules
        from src.services.memory_context.conversation_manager import ConversationManager
        from src.services.memory_context.memory_manager import MemoryManager
        from src.services.memory_context.model_history_manager import (
            ModelHistoryManager,
        )
        from src.services.memory_context.persistence_manager import PersistenceManager

        # Create test instances
        test_user_id = 999999  # Test user ID
        test_model = "deepseek-r1-0528"

        # Initialize memory components
        persistence_manager = PersistenceManager()
        memory_manager = MemoryManager(persistence_manager)
        model_history_manager = ModelHistoryManager(memory_manager)
        conversation_manager = ConversationManager(
            memory_manager, model_history_manager
        )

        logger.info(f"Testing with user_id: {test_user_id}, model: {test_model}")

        # Test 1: Save a text message
        text_message = "Hello, my name is TestUser"
        text_response = "Hi TestUser! Nice to meet you. How can I help you today?"

        logger.info("📝 Saving text message...")
        await conversation_manager.save_message_pair(
            test_user_id, text_message, text_response, test_model
        )

        # Test 2: Save a voice message (using the same format as the updated voice handler)
        voice_message = "[Voice Message Transcribed]: What's my name?"
        voice_response = "Your name is TestUser, as you mentioned earlier."

        logger.info("🎤 Saving voice message...")
        await conversation_manager.save_message_pair(
            test_user_id, voice_message, voice_response, test_model
        )

        # Test 3: Check conversation history
        logger.info("📚 Retrieving conversation history...")
        history = await conversation_manager.get_conversation_history(
            test_user_id, max_messages=10, model=test_model
        )

        logger.info(f"Retrieved {len(history)} messages from conversation history:")

        # Verify the messages are in the same conversation
        text_found = False
        voice_found = False

        for i, message in enumerate(history):
            logger.info(
                f"  Message {i+1}: {message.get('role', 'unknown')} - {message.get('content', '')[:50]}..."
            )

            if text_message in message.get("content", ""):
                text_found = True
                logger.info("    ✅ Found original text message in history")

            if "[Voice Message Transcribed]" in message.get("content", ""):
                voice_found = True
                logger.info("    ✅ Found voice message in history")

        # Test results
        if text_found and voice_found:
            logger.info(
                "🎉 SUCCESS: Both text and voice messages are in the same conversation!"
            )
            logger.info(
                "✅ Memory context is unified - voice and text messages share the same history"
            )
        else:
            logger.error("❌ FAILED: Messages are not in the same conversation")
            logger.error(f"   Text found: {text_found}, Voice found: {voice_found}")

        # Test 4: Check model-specific conversation ID
        model_conversation_id = f"user_{test_user_id}_model_{test_model}"
        logger.info(f"🔍 Expected conversation ID: {model_conversation_id}")

        # Clean up test data
        logger.info("🧹 Cleaning up test data...")
        await conversation_manager.reset_conversation(test_user_id)

        return text_found and voice_found

    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function"""
    logger.info("🚀 Starting unified memory context test...")

    success = await test_unified_memory_context()

    if success:
        logger.info("🏆 UNIFIED MEMORY TEST PASSED!")
        logger.info("Voice messages and text messages now use the same memory context.")
    else:
        logger.error("💥 UNIFIED MEMORY TEST FAILED!")
        logger.error("There may still be issues with memory context consistency.")

    logger.info("🏁 Test completed!")


if __name__ == "__main__":
    asyncio.run(main())
