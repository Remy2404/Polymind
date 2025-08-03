import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.handlers.message_handlers import MessageHandlers
from src.handlers.text_handlers import TextHandler
from src.services.user_data_manager import UserDataManager
from src.services.openrouter_api import OpenRouterAPI
from src.services.rate_limiter import RateLimiter
from src.utils.log.telegramlog import TelegramLogger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_conversation_context():
    """Test if conversation context is properly passed to AI models"""

    # Initialize components
    import src.database as Database

    db = Database()
    await db.connect()

    user_data_manager = UserDataManager(db)
    telegram_logger = TelegramLogger()
    rate_limiter = RateLimiter()
    openrouter_api = OpenRouterAPI(rate_limiter)

    # Create text handler
    text_handler = TextHandler(
        gemini_api=None,
        user_data_manager=user_data_manager,
        openrouter_api=openrouter_api,
        deepseek_api=None,
    )

    # Create message handler
    message_handler = MessageHandlers(
        gemini_api=None,
        user_data_manager=user_data_manager,
        telegram_logger=telegram_logger,
        text_handler=text_handler,
        openrouter_api=openrouter_api,
    )

    # Test user ID
    test_user_id = 806762900

    # Initialize user
    await user_data_manager.initialize_user(test_user_id)

    # Check if conversation manager is shared
    text_conv_manager = text_handler.conversation_manager

    # Get conversation history
    conversation_history = await text_conv_manager.get_conversation_history(
        test_user_id, max_messages=5, model="deepseek-r1-zero"
    )

    print(f"\n=== CONVERSATION CONTEXT TEST ===")
    print(f"User ID: {test_user_id}")
    print(f"Retrieved {len(conversation_history)} conversation history messages")

    if conversation_history:
        print(f"\n--- Conversation History ---")
        for i, msg in enumerate(conversation_history):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            print(f"{i+1}. [{role.upper()}]: {content[:100]}...")

        # Test AI response generation with context
        test_prompt = "What is my name?"
        print(f"\n--- Testing AI Response ---")
        print(f"Prompt: {test_prompt}")
        print(f"Model: deepseek-r1-zero")
        print(f"Context messages: {len(conversation_history)}")

        # Generate response
        response = await message_handler.generate_ai_response(
            prompt=test_prompt,
            model_id="deepseek-r1-zero",
            user_id=test_user_id,
            conversation_context=conversation_history,
        )

        print(f"\nAI Response: {response}")

        # Check if response contains contextual information
        if "rami" in response.lower():
            print("\n✅ SUCCESS: AI remembered the name from context!")
        else:
            print("\n❌ ISSUE: AI did not use the conversation context effectively")

    else:
        print("\n❌ No conversation history found")

    print(f"\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_conversation_context())
