import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_openrouter_context_format():
    """Test the OpenRouter API context format"""

    print("=== TESTING OPENROUTER CONTEXT FORMAT ===")

    # Sample conversation history as it would be retrieved from memory
    sample_context = [
        {"role": "user", "content": "Hello, my name is Rami"},
        {
            "role": "assistant",
            "content": "Hello Rami! Nice to meet you. How can I help you today?",
        },
        {"role": "user", "content": "Can you help me with Python programming?"},
        {
            "role": "assistant",
            "content": "Of course, Rami! I'd be happy to help you with Python programming. What specific topic or problem would you like to work on?",
        },
    ]

    # Test prompt
    test_prompt = "What is my name?"

    print(f"Context messages: {len(sample_context)}")
    for i, msg in enumerate(sample_context):
        print(f"{i+1}. [{msg['role'].upper()}]: {msg['content']}")

    print(f"\nNew prompt: {test_prompt}")

    # Test OpenRouter API call format
    from src.services.openrouter_api import OpenRouterAPI
    from src.services.rate_limiter import RateLimiter

    rate_limiter = RateLimiter()
    openrouter_api = OpenRouterAPI(rate_limiter)

    try:
        response = await openrouter_api.generate_response(
            prompt=test_prompt,
            context=sample_context,
            model="deepseek-r1-zero",
            max_tokens=150,
        )

        print(f"\nOpenRouter Response: {response}")

        if response and "rami" in response.lower():
            print("\n✅ SUCCESS: OpenRouter API used context correctly!")
        else:
            print("\n❌ ISSUE: OpenRouter API may not be using context effectively")
            print("This could be due to:")
            print("1. Model not following conversation context")
            print("2. Context format issues")
            print("3. API rate limiting or errors")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_openrouter_context_format())
