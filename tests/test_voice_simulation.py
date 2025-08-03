import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_voice_processing_simulation():
    """Simulate the voice message processing flow"""

    print("=== VOICE MESSAGE PROCESSING SIMULATION ===")

    # Simulate the conversation history that would be retrieved
    conversation_history = [
        {"role": "user", "content": "My name is Rami. How about you, bro?"},
        {
            "role": "assistant",
            "content": "Hey Rami! Nice to meet you! I'm DeepSeek R1 — you can just call me your friendly AI assistant, bro. Whether you wanna chat, need help with something, or just feel like shooting the breeze, I'm here for it. What's on your mind?",
        },
        {"role": "user", "content": "Can you help me with Python programming?"},
        {
            "role": "assistant",
            "content": "Absolutely, Rami! I'd love to help you with Python programming. What specific area are you interested in or what problem are you trying to solve?",
        },
    ]

    # Simulate the new voice prompt
    new_voice_prompt = "Haha right now you know my name can you tell me my name please"

    print(f"Conversation History ({len(conversation_history)} messages):")
    for i, msg in enumerate(conversation_history):
        print(f"  {i + 1}. [{msg['role'].upper()}]: {msg['content']}")

    print(f"\nNew Voice Prompt: {new_voice_prompt}")

    # Check if context contains name information
    context_text = " ".join([msg.get("content", "") for msg in conversation_history])
    name_mentioned = "rami" in context_text.lower() or "name" in context_text.lower()

    print("\nContext Analysis:")
    print(f"  - Contains name information: {'✅ YES' if name_mentioned else '❌ NO'}")
    print(f"  - Context length: {len(context_text)} characters")

    # Test OpenRouter API with this exact scenario
    print("\n--- Testing OpenRouter API ---")
    try:
        from src.services.openrouter_api import OpenRouterAPI
        from src.services.rate_limiter import RateLimiter

        rate_limiter = RateLimiter()
        openrouter_api = OpenRouterAPI(rate_limiter)

        response = await openrouter_api.generate_response_with_model_key(
            prompt=new_voice_prompt,
            openrouter_model_key="deepseek/deepseek-r1-0528:free",
            context=conversation_history,
            max_tokens=200,
        )

        print(f"AI Response: {response}")

        # Analyze response
        if response:
            if "rami" in response.lower():
                print("\n✅ SUCCESS: AI correctly identified the name!")
            else:
                print("\n❌ ISSUE: AI did not mention the name 'Rami'")
                print("Possible causes:")
                print("  1. Model not effectively using conversation context")
                print("  2. System message not emphasizing context usage")
                print("  3. Context format issues")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

    print("\n=== Simulation Complete ===")


if __name__ == "__main__":
    asyncio.run(test_voice_processing_simulation())
