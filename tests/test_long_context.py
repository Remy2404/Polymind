#!/usr/bin/env python3
"""
Test script to verify that all models can generate long context messages
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.gemini_api import GeminiAPI
from src.services.openrouter_api import OpenRouterAPI
from src.services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM
from src.services.model_handlers.simple_api_manager import SuperSimpleAPIManager
from src.services.rate_limiter import RateLimiter
from dotenv import load_dotenv
import asyncio
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_long_context_generation():
    """Test that all models can generate long context responses"""
    print("🔍 Testing Long Context Generation for All Models...")

    # Initialize APIs
    rate_limiter = RateLimiter(requests_per_minute=30)
    gemini_api = GeminiAPI(rate_limiter=rate_limiter)

    openrouter_rate_limiter = RateLimiter(requests_per_minute=20)
    openrouter_api = OpenRouterAPI(rate_limiter=openrouter_rate_limiter)

    deepseek_api = DeepSeekLLM()

    # Create API manager
    api_manager = SuperSimpleAPIManager(
        gemini_api=gemini_api, deepseek_api=deepseek_api, openrouter_api=openrouter_api
    )  # Test models from different providers - using correct model IDs
    test_models = [
        ("gemini", "Gemini"),  # Fixed: use correct model ID
        ("llama-3.3-8b", "OpenRouter"),
        (
            "deepseek-r1-zero",
            "DeepSeek",
        ),  # Fixed: use correct model ID that maps to deepseek-r1-distill-llama-70b
        ("moonshot-kimi-dev-72b", "OpenRouter - Moonshot"),
    ]

    # Long form prompt that should trigger long response
    long_prompt = """
    Write a comprehensive tutorial on Python programming for beginners. 
    Include the following sections:
    1. Introduction to Python and its history
    2. Setting up the development environment
    3. Basic syntax and data types
    4. Control structures (if statements, loops)
    5. Functions and modules
    6. Object-oriented programming basics
    7. File handling
    8. Error handling and debugging
    9. Popular libraries and frameworks
    10. Best practices and coding standards
    
    Make it detailed with code examples for each section.
    """

    for model_id, provider_name in test_models:
        print(f"\n🎯 Testing {provider_name} - Model: {model_id}")

        try:
            # Test dynamic token allocation
            response = await api_manager.chat(
                model_id=model_id,
                prompt=long_prompt,
                # Let it use dynamic max_tokens allocation
            )

            if response and not response.startswith("❌"):
                print(f"   ✅ Response generated successfully!")
                print(f"   📝 Response length: {len(response)} characters")
                print(f"   📊 Word count: {len(response.split())} words")
                print(f"   🔤 First 200 chars: {response[:200]}...")

                # Check if response is reasonably long for comprehensive content
                if len(response) > 2000:
                    print(f"   🎉 Long context generation SUCCESSFUL!")
                else:
                    print(
                        f"   ⚠️  Response seems shorter than expected for comprehensive tutorial"
                    )
            else:
                print(f"   ❌ Response generation failed: {response}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n🏁 Long Context Generation Test Complete!")

    # Clean up aiohttp sessions to prevent resource leaks
    try:
        await openrouter_api.close()
        print("🧹 Cleaned up OpenRouter API session")
    except Exception as e:
        print(f"⚠️  Error cleaning up OpenRouter API: {e}")


if __name__ == "__main__":
    asyncio.run(test_long_context_generation())
