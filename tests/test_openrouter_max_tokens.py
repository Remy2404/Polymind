
import asyncio
import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from src.services.openrouter_api import OpenRouterAPI
from src.services.rate_limiter import RateLimiter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_openrouter_max_tokens():
    """Test that OpenRouter API uses reasonable max_tokens values"""
    
    # Create rate limiter and OpenRouter API instance
    rate_limiter = RateLimiter()
    openrouter_api = OpenRouterAPI(rate_limiter)
    
    try:
        logger.info("Testing OpenRouter API with new max_tokens settings...")
        
        # Test with a simple prompt using the default max_tokens (should be 4096 now)
        test_prompt = "Hello, can you tell me your name and capabilities?"
        test_model = "llama-3.3-8b"  # Use a free model for testing
        
        logger.info(f"Testing model: {test_model}")
        logger.info(f"Test prompt: {test_prompt}")
        
        # Test the generate_response method (should use max_tokens=4096 by default)
        response = await openrouter_api.generate_response(
            prompt=test_prompt,
            model=test_model,
        )
        
        if response:
            logger.info(f"✅ Success! Received response with default max_tokens=4096")
            logger.info(f"Response length: {len(response)} characters")
            logger.info(f"Response preview: {response[:200]}...")
        else:
            logger.error("❌ Failed to get response from OpenRouter API")
            
        # Test with explicit max_tokens
        logger.info("\nTesting with explicit max_tokens=2048...")
        response2 = await openrouter_api.generate_response(
            prompt="Write a short poem about AI",
            model=test_model,
            max_tokens=2048
        )
        
        if response2:
            logger.info(f"✅ Success! Received response with max_tokens=2048")
            logger.info(f"Response length: {len(response2)} characters")
        else:
            logger.error("❌ Failed to get response with explicit max_tokens")
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        await openrouter_api.close()
        logger.info("Closed OpenRouter API session")

async def main():
    """Main test function"""
    logger.info("🚀 Starting OpenRouter max_tokens test...")
    await test_openrouter_max_tokens()
    logger.info("🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())
