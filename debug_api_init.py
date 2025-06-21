#!/usr/bin/env python3
"""
Debug script to test API initialization and availability
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

load_dotenv()

async def test_api_initialization():
    """Test if all APIs can be initialized properly"""
    print("🔍 Testing API Initialization...")
    
    # Test individual API initialization
    print("\n1. Testing individual API initialization:")
    
    try:
        # Gemini API
        rate_limiter = RateLimiter(requests_per_minute=30)
        gemini_api = GeminiAPI(rate_limiter=rate_limiter)
        print(f"   ✅ Gemini API initialized: {gemini_api is not None}")
        print(f"      Type: {type(gemini_api)}")
    except Exception as e:
        print(f"   ❌ Gemini API failed: {e}")
        gemini_api = None
    
    try:
        # OpenRouter API
        openrouter_rate_limiter = RateLimiter(requests_per_minute=20)
        openrouter_api = OpenRouterAPI(rate_limiter=openrouter_rate_limiter)
        print(f"   ✅ OpenRouter API initialized: {openrouter_api is not None}")
        print(f"      Type: {type(openrouter_api)}")
        print(f"      API Key available: {os.getenv('OPENROUTER_API_KEY') is not None}")
        print(f"      API Key length: {len(os.getenv('OPENROUTER_API_KEY', ''))}")
    except Exception as e:
        print(f"   ❌ OpenRouter API failed: {e}")
        openrouter_api = None
    
    try:
        # DeepSeek API
        deepseek_api = DeepSeekLLM()
        print(f"   ✅ DeepSeek API initialized: {deepseek_api is not None}")
        print(f"      Type: {type(deepseek_api)}")
    except Exception as e:
        print(f"   ❌ DeepSeek API failed: {e}")
        deepseek_api = None
    
    # Test SuperSimpleAPIManager initialization
    print("\n2. Testing SuperSimpleAPIManager initialization:")
    
    try:
        api_manager = SuperSimpleAPIManager(
            gemini_api=gemini_api,
            deepseek_api=deepseek_api,
            openrouter_api=openrouter_api
        )
        print(f"   ✅ SuperSimpleAPIManager initialized: {api_manager is not None}")
        
        # Check internal API storage
        print(f"   🔍 Internal API storage:")
        for provider, api_instance in api_manager.apis.items():
            print(f"      {provider.value}: {api_instance is not None} ({type(api_instance)})")
          # Get model list
        models = list(api_manager.models.keys())
        print(f"   📊 Available models: {len(models)}")
          # Test with a specific OpenRouter model
        print("\n3. Testing chat with OpenRouter model:")
        test_model = "llama-3.3-8b"  # Available OpenRouter model
        if test_model in models:
            print(f"   🎯 Testing model: {test_model}")
            try:
                response = await api_manager.chat(
                    model_id=test_model,
                    prompt="Hello, this is a test message. Please respond briefly."
                )
                print(f"   📝 Response: {response[:100]}...")
            except Exception as e:
                print(f"   ❌ Chat test failed: {e}")
        else:
            print(f"   ⚠️  Model {test_model} not available in model list")
            print(f"   Available OpenRouter models: {[m for m in models if 'llama' in m][:5]}")
        
    except Exception as e:
        print(f"   ❌ SuperSimpleAPIManager failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_api_initialization())
