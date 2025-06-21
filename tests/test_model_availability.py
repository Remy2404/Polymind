#!/usr/bin/env python3
"""
Test script to verify all models are available for selection
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.model_handlers.model_configs import ModelConfigurations
from src.services.model_handlers.simple_api_manager import SuperSimpleAPIManager
from src.services.gemini_api import GeminiAPI
from src.services.openrouter_api import OpenRouterAPI
from src.services.DeepSeek_R1_Distill_Llama_70B import DeepSeekLLM
from src.services.rate_limiter import RateLimiter
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def test_model_availability():
    """Test that all models are available and can be selected"""
    print("🔍 Testing Model Availability and Selection...")
    
    # Get all configured models
    all_models = ModelConfigurations.get_all_models()
    print(f"📊 Total configured models: {len(all_models)}")
    
    # Initialize APIs
    rate_limiter = RateLimiter(requests_per_minute=30)
    gemini_api = GeminiAPI(rate_limiter=rate_limiter)
    
    openrouter_rate_limiter = RateLimiter(requests_per_minute=20)
    openrouter_api = OpenRouterAPI(rate_limiter=openrouter_rate_limiter)
    
    deepseek_api = DeepSeekLLM()
    
    # Create API manager
    api_manager = SuperSimpleAPIManager(
        gemini_api=gemini_api,
        deepseek_api=deepseek_api,
        openrouter_api=openrouter_api
    )
    
    # Test key models that were previously failing
    key_models_to_test = [
        "gemini",
        "deepseek", 
        "deepseek-r1-zero",
        "llama-3.3-8b",
        "moonshot-kimi-dev-72b",
        "qwen3-32b",
        "llama4-maverick"
    ]
    
    for model_id in key_models_to_test:
        if model_id in all_models:
            model_config = all_models[model_id]
            print(f"✅ {model_id}: {model_config.display_name} ({model_config.provider.value})")
            print(f"   🎯 Max tokens: {model_config.max_tokens}")
            print(f"   {model_config.indicator_emoji} {model_config.description}")
        else:
            print(f"❌ {model_id}: Model not found in configuration")
    
    # Test a simple generation on a few key models
    test_prompt = "Explain quantum computing in one sentence."
    
    for model_id in ["gemini", "llama-3.3-8b", "moonshot-kimi-dev-72b"]:
        print(f"\n🎯 Testing quick generation with {model_id}...")
        try:
            response = await api_manager.chat(
                model_id=model_id,
                prompt=test_prompt
            )
            
            if response and not response.startswith("❌"):
                print(f"   ✅ Response: {response[:100]}...")
            else:
                print(f"   ❌ Failed: {response}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Clean up
    try:
        await openrouter_api.close()
        print("\n🧹 Cleaned up resources")
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")

if __name__ == "__main__":
    asyncio.run(test_model_availability())
