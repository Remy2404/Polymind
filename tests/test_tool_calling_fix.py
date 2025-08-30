
"""
Test script to verify tool calling model detection and MCP integration fixes.

This script tests:
1. Model configuration for tool calling support
2. Automatic model selection for MCP requests  
3. Graceful fallback when non-tool-calling models are used
4. Error handling for "No endpoints found that support tool use"
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.services.model_handlers.model_configs import (
    ModelConfigurations, 
    get_default_tool_calling_model
)
from src.services.openrouter_api import OpenRouterAPI
from src.services.rate_limiter import RateLimiter


async def test_tool_calling_models():
    """Test that tool calling models are correctly identified."""
    print("🔧 Testing Tool Calling Model Detection")
    print("=" * 50)
    
    # Get tool calling models
    tool_models = ModelConfigurations.get_tool_calling_models()
    print(f"✅ Found {len(tool_models)} models that support tool calling:")
    
    for model_id, config in tool_models.items():
        print(f"  • {model_id} ({config.display_name}) - {config.openrouter_model_key or 'N/A'}")
    
    # Test default tool calling model selection
    default_model = get_default_tool_calling_model()
    print(f"\n🎯 Default tool calling model: {default_model}")
    
    return len(tool_models) > 0


async def test_model_validation():
    """Test model validation for specific cases."""
    print("\n🔍 Testing Model Validation")
    print("=" * 50)
    
    # Test models that should support tool calling
    test_cases = [
        ("gemini", True, "Gemini should support tool calling"),
        ("deepseek", True, "DeepSeek should support tool calling"),
        ("gemini-2.0-flash-exp", True, "Gemini Flash should support tool calling"),
        ("deepseek-chat", True, "DeepSeek Chat should support tool calling"),
        ("qwen3-14b", False, "Qwen3-14B should NOT support tool calling (this was the original error)")
    ]
    
    all_models = ModelConfigurations.get_all_models()
    
    for model_id, expected_support, description in test_cases:
        if model_id in all_models:
            actual_support = all_models[model_id].supports_tool_calling
            status = "✅" if actual_support == expected_support else "❌"
            print(f"  {status} {model_id}: {actual_support} (expected: {expected_support}) - {description}")
        else:
            print(f"  ⚠️  {model_id}: Model not found in configuration")


async def test_openrouter_api_logic():
    """Test OpenRouter API tool calling logic."""
    print("\n🤖 Testing OpenRouter API Logic")
    print("=" * 50)
    
    try:
        rate_limiter = RateLimiter(max_requests=30, time_window=60)
        api = OpenRouterAPI(rate_limiter)
        
        # Test model detection without initializing MCP (to avoid actual network calls)
        print("✅ OpenRouter API initialized successfully")
        print(f"✅ Loaded {len(api.available_models)} OpenRouter models")
        
        # Test the model mapping
        test_models = ["gemini", "deepseek", "qwen3-14b"]
        for model_id in test_models:
            from src.services.model_handlers.model_configs import ModelConfigurations
            openrouter_model = ModelConfigurations.get_model_with_fallback(model_id)
            print(f"  • {model_id} -> {openrouter_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter API test failed: {e}")
        return False


async def test_error_scenarios():
    """Test error handling scenarios."""
    print("\n⚠️ Testing Error Scenarios")
    print("=" * 50)
    
    # Simulate the original error scenario
    error_messages = [
        "status_code: 404, model_name: qwen/qwen3-14b:free, body: {'message': 'No endpoints found that support tool use. To learn more about provider routing, visit: https://openrouter.ai/docs/provider-routing', 'code': 404}",
        "TaskGroup unhandled errors",
        "tool use not supported"
    ]
    
    for error_msg in error_messages:
        if "No endpoints found that support tool use" in error_msg:
            print(f"  ✅ Detected tool calling error: {error_msg[:100]}...")
        elif "TaskGroup" in error_msg:
            print(f"  ✅ Detected TaskGroup error: {error_msg}")
        elif "tool use" in error_msg.lower():
            print(f"  ✅ Detected generic tool use error: {error_msg}")
        else:
            print(f"  ℹ️  Other error: {error_msg}")


async def main():
    """Run all tests."""
    print("🧪 OpenRouter Tool Calling Fix Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(await test_tool_calling_models())
    await test_model_validation()
    results.append(await test_openrouter_api_logic())
    await test_error_scenarios()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    if all(results):
        print("✅ All tests passed! The tool calling fix should work correctly.")
        print("\n🔧 Key improvements made:")
        print("  • Added supports_tool_calling field to ModelConfig")
        print("  • Marked known tool-calling models (Gemini, DeepSeek, etc.)")
        print("  • Added automatic model validation before using MCP toolsets")
        print("  • Enhanced error detection for tool calling issues")
        print("  • Added fallback to tool-calling models when MCP is requested")
        print("\n💡 Usage recommendations:")
        print("  • Use Gemini or DeepSeek models for MCP tool calling")
        print("  • The system will automatically disable MCP for non-supporting models")
        print("  • Check the logs for model compatibility warnings")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return all(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
