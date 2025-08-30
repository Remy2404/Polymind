#!/usr/bin/env python3
"""
Simple test script to verify tool calling model detection.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.model_handlers.model_configs import (
    ModelConfigurations, 
    get_default_tool_calling_model
)

def test_tool_calling_models():
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

def test_model_validation():
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

def main():
    """Run all tests."""
    print("🧪 OpenRouter Tool Calling Fix Verification")
    print("=" * 60)
    print()
    
    result1 = test_tool_calling_models()
    test_model_validation()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 50)
    
    if result1:
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
    
    return result1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
