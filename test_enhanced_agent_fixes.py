#!/usr/bin/env python3
"""
Test script to verify the enhanced agent works correctly with JSON parsing fixes.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.agent import EnhancedAgent, AgentDeps
from services.mcp_registry import MCPRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_agent_basic():
    """Test basic agent functionality."""
    print("🧪 Testing Enhanced Agent...")

    try:
        # Initialize agent with default model
        agent = EnhancedAgent()
        print("✅ Agent initialized successfully")

        # Test basic query
        deps = AgentDeps(user_id=12345, username="test_user")
        result = await agent.run("What is Python?", deps)

        print("✅ Basic query completed")
        print(f"📝 Summary: {result.summary[:100]}...")
        print(f"🔗 Sources: {len(result.sources)} found")

        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

async def test_agent_with_tools():
    """Test agent with tool execution."""
    print("\n🛠️ Testing Agent with Tools...")

    try:
        # Initialize agent
        agent = EnhancedAgent()
        print("✅ Agent with tools initialized")

        # Test search query that should trigger tools
        deps = AgentDeps(user_id=12345, username="test_user")
        result = await agent.run("Search for latest Python news", deps)

        print("✅ Tool-based query completed")
        print(f"📝 Summary: {result.summary[:100]}...")
        print(f"🔗 Sources: {len(result.sources)} found")

        return True

    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling capabilities."""
    print("\n🚨 Testing Error Handling...")

    try:
        # Test with empty query
        agent = EnhancedAgent()
        deps = AgentDeps(user_id=12345, username="test_user")

        result = await agent.run("", deps)
        print("✅ Empty query handled gracefully")
        print(f"📝 Response: {result.summary[:50]}...")

        # Test with very long query
        long_query = "What is " + "Python " * 1000
        result = await agent.run(long_query, deps)
        print("✅ Long query handled gracefully")
        print(f"📝 Response: {result.summary[:50]}...")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

async def test_model_fallback():
    """Test model fallback functionality."""
    print("\n🔄 Testing Model Fallback...")

    try:
        # Test with different models
        models_to_test = ["gemini-flash", "mistral-small-3.2-24b-instruct", "deepseek-r1"]

        for model in models_to_test:
            print(f"Testing model: {model}")
            agent = EnhancedAgent(preferred_model=model)
            deps = AgentDeps(user_id=12345, username="test_user")

            result = await agent.run("Hello world", deps)
            print(f"✅ Model {model} works: {result.summary[:30]}...")

        return True

    except Exception as e:
        print(f"❌ Model fallback test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Agent Tests")
    print("=" * 50)

    results = []

    # Run tests
    results.append(await test_agent_basic())
    results.append(await test_agent_with_tools())
    results.append(await test_error_handling())
    results.append(await test_model_fallback())

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")

    passed = sum(results)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! The agent is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
