#!/usr/bin/env python3

"""
Test the complete research agent functionality with fallback.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.agent import EnhancedAgent, AgentDeps


async def test_enhanced_agent():
    """Test the complete Enhanced Agent with MCP tools."""
    print("Testing EnhancedAgent with MCP fallback...")
    
    # Create agent
    agent = EnhancedAgent()
    
    # Test dependencies
    deps = AgentDeps(
        user_id=12345,
        username="test_user",
        preferred_model="gemini"  # This should trigger tool-compatible model selection
    )
    
    # Test search query
    print("\n1. Testing search query...")
    try:
        result = await agent.run(
            "What is the current time in Phnom Penh?",
            deps=deps
        )
        print(f"Summary: {result.summary[:200]}...")
        print(f"Sources: {len(result.sources)} found")
        for i, source in enumerate(result.sources[:3], 1):
            print(f"  {i}. {source}")
        print("✅ Search query working")
    except Exception as e:
        print(f"❌ Search query failed: {e}")
    
    # Test company research
    print("\n2. Testing company research...")
    try:
        result = await agent.run(
            "Research Tesla company overview and recent performance",
            deps=deps
        )
        print(f"Summary: {result.summary[:200]}...")
        print(f"Sources: {len(result.sources)} found")
        for i, source in enumerate(result.sources[:3], 1):
            print(f"  {i}. {source}")
        print("✅ Company research working")
    except Exception as e:
        print(f"❌ Company research failed: {e}")
    
    print("\n✅ Enhanced Agent tests completed")


if __name__ == "__main__":
    asyncio.run(test_enhanced_agent())
