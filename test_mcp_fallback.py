#!/usr/bin/env python3

"""
Test the updated SimpleMCPExecutor with fallback functionality.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.simple_mcp_executor import SimpleMCPExecutor


async def test_mcp_executor():
    """Test MCP executor with fallback."""
    print("Testing SimpleMCPExecutor with fallback...")
    
    executor = SimpleMCPExecutor()
    
    # Test search
    print("\n1. Testing Exa search (should fallback to direct search)...")
    try:
        result = await executor.execute_exa_search("What is the current time in Phnom Penh")
        print(f"Result: {result[:200]}...")
        print("✅ Search working (with fallback)")
    except Exception as e:
        print(f"❌ Search failed: {e}")
    
    # Test company research
    print("\n2. Testing company research (should fallback)...")
    try:
        result = await executor.execute_exa_company_research("Tesla")
        print(f"Result: {result[:200]}...")
        print("✅ Company research working (with fallback)")
    except Exception as e:
        print(f"❌ Company research failed: {e}")
    
    # Test URL crawling
    print("\n3. Testing URL crawling (should fallback)...")
    try:
        result = await executor.execute_exa_crawl("https://example.com")
        print(f"Result: {result[:200]}...")
        print("✅ URL crawling working (with fallback)")
    except Exception as e:
        print(f"❌ URL crawling failed: {e}")
    
    print("\n✅ All MCP executor tests completed")


if __name__ == "__main__":
    asyncio.run(test_mcp_executor())
