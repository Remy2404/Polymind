#!/usr/bin/env python3

"""
Test the DirectSearchExecutor to ensure fallback search functionality works.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.direct_search_executor import get_direct_search_executor


async def test_search():
    """Test direct search functionality."""
    print("Testing DirectSearchExecutor...")
    
    executor = get_direct_search_executor()
    
    # Test DuckDuckGo search
    print("\n1. Testing DuckDuckGo search...")
    try:
        result = await executor.search_duckduckgo("What is the current time in Phnom Penh")
        print(f"Result: {result[:200]}...")
        print("✅ DuckDuckGo search working")
    except Exception as e:
        print(f"❌ DuckDuckGo search failed: {e}")
    
    # Test company research
    print("\n2. Testing company research...")
    try:
        result = await executor.research_company_fallback("Tesla")
        print(f"Result: {result[:200]}...")
        print("✅ Company research working")
    except Exception as e:
        print(f"❌ Company research failed: {e}")
    
    # Test URL extraction
    print("\n3. Testing URL extraction...")
    try:
        result = await executor.extract_url_content_fallback("https://example.com")
        print(f"Result: {result[:200]}...")
        print("✅ URL extraction working")
    except Exception as e:
        print(f"❌ URL extraction failed: {e}")
    
    # Clean up
    await executor.close()
    print("\n✅ All tests completed")


if __name__ == "__main__":
    asyncio.run(test_search())
