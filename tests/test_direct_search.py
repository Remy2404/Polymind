#!/usr/bin/env python3
"""
Test Direct Search Functionality
"""
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def test_direct_search():
    """Test the direct search functionality"""
    print("🔍 Testing Direct Search Functionality...")
    
    try:
        from services.direct_search_executor import DirectSearchExecutor
        
        search_executor = DirectSearchExecutor()
        
        # Test search
        print("📝 Testing search functionality...")
        result = await search_executor.search_duckduckgo("Python programming")
        print(f"✅ Search successful! Result length: {len(result)} chars")
        
        # Test company research
        print("📝 Testing company research...")
        company_result = await search_executor.research_company_fallback("Tesla")
        print(f"✅ Company research successful! Result length: {len(company_result)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_direct_search())
    print(f"🏁 Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
