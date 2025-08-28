#!/usr/bin/env python3
"""
Test Agent with Native Gemini API
"""
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.agent import EnhancedAgent, AgentDeps

async def test_native_gemini_agent():
    """Test the enhanced agent with native Gemini API"""
    print("🤖 Testing Enhanced Agent with Native Gemini...")
    
    try:
        # Initialize the enhanced agent (should use native Gemini)
        print("📝 Initializing Enhanced Agent...")
        agent = EnhancedAgent()
        
        # Test a simple query
        print("🔍 Testing agent query...")
        deps = AgentDeps(user_id=123, username="test_user")
        
        result = await agent.run("What is Python programming language?", deps=deps)
        
        print(f"✅ Query successful!")
        print(f"📊 Summary: {result.summary[:100]}...")
        print(f"🔗 Sources: {len(result.sources)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_native_gemini_agent())
    print(f"🏁 Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
