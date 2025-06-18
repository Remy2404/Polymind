#!/usr/bin/env python3
"""
Test script to verify response formatting
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.handlers.response_formatter import ResponseFormatter

async def test_voice_response_formatting():
    """Test voice response formatting"""
    
    print("=== TESTING VOICE RESPONSE FORMATTING ===")
    
    formatter = ResponseFormatter()
    
    # Sample voice response text as it would be generated
    voice_response = """🎤 **Voice Response:**

_Based on our conversation..._

Absolutely! Based on our current conversation — yes, your name is **Rami**! 😊

Remember:
✅ I'll recall your name **throughout this chat session**, but I don't save personal info once the conversation ends.
✅ I'm here to help however you need — writing, explaining, brainstorming, or just talking.

What would you like to do today, Rami? 💬✨"""

    print("Original text:")
    print(voice_response)
    print("\n" + "="*50 + "\n")
    
    # Test different formatting methods
    print("1. Testing format_telegram_markdown:")
    try:
        result1 = await formatter.format_telegram_markdown(voice_response)
        print("✅ SUCCESS:")
        print(result1)
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "-"*30 + "\n")
    
    print("2. Testing _convert_markdown_to_html:")
    try:
        result2 = formatter._convert_markdown_to_html(voice_response)
        print("✅ SUCCESS:")
        print(result2)
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n" + "-"*30 + "\n")
    
    print("3. Testing escape_markdown_text:")
    try:
        result3 = await formatter.escape_markdown_text(voice_response)
        print("✅ SUCCESS:")
        print(result3)
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_voice_response_formatting())
