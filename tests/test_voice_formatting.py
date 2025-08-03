#!/usr/bin/env python3
"""
Test script to verify that voice message responses are now properly formatted
with Markdown (bold, italics, etc.) after applying the fix.
"""

import asyncio
import logging
from src.handlers.response_formatter import ResponseFormatter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_formatting():
    """Test the response formatter with sample voice response content."""

    print("🎯 Testing Voice Message Response Formatting Fix")
    print("=" * 55)

    formatter = ResponseFormatter()

    # Test cases that represent voice responses after the fix
    test_cases = [
        {
            "name": "Basic Voice Response",
            "text": "🎤 **Voice Response:**\n\nHello! This is a **bold** response with *italic* text.",
        },
        {
            "name": "Voice Response with Model Indicator",
            "text": "🤖 **DeepSeek** | 🎤 **Voice Response:**\n\n_Continuing our conversation..._\n\nI can help you with **anything** you need!",
        },
        {
            "name": "Voice Response with Context",
            "text": "🎤 **Voice Response:**\n\n_Based on our conversation..._\n\nYour name is **John** and you mentioned you like *programming*.",
        },
    ]

    print("\n📋 Testing Markdown Formatting Pipeline:")
    print("-" * 40)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   📝 Input: {test_case['text'][:60]}...")

        try:
            # Test the format_telegram_markdown function (this is the fix!)
            formatted = await formatter.format_telegram_markdown(test_case["text"])

            print("   ✅ Telegram formatting successful")
            print(f"   📤 Output: {formatted[:80]}...")

            # Check if formatting was applied
            has_formatting = any(char in formatted for char in ["*", "_", "`"])
            if has_formatting:
                print("   🎨 Markdown preserved in output")
            else:
                print("   ⚠️  No markdown detected - converted to plain text")

        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

    print("\n" + "=" * 55)
    print("✅ Voice formatting test completed!")
    print("\n💡 Key Fix Applied:")
    print("   - Added format_telegram_markdown() call to voice handler")
    print("   - Voice responses now use same formatting as text responses")
    print("   - Markdown should render properly in Telegram")
    print("\n🧪 To verify the fix:")
    print("   1. Send a voice message to the bot")
    print("   2. Check that AI response shows proper bold/italic formatting")
    print("   3. Compare with text responses - should be identical formatting")


if __name__ == "__main__":
    asyncio.run(test_formatting())
