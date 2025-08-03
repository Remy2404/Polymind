#!/usr/bin/env python3
"""
Test script to verify response formatting works correctly
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.handlers.response_formatter import ResponseFormatter


async def test_voice_formatting():
    """Test voice message formatting"""

    print("=== TESTING VOICE MESSAGE FORMATTING ===")

    formatter = ResponseFormatter()

    # Test voice message format
    voice_intro = "🎤 **Voice Response:**"
    context_hint = "_Based on our conversation..._\n\n"
    ai_response = (
        "Absolutely! Based on our current conversation — yes, your name is **Rami**! 😊"
    )

    voice_formatted_response = f"{voice_intro}\n\n{context_hint}{ai_response}"

    print("Original text:")
    print(repr(voice_formatted_response))
    print("\nFormatted text:")
    print(voice_formatted_response)

    # Test HTML conversion
    html_converted = formatter._convert_markdown_to_html(voice_formatted_response)
    print("\nHTML converted:")
    print(html_converted)

    # Test Markdown conversion
    try:
        markdown_converted = await formatter.format_telegram_markdown(
            voice_formatted_response
        )
        print("\nMarkdown converted:")
        print(repr(markdown_converted))
    except Exception as e:
        print(f"\nMarkdown conversion error: {e}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(test_voice_formatting())
