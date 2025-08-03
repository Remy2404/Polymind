#!/usr/bin/env python3
"""
Test script to verify formatting improvements, especially dash cleanup
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.handlers.response_formatter import ResponseFormatter
import asyncio


async def test_formatting_cleanup():
    """Test that unwanted dashes are properly cleaned up"""
    print("🔍 Testing Response Formatting and Dash Cleanup...")

    formatter = ResponseFormatter()

    # Test text with unwanted dash lines
    test_response = """
# Python Tutorial

This is a comprehensive tutorial.

---

## Section 1

Some content here.

-------

## Section 2

More content.

***

Final section.

--------
    """

    print("📝 Original response:")
    print(test_response)
    print("\n" + "=" * 50 + "\n")

    # Format the response
    formatted = await formatter.format_response(
        test_response, "test-user", "moonshot-kimi-dev-72b"
    )

    print("✨ Formatted response:")
    print(formatted)

    # Check if unwanted dashes were removed
    if (
        "---" not in formatted
        and "-------" not in formatted
        and "--------" not in formatted
    ):
        print("\n✅ Dash cleanup working correctly!")
    else:
        print("\n❌ Dash cleanup failed - unwanted dashes still present")

    # Test another case with mixed content
    test_response2 = """
**Answer:**

The solution is complex.

------ 

But we can solve it.

*************

End of response.
    """

    formatted2 = await formatter.format_response(
        test_response2, "test-user", "llama-3.3-8b"
    )
    print("\n📋 Second test result:")
    print(formatted2)


if __name__ == "__main__":
    asyncio.run(test_formatting_cleanup())
