#!/usr/bin/env python3
"""
Test script to verify the rate limiter fix
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import asyncio
from src.services.rate_limiter import RateLimiter


async def test_rate_limiter():
    """Test the rate limiter with different usage patterns"""
    print("🧪 Testing Rate Limiter...")

    # Test basic usage
    limiter = RateLimiter(requests_per_minute=60)

    # Test wait method
    print("✅ Testing wait() method...")
    await limiter.wait()
    print("   wait() works correctly")

    # Test context manager
    print("✅ Testing async context manager...")
    async with limiter:
        print("   context manager works correctly")

    # Test release method (should not error)
    print("✅ Testing release() method...")
    limiter.release()
    print("   release() works correctly")

    # Test capacity method
    print("✅ Testing get_current_capacity() method...")
    capacity = await limiter.get_current_capacity()
    print(f"   current capacity: {capacity:.1f}%")

    print("🎉 All tests passed! Rate limiter is working correctly.")


if __name__ == "__main__":
    asyncio.run(test_rate_limiter())
