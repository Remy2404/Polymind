#!/usr/bin/env python3
"""
Test script to debug intent detection
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.handlers.text_processing.intent_detector import IntentDetector
import asyncio


async def test_intent_detection():
    """Test intent detection for various message types"""
    print("🔍 Testing Intent Detection...")

    # Initialize with debug mode
    detector = IntentDetector(debug_mode=True)

    # Test messages
    test_messages = [
        # Text tutorial request (should be 'chat')
        (
            "Write a comprehensive tutorial on Python programming for beginners. Include the following sections: 1. Introduction...",
            False,
        ),
        # Image generation request (should be 'generate_image')
        ("Generate an image of a sunset over the ocean", False),
        # Media analysis request (should be 'analyze')
        ("What's in this image?", True),
        # Simple chat (should be 'chat')
        ("Hello, how are you?", False),
        # Code request (should be 'chat')
        ("Write a Python function to sort a list", False),
        # Creative writing (should be 'chat')
        ("Write a story about a robot", False),
    ]

    for message, has_media in test_messages:
        print(f"\n📝 Message: {message[:50]}...")
        print(f"📎 Has media: {has_media}")

        intent = await detector.detect_user_intent(message, has_media)
        print(f"🎯 Detected intent: {intent}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(test_intent_detection())
