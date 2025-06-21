#!/usr/bin/env python3
"""
Simple test to verify spaCy-based intent detection is working
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.smart_intent_detector import SmartIntentDetector, CommandIntent
import asyncio

class MockCommandHandlers:
    """Mock command handlers for testing"""
    pass

async def test_spacy_intent_detection():
    """Test the new spaCy-based intent detection"""
    print("🎯 Testing spaCy-based Intent Detection")
    print("=" * 50)
    
    try:
        detector = SmartIntentDetector(MockCommandHandlers())
        print("✅ spaCy detector initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize spaCy detector: {e}")
        return
    
    # Test cases that were problematic before
    test_cases = [
        # The user's specific request from the screenshot
        ("Write a comprehensive tutorial on Python programming for beginners. Include the following sections: 1. Introduction to Python and its history 2. Setting up the development environment 3. Basic syntax and data types 4. Control structures (if statements, loops) 5. Functions and modules 6. Object-oriented programming basics 7. File handling 8. Error handling and debugging 9. Popular libraries and frameworks 10. Best practices and coding standards Make it detailed with code examples for each section", False),
        
        # Other educational requests
        ("Explain machine learning step by step", False),
        ("How do I learn React.js?", False),
        ("Create a tutorial on web development", False),
        
        # Document generation
        ("Generate a business report", False),
        ("Create a PDF document", False),
        
        # Image generation
        ("Create an image of a sunset", False),
        ("Draw a cartoon character", False),
        
        # Media analysis
        ("What's in this image?", True),
        ("Analyze this photo", True),
        
        # Commands
        ("Switch to Gemini model", False),
        ("Show my statistics", False),
        ("Reset conversation", False),
        
        # Regular chat
        ("What's the weather like?", False),
        ("Tell me a joke", False),
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} cases...\n")
    
    for i, (message, has_media) in enumerate(test_cases, 1):
        intent, confidence = await detector.detect_intent(message, has_media)
        
        # Determine if result makes sense
        context = "📎 Media" if has_media else "💬 Text"
        
        print(f"{i:2d}. {context} | {intent.value.upper()} ({confidence:.2f})")
        print(f"    \"{message[:60]}{'...' if len(message) > 60 else ''}\"")
        
        # Show specific feedback for key cases
        if "comprehensive tutorial" in message.lower():
            result = "✅ CORRECT" if intent == CommandIntent.CHAT else "❌ Should be CHAT"
            print(f"    📚 Educational content: {result}")
        elif "business report" in message.lower():
            result = "✅ CORRECT" if intent == CommandIntent.GENERATE_DOCUMENT else "❌ Should be GENERATE_DOCUMENT"
            print(f"    📄 Document generation: {result}")
        elif "image" in message.lower() and not has_media:
            result = "✅ CORRECT" if intent == CommandIntent.GENERATE_IMAGE else "❌ Should be GENERATE_IMAGE"
            print(f"    🎨 Image generation: {result}")
        elif has_media:
            result = "✅ CORRECT" if intent == CommandIntent.ANALYZE else "❌ Should be ANALYZE"
            print(f"    🔍 Media analysis: {result}")
        
        print()
    
    print("🎉 spaCy Intent Detection Benefits:")
    print("  • More accurate semantic understanding")
    print("  • Better handling of complex educational requests")
    print("  • Reduced code complexity (semantic matching vs regex)")
    print("  • Easier to extend and maintain")
    print("  • Handles synonyms and natural language variations")

if __name__ == "__main__":
    asyncio.run(test_spacy_intent_detection())
