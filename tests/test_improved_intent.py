
"""
Quick test for improved intent detection
Testing the specific issues from the chat screenshot
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.services.ai_command_router import EnhancedIntentDetector, CommandIntent

async def test_improved_detection():
    """Test the improved intent detection patterns"""
    
    print("🧪 Testing Improved Intent Detection")
    print("=" * 60)
    
    detector = EnhancedIntentDetector()
    
    # Test cases from the chat screenshot and common issues
    test_cases = [
        # Model switching - these were failing before
        ("I want to change model bro", CommandIntent.SWITCH_MODEL),
        ("change model", CommandIntent.SWITCH_MODEL),
        ("switch model", CommandIntent.SWITCH_MODEL),
        ("I want to use a different model", CommandIntent.SWITCH_MODEL),
        ("Can you switch to Claude?", CommandIntent.SWITCH_MODEL),
        ("Use Gemini model", CommandIntent.SWITCH_MODEL),
        
        # Document generation
        ("Create a business plan", CommandIntent.GENERATE_DOCUMENT),
        ("Write me a report", CommandIntent.GENERATE_DOCUMENT),
        ("Generate a summary", CommandIntent.GENERATE_DOCUMENT),
        
        # Image generation
        ("Draw me a picture", CommandIntent.GENERATE_IMAGE),
        ("Create an artwork", CommandIntent.GENERATE_IMAGE),
        ("Paint me something", CommandIntent.GENERATE_IMAGE),
        
        # Educational content (should work well)
        ("How to use Python?", CommandIntent.EDUCATIONAL),
        ("Explain machine learning", CommandIntent.EDUCATIONAL),
        
        # Chat (should not be routed)
        ("Hello there", CommandIntent.CHAT),
        ("How are you?", CommandIntent.CHAT),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, (message, expected) in enumerate(test_cases, 1):
        intent, confidence = await detector.detect_intent(message, False)
        
        status = "✅" if intent == expected else "❌"
        if intent == expected:
            correct += 1
            
        print(f"{status} Test {i:2d}: {message:<35} | Expected: {expected.value:<15} | Got: {intent.value:<15} | Conf: {confidence:.2f}")
        
        # Special attention to model switching
        if expected == CommandIntent.SWITCH_MODEL:
            if intent == CommandIntent.SWITCH_MODEL:
                print(f"         🔄 Model switching correctly detected!")
            else:
                print(f"         ⚠️  Model switching MISSED - this was the main issue!")
    
    accuracy = (correct / total) * 100
    print("\n" + "=" * 60)
    print(f"📊 IMPROVED DETECTION RESULTS:")
    print(f"   Total Tests: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print(f"   🎉 EXCELLENT! Improvements working well")
    elif accuracy >= 70:
        print(f"   ✅ GOOD! Significant improvement")
    else:
        print(f"   ⚠️  Still needs work")

if __name__ == "__main__":
    print("🚀 Testing Intent Detection Improvements")
    print("Focus: Model switching and command recognition")
    print()
    
    asyncio.run(test_improved_detection())
