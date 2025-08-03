"""
Quick Test Script for Enhanced AI Command Router Features
Run this to see the AI-powered NLP capabilities in action!
"""

import asyncio
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.services.ai_command_router import EnhancedIntentDetector, CommandIntent


async def quick_demo():
    """Quick demonstration of key features"""
    print("🚀 ENHANCED AI COMMAND ROUTER - QUICK DEMO")
    print("=" * 60)

    # Initialize the enhanced intent detector
    detector = EnhancedIntentDetector()

    # Test messages
    test_messages = [
        "How to create a Python web application with Flask?",
        "Generate a business report on market analysis",
        "Draw me a beautiful landscape with mountains",
        "Switch to a different AI model",
        "Write a function to sort an array in JavaScript",
        "Explain machine learning algorithms step by step",
        "Translate this text to Spanish",
        "Create a story about space exploration",
    ]

    print("🧠 Regex-based Intent Detection: ✅")
    print(
        f"🤖 Model Configs: {len(detector.model_configs) if detector.model_configs else 0} models loaded"
    )
    print("-" * 60)

    for i, message in enumerate(test_messages, 1):
        print(f'\n{i}. Testing: "{message}"')

        try:
            intent, confidence = await detector.detect_intent(message)

            # Color coding for confidence levels
            if confidence > 0.8:
                status = "🟢 HIGH"
            elif confidence > 0.6:
                status = "🟡 MEDIUM"
            elif confidence > 0.4:
                status = "🟠 LOW"
            else:
                status = "🔴 VERY LOW"

            print(f"   🎯 Intent: {intent.value}")
            print(f"   📊 Confidence: {confidence:.3f} {status}")

            # Show specific capabilities based on intent
            if intent == CommandIntent.EDUCATIONAL:
                print("   🎓 Educational content detected - will use reasoning models")
            elif intent == CommandIntent.CODING:
                print("   💻 Coding task detected - will use coding specialist models")
            elif intent == CommandIntent.GENERATE_IMAGE:
                print(
                    "   🎨 Image generation detected - will use vision-capable models"
                )
            elif intent == CommandIntent.GENERATE_DOCUMENT:
                print(
                    "   📄 Document generation detected - will use document-focused models"
                )

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ Quick demo completed!")
    print("🔧 For full demo with linguistic analysis, run:")
    print("   python examples/enhanced_ai_command_router_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(quick_demo())
