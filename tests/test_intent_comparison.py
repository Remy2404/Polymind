
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.ai_command_router import AICommandRouter, CommandIntent
from src.services.smart_intent_detector import SmartIntentDetector
import asyncio
import time

class MockCommandHandlers:
    """Mock command handlers for testing"""
    async def generate_ai_document_command(self, update, context): pass
    async def generate_together_image(self, update, context): pass
    async def export_to_document(self, update, context): pass
    async def switch_model_command(self, update, context): pass
    async def handle_stats(self, update, context): pass
    async def help_command(self, update, context): pass
    async def reset_command(self, update, context): pass
    async def settings(self, update, context): pass

async def test_intent_detection_comparison():
    """Compare old vs new intent detection approaches"""
    print("🔬 Intent Detection Comparison: Regex vs spaCy")
    print("=" * 60)
    
    # Initialize both systems
    mock_handlers = MockCommandHandlers()
    
    try:
        old_router = AICommandRouter(mock_handlers)
        print("✅ Old regex-based router initialized")
    except Exception as e:
        print(f"❌ Failed to initialize old router: {e}")
        old_router = None
    
    try:
        new_detector = SmartIntentDetector(mock_handlers)
        print("✅ New spaCy-based detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize new detector: {e}")
        return
    
    # Test cases with expected intents
    test_cases = [
        # Tutorial/Educational requests (should be CHAT)
        ("Write a comprehensive tutorial on Python programming for beginners", CommandIntent.CHAT, False),
        ("Explain how machine learning works step by step", CommandIntent.CHAT, False),
        ("Create a detailed guide on web development", CommandIntent.CHAT, False),
        ("How do I learn React.js? Give me a roadmap", CommandIntent.CHAT, False),
        
        # Document generation (should be GENERATE_DOCUMENT)
        ("Generate a business report about market analysis", CommandIntent.GENERATE_DOCUMENT, False),
        ("Create a PDF document about company policies", CommandIntent.GENERATE_DOCUMENT, False),
        ("I need a research paper on climate change", CommandIntent.GENERATE_DOCUMENT, False),
        
        # Image generation (should be GENERATE_IMAGE)
        ("Create an image of a sunset over the ocean", CommandIntent.GENERATE_IMAGE, False),
        ("Draw a cartoon character for my logo", CommandIntent.GENERATE_IMAGE, False),
        ("Generate artwork in cyberpunk style", CommandIntent.GENERATE_IMAGE, False),
        
        # Media analysis (should be ANALYZE)
        ("What's in this image?", CommandIntent.ANALYZE, True),
        ("Analyze this document", CommandIntent.ANALYZE, True),
        ("Describe what you see", CommandIntent.ANALYZE, True),
        
        # Model switching (should be SWITCH_MODEL)
        ("Switch to Gemini model", CommandIntent.SWITCH_MODEL, False),
        ("Change AI model to DeepSeek", CommandIntent.SWITCH_MODEL, False),
        ("What models are available?", CommandIntent.SWITCH_MODEL, False),
        
        # Stats (should be GET_STATS)
        ("Show my usage statistics", CommandIntent.GET_STATS, False),
        ("How many messages have I sent?", CommandIntent.GET_STATS, False),
        
        # Help (should be HELP)
        ("What commands are available?", CommandIntent.HELP, False),
        ("How do I use this bot?", CommandIntent.HELP, False),
        
        # Chat/Conversation (should be CHAT)
        ("What's the weather like today?", CommandIntent.CHAT, False),
        ("Tell me a joke", CommandIntent.CHAT, False),
        ("Compare Python vs JavaScript", CommandIntent.CHAT, False),
        
        # Reset (should be RESET)
        ("Reset our conversation", CommandIntent.RESET, False),
        ("Clear chat history", CommandIntent.RESET, False),
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} cases...")
    print("-" * 60)
    
    old_correct = 0
    new_correct = 0
    old_time = 0
    new_time = 0
    
    for i, (message, expected_intent, has_media) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. {message[:50]}...")
        print(f"    Expected: {expected_intent.value}")
        
        # Test old router
        if old_router:
            start_time = time.time()
            try:
                old_intent, old_confidence = await old_router.detect_intent(message, has_media)
                old_time += time.time() - start_time
                old_match = "✅" if old_intent == expected_intent else "❌"
                print(f"    Old:      {old_intent.value} ({old_confidence:.2f}) {old_match}")
                if old_intent == expected_intent:
                    old_correct += 1
            except Exception as e:
                print(f"    Old:      ERROR - {e}")
        else:
            print(f"    Old:      UNAVAILABLE")
          # Test new detector
        start_time = time.time()
        try:
            new_intent, new_confidence = await new_detector.detect_intent(message, has_media)
            new_time += time.time() - start_time
            new_match = "✅" if new_intent == expected_intent else "❌"
            print(f"    New:      {new_intent.value} ({new_confidence:.2f}) {new_match}")
            if new_intent == expected_intent:
                new_correct += 1
        except Exception as e:
            print(f"    New:      ERROR - {e}")
    
    # Results summary
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    total_cases = len(test_cases)
    
    if old_router:
        old_accuracy = (old_correct / total_cases) * 100
        print(f"Old Regex-based:  {old_correct:2d}/{total_cases} ({old_accuracy:5.1f}%) - {old_time:.3f}s total")
    else:
        print("Old Regex-based:  UNAVAILABLE")
    
    new_accuracy = (new_correct / total_cases) * 100
    print(f"New spaCy-based:  {new_correct:2d}/{total_cases} ({new_accuracy:5.1f}%) - {new_time:.3f}s total")
    
    if old_router:
        improvement = new_accuracy - old_accuracy
        speed_improvement = ((old_time - new_time) / old_time) * 100 if old_time > 0 else 0
        
        print(f"\n🚀 IMPROVEMENTS:")
        print(f"   Accuracy: {improvement:+.1f} percentage points")
        print(f"   Speed:    {speed_improvement:+.1f}% faster" if speed_improvement > 0 else f"   Speed:    {abs(speed_improvement):.1f}% slower")
    
    print(f"\n📈 spaCy Benefits:")
    print(f"   • Better semantic understanding")
    print(f"   • Handles synonyms and variations")
    print(f"   • More accurate intent classification")
    print(f"   • Significantly reduced code complexity")
    print(f"   • Easy to extend and maintain")

async def test_specific_educational_cases():
    """Test specifically on educational/tutorial requests"""
    print("\n" + "=" * 60)
    print("📚 EDUCATIONAL CONTENT DETECTION TEST")
    print("=" * 60)
    
    mock_handlers = MockCommandHandlers()
    detector = SmartIntentDetector(mock_handlers)
    
    educational_cases = [
        "Write a comprehensive tutorial on Python programming for beginners. Include sections on: 1. Introduction 2. Setup 3. Basic syntax...",
        "Explain machine learning algorithms in detail with examples",
        "Create a step-by-step guide for building a React application",
        "How do neural networks work? Provide a detailed explanation",
        "Give me a complete overview of cloud computing concepts",
        "Teach me about database design principles with examples",
        "Provide a detailed comparison between Python and Java",
        "Explain the fundamentals of cybersecurity",
    ]
    
    print("Testing educational content detection...")
    
    for i, message in enumerate(educational_cases, 1):
        intent, confidence = await detector.detect_intent(message, False)
        result = "✅ CHAT" if intent == CommandIntent.CHAT else f"❌ {intent.value}"
        print(f"{i}. {message[:60]}...")
        print(f"   Result: {result} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    asyncio.run(test_intent_detection_comparison())
    asyncio.run(test_specific_educational_cases())
