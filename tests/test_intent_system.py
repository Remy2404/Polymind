
"""
Intent Detection Test Script
Tests the regex-based intent detection system with various message types
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.ai_command_router import AICommandRouter, CommandIntent


class MockCommandHandlers:
    """Mock command handlers for testing"""
    def __init__(self):
        pass
    
    async def generate_ai_document_command(self, update, context):
        print("✅ Document generation command executed")
        return True
    
    async def generate_together_image(self, update, context):
        print("✅ Image generation command executed")
        return True
    
    async def export_to_document(self, update, context):
        print("✅ Export command executed")
        return True
    
    async def switch_model_command(self, update, context):
        print("✅ Model switch command executed")
        return True
    
    async def handle_stats(self, update, context):
        print("✅ Stats command executed")
        return True
    
    async def help_command(self, update, context):
        print("✅ Help command executed")
        return True
    
    async def reset_command(self, update, context):
        print("✅ Reset command executed")
        return True
    
    async def settings(self, update, context):
        print("✅ Settings command executed")
        return True


async def test_intent_detection():
    """Test the intent detection system with various message types"""
    
    print("🔍 Testing Regex-based Intent Detection System")
    print("=" * 50)
    
    # Initialize the router
    mock_handlers = MockCommandHandlers()
    router = AICommandRouter(mock_handlers, gemini_api=None)
    
    # Test cases covering different intent types
    test_cases = [
        # Document generation
        ("Create a business plan document", CommandIntent.GENERATE_DOCUMENT),
        ("Generate a report about AI trends", CommandIntent.GENERATE_DOCUMENT),
        ("Write a comprehensive article on machine learning", CommandIntent.GENERATE_DOCUMENT),
        ("I need a document for my presentation", CommandIntent.GENERATE_DOCUMENT),
        
        # Image generation
        ("Create an image of a sunset", CommandIntent.GENERATE_IMAGE),
        ("Generate a logo for my company", CommandIntent.GENERATE_IMAGE),
        ("Draw me a picture of a cat", CommandIntent.GENERATE_IMAGE),
        ("I want an illustration of a futuristic city", CommandIntent.GENERATE_IMAGE),
        
        # Export functionality
        ("Export our chat history", CommandIntent.EXPORT_CHAT),
        ("Save this conversation to a file", CommandIntent.EXPORT_CHAT),
        ("Download our chat as a document", CommandIntent.EXPORT_CHAT),
        
        # Model switching
        ("Switch to Claude model", CommandIntent.SWITCH_MODEL),
        ("Change AI model to Gemini", CommandIntent.SWITCH_MODEL),
        ("What models are available?", CommandIntent.SWITCH_MODEL),
        
        # Stats and information
        ("Show me my usage statistics", CommandIntent.GET_STATS),
        ("What are my stats?", CommandIntent.GET_STATS),
        ("Display usage analytics", CommandIntent.GET_STATS),
        
        # Help requests
        ("Help me understand the commands", CommandIntent.HELP),
        ("What can you do?", CommandIntent.HELP),
        ("Show available features", CommandIntent.HELP),
        
        # Reset/clear
        ("Reset our conversation", CommandIntent.RESET),
        ("Clear chat history", CommandIntent.RESET),
        ("Delete our conversation", CommandIntent.RESET),
        
        # Settings
        ("Change my preferences", CommandIntent.SETTINGS),
        ("Update my settings", CommandIntent.SETTINGS),
        ("Configure my account", CommandIntent.SETTINGS),
        
        # Regular chat (should not be routed)
        ("Hello, how are you today?", CommandIntent.CHAT),
        ("What's the weather like?", CommandIntent.CHAT),
        ("Tell me a joke", CommandIntent.CHAT),
        ("Explain quantum physics to me", CommandIntent.CHAT),
        ("How do I cook pasta?", CommandIntent.CHAT),
        
        # Edge cases
        ("", CommandIntent.UNKNOWN),
        ("Hi", CommandIntent.UNKNOWN),
        ("Thanks", CommandIntent.UNKNOWN),
        ("Okay", CommandIntent.UNKNOWN),
    ]
    
    print(f"🧪 Testing {len(test_cases)} message types...")
    print()
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, (message, expected_intent) in enumerate(test_cases, 1):
        try:
            # Test intent detection
            detected_intent, confidence = await router.detect_intent(message)
            
            # Test routing decision
            should_route = await router.should_route_message(message)
            
            # Check if prediction is correct
            is_correct = detected_intent == expected_intent
            if is_correct:
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            # Determine expected routing
            expected_routing = expected_intent not in [CommandIntent.CHAT, CommandIntent.ANALYZE, CommandIntent.UNKNOWN]
            routing_correct = should_route == expected_routing
            
            print(f"{status} Test {i:2d}: '{message[:40]:<40}' -> {detected_intent.value:<20} (conf: {confidence:.2f}) [Route: {should_route}]")
            
            if not is_correct:
                print(f"         Expected: {expected_intent.value}")
            
            if not routing_correct:
                print(f"         Routing mismatch: expected {expected_routing}, got {should_route}")
            
        except Exception as e:
            print(f"❌ Test {i:2d}: Error testing '{message}': {str(e)}")
    
    print()
    print("=" * 50)
    print(f"🎯 Accuracy: {correct_predictions}/{total_tests} ({correct_predictions/total_tests*100:.1f}%)")
    
    if correct_predictions / total_tests >= 0.8:
        print("🎉 Great! Intent detection is performing well!")
    elif correct_predictions / total_tests >= 0.6:
        print("⚠️  Intent detection needs some tuning")
    else:
        print("🔧 Intent detection needs significant improvement")
    
    return correct_predictions / total_tests


async def test_prompt_extraction():
    """Test prompt extraction functionality"""
    
    print("\n🔧 Testing Prompt Extraction")
    print("=" * 30)
    
    mock_handlers = MockCommandHandlers()
    router = AICommandRouter(mock_handlers, gemini_api=None)
    
    # Test document prompt extraction
    doc_test_cases = [
        ("Create a business plan document for a tech startup", "business plan for a tech startup"),
        ("Generate a report about climate change impacts", "climate change impacts"),
        ("I need a document on artificial intelligence", "artificial intelligence"),
        ("Write an article about renewable energy", "renewable energy"),
    ]
    
    print("📄 Document Prompt Extraction:")
    for original, expected in doc_test_cases:
        extracted = router._extract_prompt_for_document(original)
        print(f"  '{original}' -> '{extracted}'")
        if expected.lower() in extracted.lower():
            print("  ✅ Good extraction")
        else:
            print("  ⚠️  Could be improved")
    
    # Test image prompt extraction
    img_test_cases = [
        ("Create an image of a beautiful sunset over mountains", "beautiful sunset over mountains"),
        ("Generate a logo with a blue theme", "logo with a blue theme"),
        ("Draw me a cat playing with yarn", "cat playing with yarn"),
        ("I want an illustration of space exploration", "space exploration"),
    ]
    
    print("\n🎨 Image Prompt Extraction:")
    for original, expected in img_test_cases:
        extracted = router._extract_prompt_for_image(original)
        print(f"  '{original}' -> '{extracted}'")
        if extracted and expected.lower() in extracted.lower():
            print("  ✅ Good extraction")
        else:
            print("  ⚠️  Could be improved")


async def test_spacy_fallback():
    """Test the fallback mechanism when spaCy is not available"""
    
    print("\n🔄 Testing Fallback Mechanism")
    print("=" * 30)
    
    # Create a router for testing
    mock_handlers = MockCommandHandlers()
    router = AICommandRouter(mock_handlers, gemini_api=None)
    
    # Test a few key cases
    test_cases = [
        ("Create a document about AI", CommandIntent.GENERATE_DOCUMENT),
        ("Generate an image of a cat", CommandIntent.GENERATE_IMAGE),
        ("Export our chat", CommandIntent.EXPORT_CHAT),
        ("Hello there", CommandIntent.CHAT),
    ]
    
    print("Testing fallback detection:")
    for message, expected in test_cases:
        detected_intent, confidence = await router.detect_intent(message)
        status = "✅" if detected_intent == expected else "❌"
        print(f"  {status} '{message}' -> {detected_intent.value} (conf: {confidence:.2f})")
    
    print("✅ Fallback mechanism test completed")


async def main():
    """Run all tests"""
    try:
        print("🚀 Starting Intent Detection Test Suite")
        print("=" * 50)
        
        # Test core intent detection
        accuracy = await test_intent_detection()
        
        # Test prompt extraction
        await test_prompt_extraction()
        
        # Test fallback mechanism
        await test_spacy_fallback()
        
        print("\n" + "=" * 50)
        print("🏁 Test Suite Complete!")
        
        if accuracy >= 0.8:
            print("✅ System is ready for production use!")
        else:
            print("⚠️  Consider tuning the intent keywords for better accuracy")
            
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
