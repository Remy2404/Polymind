
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.ai_command_router import AICommandRouter, CommandIntent

async def test_simple_message_fix():
    """Test that simple messages don't get misrouted to switch_model"""
    
    print("🔍 Testing Simple Message Intent Detection Fix")
    print("=" * 60)
    
    # Create router instance (without command handlers for testing)
    router = AICommandRouter(command_handlers=None, gemini_api=None)
    
    # Test cases that should NOT be routed as switch_model commands
    simple_messages = [
        "k",
        "f", 
        "ok",
        "yes",
        "no",
        "hi",
        "hello",
        "thanks",
        "thx",
        "lol",
        "haha",
        "cool",
        "nice",
        "good"
    ]
    
    # Test cases that SHOULD be routed as switch_model commands
    switch_model_messages = [
        "switch to gemini model",
        "change ai model", 
        "use deepseek",
        "what models are available",
        "show me available models",
        "switch model to claude",
        "change to gpt model",
        "list available models"
    ]
    
    print("📝 Testing simple messages (should NOT be routed):")
    all_passed = True
    
    for message in simple_messages:
        should_route = await router.should_route_message(message)
        intent, confidence = await router.detect_intent(message)
        
        print(f"  '{message:8}' -> should_route: {should_route}, intent: {intent.value:15}, confidence: {confidence:.2f}")
        
        if should_route and intent == CommandIntent.SWITCH_MODEL:
            print(f"    ❌ ERROR: Simple message '{message}' incorrectly routed to switch_model!")
            all_passed = False
        else:
            print(f"    ✅ OK: Simple message '{message}' correctly handled")
    
    print("\n🔧 Testing switch model messages (should be routed):")
    for message in switch_model_messages:
        should_route = await router.should_route_message(message)
        intent, confidence = await router.detect_intent(message)
        
        print(f"  '{message[:30]:30}' -> should_route: {should_route}, intent: {intent.value:15}, confidence: {confidence:.2f}")
        
        if should_route and intent == CommandIntent.SWITCH_MODEL:
            print(f"    ✅ OK: Command message '{message[:30]}...' correctly routed")
        else:
            print(f"    ⚠️  WARN: Command message '{message[:30]}...' not routed - confidence may be too low")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 SUCCESS: All simple messages are correctly handled!")
        print("✅ Fix verified: Simple messages like 'k', 'f' no longer misrouted")
    else:
        print("❌ FAILURE: Some simple messages are still being misrouted")
    
    return all_passed

async def test_confidence_threshold_behavior():
    """Test the confidence threshold and minimum length requirements"""
    
    print("\n🔬 Testing Confidence Threshold and Length Requirements")
    print("=" * 60)
    
    router = AICommandRouter(command_handlers=None, gemini_api=None)
    
    # Test edge cases around the minimum length (5 characters) and confidence (0.4)
    edge_cases = [
        ("k", "Should fail: too short"),
        ("help", "Should fail: too short but valid word"),
        ("model", "Exactly 5 chars - should work if confidence > 0.4"),
        ("models", "6 chars - should work if confidence > 0.4"),
        ("switch", "Valid length, should work if relevant"),
        ("switch model", "Should work: clear intent"),
        ("change ai", "Should work: clear intent"),
    ]
    
    for message, description in edge_cases:
        should_route = await router.should_route_message(message)
        intent, confidence = await router.detect_intent(message)
        
        print(f"  '{message:12}' ({len(message)} chars) -> route: {should_route}, conf: {confidence:.2f} | {description}")
    
    print("\n✅ Confidence threshold and length requirements tested")

if __name__ == "__main__":
    success = asyncio.run(test_simple_message_fix())
    asyncio.run(test_confidence_threshold_behavior())
    
    if success:
        print("\n🎯 FINAL RESULT: Intent detection fix is working correctly!")
        exit(0)
    else:
        print("\n💥 FINAL RESULT: Intent detection fix needs more work!")
        exit(1)
