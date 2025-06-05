"""
Test script to demonstrate AI Command Router functionality
Run this to see how natural language gets converted to commands
"""
import asyncio
import sys
import os

# Add the parent directory to sys.path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.ai_command_router import AICommandRouter, CommandIntent


class MockCommandHandlers:
    """Mock command handlers for testing"""
    
    async def generate_ai_document_command(self, update, context):
        print(f"✅ EXECUTING: Document generation with prompt: {' '.join(context.args)}")
        return True
    
    async def generate_together_image(self, update, context):
        print(f"✅ EXECUTING: Image generation with prompt: {' '.join(context.args)}")
        return True
    
    async def export_to_document(self, update, context):
        print("✅ EXECUTING: Chat export to document")
        return True
    
    async def switch_model_command(self, update, context):
        print("✅ EXECUTING: Model switching interface")
        return True
    
    async def handle_stats(self, update, context):
        print("✅ EXECUTING: Show user statistics")
        return True
    
    async def help_command(self, update, context):
        print("✅ EXECUTING: Show help information")
        return True
    
    async def reset_command(self, update, context):
        print("✅ EXECUTING: Reset conversation")
        return True
    
    async def settings(self, update, context):
        print("✅ EXECUTING: Show settings")
        return True


class MockUpdate:
    def __init__(self):
        self.message = MockMessage()


class MockMessage:
    def __init__(self):
        pass
    
    async def reply_text(self, text):
        print(f"🤖 BOT REPLY: {text}")


class MockContext:
    def __init__(self):
        self.args = []


async def test_ai_command_router():
    """Test the AI command router with various user inputs"""
    
    print("🤖 AI Command Router Test")
    print("=" * 50)
    print("Testing how natural language gets converted to commands...\n")
    
    # Initialize mock handlers and router
    mock_handlers = MockCommandHandlers()
    router = AICommandRouter(mock_handlers)
    
    # Test cases - natural language inputs
    test_cases = [
        "Create a document about artificial intelligence",
        "Generate an image of a sunset over mountains", 
        "I need a business report on renewable energy",
        "Draw me a picture of a cute robot",
        "Export this chat to PDF",
        "Switch to a different AI model",
        "Show my usage statistics",
        "Can you help me with the available commands?",
        "Reset our conversation",
        "I want to change my settings",
        "Write an article about space exploration",
        "Create a visual of futuristic city",
        "Save our conversation as a document",
        "What models are available?",
        "How many messages have I sent?",
        "Just having a normal conversation here",  # Should not route
    ]
    
    for i, user_input in enumerate(test_cases, 1):
        print(f"{i:2d}. USER: \"{user_input}\"")
        
        # Detect intent
        intent, confidence = await router.detect_intent(user_input)
        print(f"    🔍 DETECTED: {intent.value} (confidence: {confidence:.2f})")
        
        # Check if should route
        should_route = await router.should_route_message(user_input)
        
        if should_route:
            # Create mock objects
            mock_update = MockUpdate()
            mock_context = MockContext()
            
            # Route the command
            success = await router.route_command(mock_update, mock_context, intent, user_input)
            
            if not success:
                print("    ❌ ROUTING FAILED - falling back to normal chat")
        else:
            print("    💬 NORMAL CHAT - no command routing needed")
        
        print()  # Empty line for readability


if __name__ == "__main__":
    print("🚀 Starting AI Command Router Demo...")
    print("This shows how users can use natural language instead of remembering commands!\n")
    
    # Run the test
    asyncio.run(test_ai_command_router())
    
    print("\n" + "=" * 50)
    print("✨ Demo Complete!")
    print("\nHow it works in your bot:")
    print("1. User types: 'Create a document about climate change'")
    print("2. AI detects: GENERATE_DOCUMENT intent")
    print("3. System extracts: 'climate change' as the topic")
    print("4. Automatically calls: /gendoc climate change")
    print("5. User gets their document without remembering commands!")
