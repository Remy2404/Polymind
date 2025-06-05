"""
Example Usage of AI Command Router

This file demonstrates how the new intelligent command routing works.
Users can now chat naturally and the bot will automatically detect
what they want to do and execute the appropriate commands.
"""

# Example conversations that will now be automatically routed:

EXAMPLE_CONVERSATIONS = [
    # Document Generation Examples
    {
        "user_input": "Create a document about artificial intelligence",
        "detected_intent": "GENERATE_DOCUMENT", 
        "auto_command": "/gendoc artificial intelligence",
        "explanation": "Detects document generation intent and automatically calls document generation with the topic"
    },
    
    {
        "user_input": "I need a business report on renewable energy",
        "detected_intent": "GENERATE_DOCUMENT",
        "auto_command": "/gendoc business report on renewable energy", 
        "explanation": "Recognizes business report as document type and extracts the topic"
    },
    
    {
        "user_input": "Write me an article about space exploration",
        "detected_intent": "GENERATE_DOCUMENT",
        "auto_command": "/gendoc space exploration",
        "explanation": "Identifies article writing request and extracts topic"
    },
    
    # Image Generation Examples
    {
        "user_input": "Create an image of a sunset over mountains",
        "detected_intent": "GENERATE_IMAGE",
        "auto_command": "/generate_image sunset over mountains",
        "explanation": "Detects image creation request and extracts visual description"
    },
    
    {
        "user_input": "Draw me a picture of a cute robot",
        "detected_intent": "GENERATE_IMAGE", 
        "auto_command": "/generate_image cute robot",
        "explanation": "Recognizes drawing/picture request and extracts description"
    },
    
    {
        "user_input": "I want to visualize a futuristic city",
        "detected_intent": "GENERATE_IMAGE",
        "auto_command": "/generate_image futuristic city", 
        "explanation": "Understands visualization request as image generation"
    },
    
    # Export Examples
    {
        "user_input": "Export this chat to PDF",
        "detected_intent": "EXPORT_CHAT",
        "auto_command": "/exportdoc",
        "explanation": "Recognizes export request and calls export command"
    },
    
    {
        "user_input": "Save our conversation as a document", 
        "detected_intent": "EXPORT_CHAT",
        "auto_command": "/exportdoc",
        "explanation": "Detects conversation saving intent"
    },
    
    # Model Switching Examples
    {
        "user_input": "Switch to a different AI model",
        "detected_intent": "SWITCH_MODEL",
        "auto_command": "/switchmodel",
        "explanation": "Detects model switching request"
    },
    
    {
        "user_input": "What models are available?",
        "detected_intent": "SWITCH_MODEL", 
        "auto_command": "/switchmodel",
        "explanation": "Shows available models when asked"
    },
    
    # Stats Examples
    {
        "user_input": "Show my usage statistics",
        "detected_intent": "GET_STATS",
        "auto_command": "/stats",
        "explanation": "Detects stats request and displays usage data"
    },
    
    {
        "user_input": "How many messages have I sent?",
        "detected_intent": "GET_STATS",
        "auto_command": "/stats", 
        "explanation": "Recognizes usage inquiry as stats request"
    }
]

# How the AI Command Router works:

WORKFLOW_EXPLANATION = """
🤖 AI Command Router Workflow:

1. USER SENDS MESSAGE
   - User types natural language: "Create a document about climate change"

2. INTENT DETECTION  
   - AI analyzes message using pattern matching and keywords
   - Detects intent: GENERATE_DOCUMENT (confidence: 0.85)

3. PROMPT EXTRACTION
   - Extracts core topic: "climate change" 
   - Removes command words like "create", "document", etc.

4. COMMAND ROUTING
   - Automatically calls: command_handlers.generate_ai_document_command()
   - Sets context.args = ["climate", "change"]

5. EXECUTION
   - Document generation proceeds as if user typed "/gendoc climate change"
   - User gets their document without needing to know specific commands

6. FALLBACK
   - If intent detection fails or command execution fails
   - Falls back to normal chat conversation with AI
"""

BENEFITS = """
✨ Benefits for Users:

🔹 NATURAL INTERACTION: Just describe what you want in plain English
🔹 NO COMMAND MEMORIZATION: Don't need to remember /gendoc, /generate_image, etc.
🔹 INTELLIGENT ROUTING: Bot understands context and intent automatically  
🔹 SEAMLESS EXPERIENCE: Commands execute automatically in background
🔹 FALLBACK SAFETY: Still works as normal chatbot if intent unclear
🔹 MULTILINGUAL: Works with natural language in different styles

📈 Business Value:

💰 INCREASED USAGE: Users more likely to use features they don't need to remember
📊 BETTER UX: Reduces friction between user intent and feature execution  
🎯 FEATURE DISCOVERY: Users naturally discover capabilities through conversation
🚀 COMPETITIVE ADVANTAGE: More intuitive than command-based bots
"""

# Advanced Examples - Complex Intent Detection

ADVANCED_EXAMPLES = [
    {
        "user_input": "I'm working on a presentation and need some visual content showing data trends",
        "detected_intent": "GENERATE_IMAGE",
        "extracted_prompt": "data trends visualization charts graphs",
        "explanation": "Understands 'visual content' in context means image generation"
    },
    
    {
        "user_input": "Can you help me prepare a comprehensive analysis of market conditions for my boss?",
        "detected_intent": "GENERATE_DOCUMENT", 
        "extracted_prompt": "comprehensive analysis of market conditions",
        "explanation": "Recognizes 'prepare analysis' as document generation request"
    },
    
    {
        "user_input": "My team needs to review our chat history from this project discussion",
        "detected_intent": "EXPORT_CHAT",
        "extracted_prompt": None,
        "explanation": "Understands 'review chat history' implies need to export conversation"
    }
]

if __name__ == "__main__":
    print("🤖 AI Command Router - Natural Language to Commands")
    print("=" * 60)
    print(WORKFLOW_EXPLANATION)
    print("\n" + "=" * 60)
    print(BENEFITS)
    print("\n" + "=" * 60)
    print("📝 Example Conversations:")
    
    for i, example in enumerate(EXAMPLE_CONVERSATIONS[:5], 1):
        print(f"\n{i}. USER: \"{example['user_input']}\"")
        print(f"   🔍 DETECTED: {example['detected_intent']}")
        print(f"   ⚡ EXECUTES: {example['auto_command']}")
        print(f"   💡 {example['explanation']}")
