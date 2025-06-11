
import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager

async def test_refactored_memory_manager():
    """Test the refactored memory management system."""
    print("🧪 Testing Refactored Memory Manager")
    print("=" * 50)
    
    # Create memory manager with a test-specific storage path
    test_storage_path = "./test_data/memory"
    os.makedirs(test_storage_path, exist_ok=True)
    memory_manager = MemoryManager(storage_path=test_storage_path)

    # Test user profile management
    print("\n📝 Testing User Profile Management:")
    user_id = 806762900
    
    # Save user profile
    profile_data = {
        "name": "Ramy",
        "preferences": {"language": "en"},
        "created_at": 1620000000
    }
    await memory_manager.save_user_profile(user_id, profile_data)
    print(f"✅ Saved user profile for user {user_id}")
    
    # Retrieve user profile
    retrieved_profile = await memory_manager.get_user_profile(user_id)
    print(f"✅ Retrieved profile: {retrieved_profile}")
    
    # Test message storage and retrieval
    print("\n💬 Testing Message Storage:")
    
    # Add user message
    conversation_id = f"user_{user_id}"
    await memory_manager.add_user_message(
        conversation_id=conversation_id,
        content="Hello, my name is Ramy. How are you?",
        user_id=str(user_id),
        message_type="text",
        importance=0.8
    )
    print("✅ Added user message")
    
    # Add assistant response
    await memory_manager.add_assistant_message(
        conversation_id=conversation_id,
        content="Nice to meet you, Ramy! I'm doing well, thank you for asking. How can I help you today?",
        message_type="text",
        importance=0.7
    )
    print("✅ Added assistant message")
    
    # Test semantic search
    print("\n🔍 Testing Semantic Search:")
    relevant_messages = await memory_manager.get_relevant_memory(
        conversation_id=conversation_id,
        query="name introduction",
        limit=3
    )
    print(f"✅ Found {len(relevant_messages)} relevant messages for 'name introduction'")
    for i, msg in enumerate(relevant_messages):
        print(f"  {i+1}. {msg.get('role', 'unknown')}: {msg.get('content', '')[:60]}...")
    
    # Test short-term memory
    print("\n📋 Testing Short-term Memory:")
    recent_messages = await memory_manager.get_short_term_memory(
        conversation_id=conversation_id,
        limit=5
    )
    print(f"✅ Retrieved {len(recent_messages)} recent messages")
    
    # Test conversation summary
    print("\n📊 Testing Conversation Summary:")
    summary = await memory_manager.get_conversation_summary(conversation_id)
    print(f"✅ Generated summary: {summary}")
    
    # Test group functionality
    print("\n👥 Testing Group Memory:")
    group_id = "test_group_123"
    
    # Add group messages
    await memory_manager.add_user_message(
        conversation_id="",
        content="Hey everyone, let's discuss our project deadline",
        user_id="user1",
        is_group=True,
        group_id=group_id,
        importance=0.9
    )
    
    await memory_manager.add_user_message(
        conversation_id="",
        content="I think we should aim for next Friday",
        user_id="user2", 
        is_group=True,
        group_id=group_id,
        importance=0.8
    )
    
    await memory_manager.add_assistant_message(
        conversation_id="",
        content="That sounds like a reasonable timeline. Would you like me to help organize the tasks?",
        is_group=True,
        group_id=group_id,
        importance=0.7
    )
    print("✅ Added group messages")
    
    # Test group participants
    participants = await memory_manager.get_group_participants(group_id)
    print(f"✅ Group participants: {participants}")
    
    # Test group activity summary
    activity_summary = await memory_manager.get_group_activity_summary(group_id)
    print(f"✅ Group activity summary: {activity_summary}")
    
    # Test group relevant memory
    group_relevant = await memory_manager.get_relevant_memory(
        conversation_id="",
        query="project deadline",
        limit=3,
        is_group=True,
        group_id=group_id,
        include_group_knowledge=True
    )
    print(f"✅ Found {len(group_relevant)} relevant group messages for 'project deadline'")
    
    # Test conversation export
    print("\n📤 Testing Data Export:")
    export_data = await memory_manager.export_conversation_data(conversation_id)
    print(f"✅ Exported conversation data with {export_data.get('total_messages', 0)} messages")
    
    group_export_data = await memory_manager.export_conversation_data(
        "", is_group=True, group_id=group_id
    )
    print(f"✅ Exported group data with {group_export_data.get('total_messages', 0)} messages")
    
    # Test user info extraction
    print("\n🔍 Testing User Info Extraction:")
    await memory_manager.extract_and_save_user_info(
        user_id, "Actually, my name is Ahmed and I prefer to be called Ahmed"
    )
    
    updated_profile = await memory_manager.get_user_profile(user_id)
    print(f"✅ Updated profile after extraction: {updated_profile}")
    
    print("\n🎉 All tests completed successfully!")
    print("=" * 50)

async def test_model_history_compatibility():
    """Test compatibility with ModelHistoryManager"""
    print("\n🔗 Testing ModelHistoryManager Compatibility:")
    
    memory_manager = MemoryManager()
    model_history_manager = ModelHistoryManager(memory_manager)
    
    # Test saving a message pair
    user_id = 806762900
    await model_history_manager.save_message_pair(
        user_id,
        "What's the weather like?",
        "I don't have access to real-time weather data, but I can help you find weather information!",
        "gemini"
    )
    print("✅ Saved message pair through ModelHistoryManager")
    
    # Test loading history
    history = await model_history_manager.get_history(user_id, model_id="gemini")
    print(f"✅ Loaded {len(history)} messages from ModelHistoryManager")
    
    print("✅ ModelHistoryManager compatibility confirmed!")

if __name__ == "__main__":
    asyncio.run(test_refactored_memory_manager())
    asyncio.run(test_model_history_compatibility())
