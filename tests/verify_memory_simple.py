
import asyncio
import sys
import os
import time
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# app root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

async def verify_memory_manager():
    print("Verifying MemoryManager...")
    
    # Mock 'database.connection' in sys.modules to prevent real import
    mock_db_module = MagicMock()
    mock_db_module.get_database.return_value = (MagicMock(), MagicMock())
    sys.modules["database.connection"] = mock_db_module
    
    # Now import MemoryManager
    try:
        from services.memory_context.memory_manager import MemoryManager
    except ImportError as e:
        print(f"FAILED to import MemoryManager: {e}")
        return

    # Mock DB instance
    mock_db = MagicMock()
    
    # Initialize
    try:
        mm = MemoryManager(db=mock_db)
        print("MemoryManager initialized successfully.")
    except Exception as e:
        print(f"FAILED to initialize MemoryManager: {e}")
        return

    # User ID
    user_id = "test_user_123"
    conv_id = "test_conv_1"

    # Add messages
    print("Adding messages...")
    try:
        await mm.add_user_message(conv_id, "Hello, I need to remember to buy milk.", user_id)
        await mm.add_assistant_message(conv_id, "I made a note of that.")
        await mm.add_user_message(conv_id, "What did I need to buy?", user_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED to add message: {e}")
        return
    
    # Get relevant memory (should use simple keyword matching)
    print("Testing retrieval (get_relevant_memory)...")
    relevant = await mm.get_relevant_memory(conv_id, "buy milk", limit=2)
    print(f"Relevant messages found: {len(relevant)}")
    for msg in relevant:
        print(f" - {msg['role']}: {msg['content']}")

    # Build context bundle
    print("Testing context bundle...")
    context = await mm.build_context_bundle(conv_id, limit=5)
    print(f"Context keys: {context.keys()}")
    if context.get("recent"):
        print(f"Recent messages in context: {len(context['recent'])}")
    else:
        print("No recent messages returned in context.")
    
    print("Verification complete.")

if __name__ == "__main__":
    asyncio.run(verify_memory_manager())
