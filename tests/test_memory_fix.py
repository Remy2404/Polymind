import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager


async def test_memory():
    """Test the memory management system."""
    # Create memory manager with a test-specific storage path
    test_storage_path = "./test_data/memory"
    os.makedirs(test_storage_path, exist_ok=True)
    memory_manager = MemoryManager(storage_path=test_storage_path)

    # Create model history manager
    model_history_manager = ModelHistoryManager(memory_manager)

    # Test saving a conversation
    user_id = 806762900
    await model_history_manager.save_message_pair(
        user_id,
        "my name is Ramy how about u ?",
        "Nice to meet you, Ramy! I am Llama-4 Maverick.",
        "llama4_maverick",
    )

    print("✅ Saved conversation pair")

    # Test loading conversation
    history = await model_history_manager.get_history(
        user_id, model_id="llama4_maverick"
    )
    print(f"✅ Loaded {len(history)} messages:")
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        print(f"  {role}: {content}")

    # Test if memory persists by loading again
    print("\n🔄 Testing persistence...")
    history2 = await model_history_manager.get_history(
        user_id, model_id="llama4_maverick"
    )
    print(f"✅ Reloaded {len(history2)} messages")

    return len(history) > 0


if __name__ == "__main__":
    result = asyncio.run(test_memory())
    print(f"\n🎯 Memory test {'PASSED' if result else 'FAILED'}")
