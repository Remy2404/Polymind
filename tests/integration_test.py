import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.services.memory_context.memory_manager import MemoryManager
from src.services.ai_command_router import AICommandRouter
from src.database.connection import get_database


async def integration_test():
    print("🔧 Integration Test: Memory Manager + Intent Detection")
    print("=" * 55)

    try:
        # Test Memory Manager initialization
        db, client = get_database()
        memory_manager = MemoryManager(db=db, client=client)
        print("✅ Memory Manager initialized successfully")

        # Test AI Command Router
        router = AICommandRouter(command_handlers=None, gemini_api=None)
        print("✅ AI Command Router initialized successfully")

        # Test simple message routing (should NOT be routed)
        simple_msgs = ["k", "f", "ok", "hi"]
        all_passed = True

        for msg in simple_msgs:
            should_route = await router.should_route_message(msg)
            if should_route:
                print(f'❌ FAIL: "{msg}" incorrectly routed')
                all_passed = False
            else:
                print(f'✅ PASS: "{msg}" correctly ignored')

        # Test memory operations
        await memory_manager.add_user_message(
            conversation_id="integration_test",
            content="This is a test message",
            user_id="123",
        )
        print("✅ Memory operations working")

        # Test memory retrieval
        relevant = await memory_manager.get_relevant_memory(
            conversation_id="integration_test", query="test"
        )
        print(f"✅ Memory retrieval working ({len(relevant)} results)")

        if all_passed:
            print("\n🎉 INTEGRATION TEST PASSED!")
            print("✅ Both systems working correctly together")
            return True
        else:
            print("\n❌ INTEGRATION TEST FAILED!")
            print("❌ Some simple messages are still being misrouted")
            return False

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(integration_test())
    exit(0 if success else 1)
