"""Test script to verify the model commands work correctly"""

import sys
import os
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.services.model_handlers.simple_api_manager import SuperSimpleAPIManager
from src.handlers.commands.model_commands import ModelCommands


class MockQuery:
    def __init__(self, data):
        self.data = data
        self.from_user = MockUser()

    async def answer(self):
        pass

    async def edit_message_text(self, text, **kwargs):
        print(f"Bot would send: {text}")


class MockUser:
    def __init__(self):
        self.id = 12345


class MockUpdate:
    def __init__(self, callback_data):
        self.callback_query = MockQuery(callback_data)


class MockUserDataManager:
    async def get_user_preference(self, user_id, key, default=None):
        return default


async def test_category_selection():
    print("=== TESTING CATEGORY SELECTION ===\n")

    # Create API manager
    api_manager = SuperSimpleAPIManager()
    user_data_manager = MockUserDataManager()

    # Create model commands
    model_commands = ModelCommands(api_manager, user_data_manager)

    # Test different callback scenarios
    test_cases = [
        "category_gemini",
        "category_reasoning",
        "category_current",
        "category_all",
        "category_nonexistent",  # This should trigger the error
    ]

    for callback_data in test_cases:
        print(f"Testing callback: {callback_data}")
        try:
            # Create mock update
            update = MockUpdate(callback_data)

            # Test the category selection
            await model_commands.handle_category_selection(update, None)
            print("✅ Success")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(test_category_selection())
