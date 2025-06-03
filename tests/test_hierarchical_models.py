import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")

    try:
        from src.services.model_handlers.simple_api_manager import (
            SuperSimpleAPIManager,
            PROVIDER_GROUPS,
        )

        print("✅ SuperSimpleAPIManager imported successfully")

        from src.handlers.commands.model_commands import ModelCommands

        print("✅ ModelCommands imported successfully")

        from src.handlers.commands.callback_handlers import CallbackHandlers

        print("✅ CallbackHandlers imported successfully")

        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_provider_groups():
    """Test that provider groups are properly defined"""
    print("\n🧪 Testing provider groups...")

    try:
        from src.services.model_handlers.simple_api_manager import PROVIDER_GROUPS

        print(f"✅ Found {len(PROVIDER_GROUPS)} provider groups:")
        for group_name, group_info in PROVIDER_GROUPS.items():
            print(
                f"   - {group_name}: {group_info.get('description', 'No description')}"
            )

        return True
    except Exception as e:
        print(f"❌ Provider groups error: {e}")
        return False


def test_model_manager():
    """Test SuperSimpleAPIManager functionality"""
    print("\n🧪 Testing SuperSimpleAPIManager...")

    try:
        from src.services.model_handlers.simple_api_manager import SuperSimpleAPIManager

        # Create manager with mock APIs (None for testing)
        manager = SuperSimpleAPIManager(None, None, None)

        # Test get_models_by_category
        categories = manager.get_models_by_category()
        print(f"✅ Found {len(categories)} model categories:")
        for cat_id, cat_info in categories.items():
            model_count = len(cat_info.get("models", {}))
            print(f"   - {cat_info.get('name', cat_id)}: {model_count} models")

        return True
    except Exception as e:
        print(f"❌ Model manager error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🎯 HIERARCHICAL MODEL SELECTION SYSTEM TEST")
    print("=" * 50)

    tests = [test_imports, test_provider_groups, test_model_manager]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Hierarchical model selection system is ready!")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
