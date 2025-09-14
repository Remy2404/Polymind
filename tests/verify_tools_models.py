"""
Verify tools.json models are in model_configs.py with correct tool calling support
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_tools_json():
    """Load the tested tool-calling models from tools.json"""
    with open("tools.json", "r", encoding="utf-8") as f:
        return json.load(f)


def check_tools_models_in_configs():
    """Check if all tools.json models are in model_configs.py with tool calling support"""
    print("🔍 Verifying tools.json models in model_configs.py")
    print("=" * 60)

    try:
        from src.services.model_handlers.model_configs import ModelConfigurations

        # Get all models and tool-calling models from configs
        all_models = ModelConfigurations.get_all_models()
        tool_call_models = ModelConfigurations.get_models_with_tool_calls()

        # Load tested tool-calling models
        tools_models = load_tools_json()

        print(f"📄 models in tools.json (tested tool calling): {len(tools_models)}")
        print(f"📦 Total models in model_configs.py: {len(all_models)}")
        print(f"🛠️  Tool-calling models in model_configs.py: {len(tool_call_models)}")

        print("\n🔍 Checking each tools.json model:")
        print("-" * 40)

        missing_models = []
        incorrect_tool_support = []
        correct_models = []

        for tool_model in tools_models:
            openrouter_id = tool_model["id"]
            name = tool_model["name"]

            # Find matching model in configs by OpenRouter key
            found_model = None
            found_model_id = None

            for model_id, config in all_models.items():
                if (
                    hasattr(config, "openrouter_model_key")
                    and config.openrouter_model_key
                    and config.openrouter_model_key == openrouter_id
                ):
                    found_model = config
                    found_model_id = model_id
                    break

            if not found_model:
                missing_models.append(
                    {
                        "openrouter_id": openrouter_id,
                        "name": name,
                        "context_length": tool_model.get("context_length", 32000),
                    }
                )
                print(f"❌ MISSING: {openrouter_id}")
                print(f"   Name: {name}")
            else:
                # Check if it has tool calling support
                has_tool_support = found_model_id in tool_call_models

                if has_tool_support:
                    correct_models.append(found_model_id)
                    print(f"✅ CORRECT: {found_model_id}")
                    print(f"   OpenRouter: {openrouter_id}")
                    print("   Tool Support: Yes")
                else:
                    incorrect_tool_support.append(
                        {
                            "model_id": found_model_id,
                            "openrouter_id": openrouter_id,
                            "name": name,
                        }
                    )
                    print(f"⚠️  WRONG TOOL SUPPORT: {found_model_id}")
                    print(f"   OpenRouter: {openrouter_id}")
                    print("   Should support tools but doesn't")

        print("\n" + "=" * 60)
        print("📊 Summary:")
        print(f"✅ Correct models: {len(correct_models)}")
        print(f"❌ Missing models: {len(missing_models)}")
        print(f"⚠️  Wrong tool support: {len(incorrect_tool_support)}")

        if missing_models:
            print("\n❌ Missing models that need to be added:")
            for model in missing_models:
                print(f"   • {model['openrouter_id']}")
                print(f"     Name: {model['name']}")

        if incorrect_tool_support:
            print("\n⚠️  Models with wrong tool support (should support tools):")
            for model in incorrect_tool_support:
                print(f"   • {model['model_id']} -> {model['openrouter_id']}")

        return {
            "missing": missing_models,
            "incorrect_tool_support": incorrect_tool_support,
            "correct": correct_models,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    print("🧪 Tool-Calling Models Verification")
    print("Checking if all tested tool-calling models from tools.json")
    print("are properly configured in model_configs.py")
    print()

    result = check_tools_models_in_configs()

    if result:
        if not result["missing"] and not result["incorrect_tool_support"]:
            print("\n🎉 All tools.json models are correctly configured!")
        else:
            print("\n⚡ Action needed:")
            if result["missing"]:
                print(f"   - Add {len(result['missing'])} missing models")
            if result["incorrect_tool_support"]:
                print(
                    f"   - Fix tool support for {len(result['incorrect_tool_support'])} models"
                )


if __name__ == "__main__":
    main()
