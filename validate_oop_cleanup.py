
"""
Simple validation script for model_configs OOP cleanup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.services.model_handlers.model_configs import ModelConfigurations, Provider
    print("✓ Model configs import successful")

    # Test class instantiation
    mc = ModelConfigurations()
    print("✓ ModelConfigurations instance created")

    # Test provider enum
    providers = [p.value for p in Provider]
    print(f"✓ Available providers: {providers}")

    # Test class methods
    openrouter_models = ModelConfigurations.get_models_by_provider(Provider.OPENROUTER)
    print(f"✓ OpenRouter models loaded: {len(openrouter_models)} models")

    free_models = ModelConfigurations.get_free_models()
    print(f"✓ Free models available: {len(free_models)} models")

    all_models = ModelConfigurations.get_all_models()
    print(f"✓ All models loaded: {len(all_models)} models")

    # Test model config retrieval
    if openrouter_models:
        first_model_id = list(openrouter_models.keys())[0]
        config = ModelConfigurations.get_model_config(first_model_id)
        if config:
            print(f"✓ Model config retrieved for {first_model_id}")
        else:
            print(f"⚠ Could not retrieve config for {first_model_id}")

    print("\n✓ All model_configs functionality working correctly after OOP cleanup!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
