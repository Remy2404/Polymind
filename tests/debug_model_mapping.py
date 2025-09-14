#!/usr/bin/env python3
"""
Debug script for model mapping issue
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.services.model_handlers.model_configs import ModelConfigurations, Provider

    print("=== Model Configuration Debug ===")

    # Test if model exists
    model_id = "qwen-2.5-72b-instruct"
    print(f"Testing model: {model_id}")

    # Get all models
    all_models = ModelConfigurations.get_all_models()
    print(f"Total models loaded: {len(all_models)}")

    # Check if our model exists
    if model_id in all_models:
        model_config = all_models[model_id]
        print(f"✓ Model found: {model_config.display_name}")
        print(f"  Provider: {model_config.provider.value}")
        print(f"  OpenRouter key: {model_config.openrouter_model_key}")
    else:
        print(f"✗ Model '{model_id}' not found in configuration")
        print("Available models:")
        for mid in sorted(all_models.keys())[:10]:  # Show first 10
            print(f"  - {mid}")
        print("  ...")

    # Test get_model_config method
    config = ModelConfigurations.get_model_config(model_id)
    if config:
        print(f"✓ get_model_config() found model: {config.display_name}")
        print(f"  OpenRouter key: {config.openrouter_model_key}")
    else:
        print(f"✗ get_model_config() returned None for {model_id}")

    # Test get_model_with_fallback
    fallback_result = ModelConfigurations.get_model_with_fallback(model_id)
    print(f"get_model_with_fallback() result: {fallback_result}")

    # Test OpenRouter models specifically
    openrouter_models = ModelConfigurations.get_models_by_provider(Provider.OPENROUTER)
    print(f"\nOpenRouter models count: {len(openrouter_models)}")

    # Check if our model is in OpenRouter models
    if model_id in openrouter_models:
        or_config = openrouter_models[model_id]
        print(f"✓ Model in OpenRouter list: {or_config.openrouter_model_key}")
    else:
        print("✗ Model not in OpenRouter provider list")

    print("\n=== Debug Complete ===")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
