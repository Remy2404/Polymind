#!/usr/bin/env python3
"""Simple test to check model selection logic."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.model_handlers.model_configs import ModelConfigurations, get_default_agent_model

# Test tool-compatible models
tool_models = ModelConfigurations.get_tool_compatible_models()
print(f"Found {len(tool_models)} tool-compatible models")

# Check if native gemini is excluded
if 'gemini' in tool_models:
    print("ERROR: Native 'gemini' should not be in tool-compatible models")
else:
    print("✓ Native 'gemini' correctly excluded from tool-compatible models")

# Check if OpenRouter gemini is included
if 'gemini-2.0-flash-exp' in tool_models:
    print("✓ OpenRouter 'gemini-2.0-flash-exp' is in tool-compatible models")
else:
    print("ERROR: OpenRouter 'gemini-2.0-flash-exp' should be in tool-compatible models")

# Check default model
default = get_default_agent_model()
print(f"Default model: {default.model_id} (supports_tool_use: {getattr(default, 'supports_tool_use', True)})")

# List first few tool-compatible models
print("\nFirst 5 tool-compatible models:")
for i, (name, config) in enumerate(tool_models.items()):
    if i >= 5:
        break
    print(f"  {name}: {config.display_name}")
