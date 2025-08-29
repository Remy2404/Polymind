#!/usr/bin/env python3
"""Test script to verify tool compatibility in model configurations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.model_handlers.model_configs import ModelConfigurations, get_default_agent_model

def test_tool_compatibility():
    """Test tool compatibility detection."""
    all_models = ModelConfigurations.get_all_models()
    tool_models = ModelConfigurations.get_tool_compatible_models()
    
    print(f"Total models: {len(all_models)}")
    print(f"Tool-compatible models: {len(tool_models)}")
    print()
    
    print("Tool-compatible models (first 10):")
    for i, (name, config) in enumerate(tool_models.items()):
        if i >= 10:
            break
        supports_tools = getattr(config, 'supports_tool_use', True)
        print(f"  {name}: {config.display_name} (supports_tool_use: {supports_tools})")
    
    print()
    print("Models that DON'T support tools:")
    non_tool_models = []
    for name, config in all_models.items():
        if not getattr(config, 'supports_tool_use', True):
            non_tool_models.append((name, config))
    
    for name, config in non_tool_models[:5]:
        print(f"  {name}: {config.display_name}")
    
    print()
    default = get_default_agent_model()
    default_supports_tools = getattr(default, 'supports_tool_use', True)
    print(f"Default model: {default.model_id}")
    print(f"Default supports tools: {default_supports_tools}")

if __name__ == "__main__":
    test_tool_compatibility()
