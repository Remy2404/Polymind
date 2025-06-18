# Unified Model System Usage Guide

## Overview
The new unified model system simplifies API management and makes it easy to add new models or entire APIs. Here's how it works:

## Architecture

### Core Components:
1. **`unified_handler.py`** - Single handler that works with any API
2. **`model_configs.py`** - Centralized configuration for all models  
3. **`factory.py`** - Simplified factory using configurations
4. **`model_commands.py`** - Auto-generated UI from configurations

## Key Benefits

✅ **Reduced File Count**: No more separate handler files for each model  
✅ **Easy Management**: Add models by editing one config file  
✅ **Consistent Interface**: All models work the same way  
✅ **Auto UI Generation**: Model switching UI updates automatically  
✅ **Type Safety**: Proper typing and validation  

## How to Add New Models

### Adding OpenRouter Models
```python
# In model_configs.py, just add to the get_all_models() return dict:

"new-model": ModelConfig(
    model_id="new-model",
    display_name="New Model Name",
    provider=Provider.OPENROUTER,
    openrouter_model_key="provider/model-name:free",
    indicator_emoji="🎯",
    system_message="Custom system message for this model",
    description="Brief description"
),
```

### Adding a New API Provider
```python
# 1. Add to Provider enum in model_configs.py:
class Provider(Enum):
    GEMINI = "gemini"
    OPENROUTER = "openrouter" 
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"  # New provider

# 2. Add handler logic in unified_handler.py:
async def generate_response(self, ...):
    if self.provider == "anthropic":
        return await self._handle_anthropic_request(...)
    # ... existing code

async def _handle_anthropic_request(self, ...):
    # Implementation for new API
    return await self.api_instance.generate_response(...)

# 3. Update factory.py provider logic:
elif model_config.provider == Provider.ANTHROPIC:
    if anthropic_api is None:
        raise ValueError(f"AnthropicAPI instance required for: {model_name}")
    api_instance = anthropic_api
```

## Current Available Models

### Gemini (1 model)
- ✨ Gemini 2.0 Flash - Google's latest multimodal AI

### DeepSeek (1 model) 
- 🧠 DeepSeek R1 - Advanced reasoning model

### OpenRouter Free (8+ models)
- 🧠 DeepSeek R1 Qwen3 8B - Latest DeepSeek with Qwen3 base
- 🔬 DeepSeek R1 Zero - RL-trained reasoning model
- 💻 DeepCoder 14B - Programming specialist
- 🦙 Llama-4 Maverick - Meta's latest model
- 👁️ Llama 3.2 11B Vision - Vision-capable model
- 🌟 Qwen3 32B - Large parameter model
- 🌊 Mistral Small 3 - Latest European AI
- 💎 Gemma 2 9B - Google's efficient model
- 🔬 Phi-4 Reasoning - Microsoft's reasoning model
- 🏆 OlympicCoder 32B - Competitive programming

## Easy Model Switching

Users can switch models with:
- `/switchmodel` - Interactive model selection
- `/listmodels` - View all available models

The UI automatically updates when you add new models to the configuration!

## Migration Benefits

### Before (Complex):
- 10+ separate handler files
- Hardcoded model lists in multiple places
- Manual UI updates for each new model
- Inconsistent interfaces

### After (Simple):
- 1 unified handler
- 1 configuration file
- Auto-updating UI
- Consistent API interface

## Example: Adding 10 New OpenRouter Models

Just add them to `model_configs.py`:
```python
# Add these to get_all_models() return dict:
"claude-3-haiku": ModelConfig(...),
"gpt-3.5-turbo": ModelConfig(...),
"cohere-command": ModelConfig(...),
# ... 7 more models
```

That's it! The UI, factory, and handlers all update automatically.

## File Structure

```
src/services/model_handlers/
├── __init__.py              # Base ModelHandler interface
├── unified_handler.py       # Single handler for all APIs
├── model_configs.py         # All model configurations
├── factory.py              # Simplified factory
└── [REMOVED FILES]:
    ├── gemini_handler.py    # No longer needed
    ├── deepseek_handler.py  # No longer needed
    ├── openrouter_models.py # No longer needed
    └── model_registry.py    # No longer needed
```

The system is now **50% fewer files** and **infinitely more maintainable**! 🚀
