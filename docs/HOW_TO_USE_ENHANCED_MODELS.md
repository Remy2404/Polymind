# How to Use Enhanced get_all_models() Features

## Quick Start Guide

### 1. Basic Usage - Get All Models

```python
from src.services.model_handlers.model_configs import ModelConfigurations

# Get all available models
all_models = ModelConfigurations.get_all_models()
print(f"Total models available: {len(all_models)}")

# Access a specific model
gemini_config = all_models['gemini']
print(f"Gemini display name: {gemini_config.display_name}")
print(f"Supports images: {gemini_config.supports_images}")
```

### 2. Find Models for Specific Tasks

#### For Image/Vision Tasks:
```python
# Get all vision-capable models
vision_models = ModelConfigurations.get_models_by_capability('images')
print("Vision models:")
for model_id, config in vision_models.items():
    print(f"  {config.indicator_emoji} {config.display_name} ({model_id})")
```

#### For Programming/Coding:
```python
# Find coding-specialized models
coding_models = ModelConfigurations.search_models('code')
# Or get from categories
categories = ModelConfigurations.get_model_categories()
coding_category = categories['coding']
print(f"Coding models: {coding_category}")
```

#### For Complex Reasoning:
```python
# Get reasoning-focused models
reasoning_models = ModelConfigurations.search_models('reasoning')
print(f"Found {len(reasoning_models)} reasoning models")
```

### 3. Filter Models by Cost (Free vs Paid)

```python
# Get only free models
free_models = ModelConfigurations.get_free_models()
print(f"Free models available: {len(free_models)}")

# Get free vision models
free_vision = {k: v for k, v in vision_models.items() if k in free_models}
print("Free vision models:")
for model_id in free_vision:
    print(f"  - {model_id}")
```

### 4. Search for Specific Models

```python
# Search for DeepSeek models
deepseek_models = ModelConfigurations.search_models('deepseek')
print(f"DeepSeek models: {list(deepseek_models.keys())}")

# Search for Qwen models
qwen_models = ModelConfigurations.search_models('qwen')
print(f"Qwen models: {list(qwen_models.keys())}")

# Search for lightweight models
efficient_models = ModelConfigurations.search_models('8b')
```

### 5. Get Model Statistics and Overview

```python
# Get comprehensive statistics
stats = ModelConfigurations.get_model_stats()
print(f"Total models: {stats['total_models']}")
print(f"Free models: {stats['free_models']}")
print(f"Provider breakdown: {stats['provider_distribution']}")
print(f"Vision models: {stats['capability_distribution']['images']}")
```

### 6. Organize Models by Categories

```python
# Get all model categories
categories = ModelConfigurations.get_model_categories()
for category_name, model_list in categories.items():
    if model_list:  # Only show non-empty categories
        print(f"{category_name.title()}: {len(model_list)} models")
        print(f"  Examples: {', '.join(model_list[:3])}")
```

### 7. Sort Models for Better Organization

```python
# Sort by display name (alphabetical)
sorted_by_name = ModelConfigurations.get_all_models_sorted('display_name')
first_five = list(sorted_by_name.items())[:5]
print("First 5 models alphabetically:")
for model_id, config in first_five:
    print(f"  {config.display_name}")

# Sort by provider
sorted_by_provider = ModelConfigurations.get_all_models_sorted('provider')
```

### 8. Validate Model Configurations

```python
# Check if a model is properly configured
model_to_check = 'gemini'
validation = ModelConfigurations.validate_model_config(model_to_check)

if validation['valid']:
    print(f"✅ {model_to_check} is valid")
    print(f"Info: {validation['info']}")
else:
    print(f"❌ {model_to_check} has issues: {validation['error']}")
```

### 9. Export Model Information

```python
# Get simple list of model IDs
model_ids = ModelConfigurations.export_model_list('simple')
print(f"All model IDs: {model_ids[:10]}...")  # First 10

# Get detailed information
detailed_info = ModelConfigurations.export_model_list('detailed')
gemini_details = detailed_info['gemini']
print(f"Gemini details: {gemini_details}")

# Generate markdown documentation
markdown_docs = ModelConfigurations.export_model_list('markdown')
# Save to file or use for documentation
```

## Real-World Use Cases

### Use Case 1: Smart Model Selection for User Requests

```python
def recommend_model_for_user(user_request: str, needs_free: bool = True):
    """Recommend the best model based on user request"""
    
    # Determine task type from request
    if any(word in user_request.lower() for word in ['image', 'picture', 'photo', 'visual']):
        candidates = ModelConfigurations.get_models_by_capability('images')
        task_type = "vision"
    elif any(word in user_request.lower() for word in ['code', 'program', 'debug', 'script']):
        categories = ModelConfigurations.get_model_categories()
        candidates = {k: ModelConfigurations.get_all_models()[k] for k in categories['coding']}
        task_type = "coding"
    elif any(word in user_request.lower() for word in ['math', 'solve', 'calculate', 'reason']):
        candidates = ModelConfigurations.search_models('reasoning')
        task_type = "reasoning"
    else:
        # General purpose - use efficient models
        categories = ModelConfigurations.get_model_categories()
        candidates = {k: ModelConfigurations.get_all_models()[k] for k in categories['efficient_models']}
        task_type = "general"
    
    # Filter for free models if requested
    if needs_free:
        free_models = ModelConfigurations.get_free_models()
        candidates = {k: v for k, v in candidates.items() if k in free_models}
    
    # Return top recommendation
    if candidates:
        best_model = list(candidates.keys())[0]
        return best_model, task_type
    else:
        return 'llama-3.3-8b', 'fallback'  # Default fallback

# Example usage
model, task = recommend_model_for_user("Help me analyze this image", needs_free=True)
print(f"Recommended {model} for {task} task")
```

### Use Case 2: Model Comparison Tool

```python
def compare_models_for_task(task_type: str, max_models: int = 3):
    """Compare top models for a specific task"""
    
    if task_type == 'vision':
        candidates = ModelConfigurations.get_models_by_capability('images')
    elif task_type == 'reasoning':
        candidates = ModelConfigurations.search_models('reasoning')
    elif task_type == 'coding':
        categories = ModelConfigurations.get_model_categories()
        candidates = {k: ModelConfigurations.get_all_models()[k] for k in categories['coding']}
    else:
        candidates = ModelConfigurations.get_all_models()
    
    # Get free models only
    free_models = ModelConfigurations.get_free_models()
    free_candidates = {k: v for k, v in candidates.items() if k in free_models}
    
    print(f"\n🔍 Top {max_models} FREE models for {task_type}:")
    print("-" * 50)
    
    for i, (model_id, config) in enumerate(list(free_candidates.items())[:max_models], 1):
        validation = ModelConfigurations.validate_model_config(model_id)
        print(f"{i}. {config.indicator_emoji} {config.display_name}")
        print(f"   ID: {model_id}")
        print(f"   Provider: {config.provider.value}")
        print(f"   Max Tokens: {config.max_tokens}")
        print(f"   Description: {config.description}")
        print()

# Example usage
compare_models_for_task('vision')
compare_models_for_task('reasoning')
```

### Use Case 3: Dynamic Model Loading for Bot

```python
class BotModelManager:
    def __init__(self):
        self.all_models = ModelConfigurations.get_all_models()
        self.free_models = ModelConfigurations.get_free_models()
        self.categories = ModelConfigurations.get_model_categories()
    
    def get_model_for_capability(self, capability: str, prefer_free: bool = True):
        """Get best model for specific capability"""
        models = ModelConfigurations.get_models_by_capability(capability)
        
        if prefer_free:
            models = {k: v for k, v in models.items() if k in self.free_models}
        
        if models:
            # Return the first available model
            return list(models.keys())[0]
        return None
    
    def get_fallback_models(self, count: int = 3):
        """Get reliable fallback models"""
        efficient = self.categories.get('efficient_models', [])
        free_efficient = [m for m in efficient if m in self.free_models]
        return free_efficient[:count]
    
    def is_model_available(self, model_id: str):
        """Check if model is available and valid"""
        validation = ModelConfigurations.validate_model_config(model_id)
        return validation['valid']

# Example usage in bot
bot_manager = BotModelManager()
vision_model = bot_manager.get_model_for_capability('images')
fallbacks = bot_manager.get_fallback_models()
print(f"Vision model: {vision_model}")
print(f"Fallback models: {fallbacks}")
```

## Integration with Existing Code

### In Message Handlers:

```python
# In your message handler
def handle_user_request(request_type, user_preferences):
    # Get appropriate model based on request
    if request_type == 'image_analysis':
        models = ModelConfigurations.get_models_by_capability('images')
        free_models = ModelConfigurations.get_free_models()
        available = [m for m in models if m in free_models]
        selected_model = available[0] if available else 'gemini'
    
    elif request_type == 'code_help':
        categories = ModelConfigurations.get_model_categories()
        selected_model = categories['coding'][0] if categories['coding'] else 'deepcoder'
    
    else:
        # Use efficient general-purpose model
        categories = ModelConfigurations.get_model_categories()
        selected_model = categories['efficient_models'][0]
    
    return selected_model
```

### In Configuration Loading:

```python
# At startup, get model statistics
def initialize_bot():
    stats = ModelConfigurations.get_model_stats()
    logger.info(f"Bot initialized with {stats['total_models']} models")
    logger.info(f"Free models available: {stats['free_models']}")
    
    # Validate critical models
    critical_models = ['gemini', 'llama-3.3-8b', 'qwen3-32b']
    for model in critical_models:
        validation = ModelConfigurations.validate_model_config(model)
        if validation['valid']:
            logger.info(f"✅ {model} validated successfully")
        else:
            logger.warning(f"⚠️ {model} validation failed")
```

## Quick Reference Commands

```python
# Most common operations:

# 1. Get all models
ModelConfigurations.get_all_models()

# 2. Get free models only
ModelConfigurations.get_free_models()

# 3. Find vision models
ModelConfigurations.get_models_by_capability('images')

# 4. Search for specific models
ModelConfigurations.search_models('reasoning')

# 5. Get model categories
ModelConfigurations.get_model_categories()

# 6. Get statistics
ModelConfigurations.get_model_stats()

# 7. Validate a model
ModelConfigurations.validate_model_config('model_id')

# 8. Export model list
ModelConfigurations.export_model_list('simple')
```

That's how you use the enhanced `get_all_models()` features! The system makes it easy to find the right model for any task.
