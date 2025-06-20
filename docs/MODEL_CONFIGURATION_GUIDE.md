# Model Configuration Management Guide

## Enhanced get_all_models() Functionality

The `ModelConfigurations.get_all_models()` method has been enhanced with additional utility methods to provide comprehensive model management capabilities.

## Available Methods

### Core Methods

#### `get_all_models()` → Dict[str, ModelConfig]
Returns all available model configurations.

```python
from src.services.model_handlers.model_configs import ModelConfigurations

all_models = ModelConfigurations.get_all_models()
print(f"Total models: {len(all_models)}")
```

#### `get_all_models_sorted(sort_by: str)` → Dict[str, ModelConfig]
Returns all models sorted by specified criteria.

```python
# Sort by display name
sorted_models = ModelConfigurations.get_all_models_sorted('display_name')

# Sort by provider
sorted_models = ModelConfigurations.get_all_models_sorted('provider')

# Sort by model ID
sorted_models = ModelConfigurations.get_all_models_sorted('model_id')
```

### Filtering Methods

#### `get_models_by_provider(provider: Provider)` → Dict[str, ModelConfig]
Get models from a specific provider.

```python
from src.services.model_handlers.model_configs import Provider

openrouter_models = ModelConfigurations.get_models_by_provider(Provider.OPENROUTER)
gemini_models = ModelConfigurations.get_models_by_provider(Provider.GEMINI)
```

#### `get_models_by_capability(capability: str)` → Dict[str, ModelConfig]
Get models that support specific capabilities.

```python
# Get vision-capable models
vision_models = ModelConfigurations.get_models_by_capability('images')

# Get models that support documents
doc_models = ModelConfigurations.get_models_by_capability('documents')

# Get all multimodal models
multimodal_models = ModelConfigurations.get_models_by_capability('multimodal')
```

#### `get_free_models()` → Dict[str, ModelConfig]
Get all free OpenRouter models.

```python
free_models = ModelConfigurations.get_free_models()
print(f"Available free models: {len(free_models)}")
```

### Analysis Methods

#### `get_model_categories()` → Dict[str, List[str]]
Organize models by categories based on their capabilities and names.

```python
categories = ModelConfigurations.get_model_categories()
for category, models in categories.items():
    print(f"{category}: {len(models)} models")
```

Categories include:
- **reasoning**: Models optimized for logical reasoning
- **coding**: Programming and development-focused models
- **vision**: Image and visual content processing
- **conversation**: Chat and dialogue optimization
- **large_models**: Models with 70B+ parameters
- **efficient_models**: Lightweight models (≤8B parameters)
- **specialized**: Domain-specific or unique models

#### `get_model_stats()` → Dict[str, Any]
Get comprehensive statistics about available models.

```python
stats = ModelConfigurations.get_model_stats()
print(f"Total models: {stats['total_models']}")
print(f"Free models: {stats['free_models']}")
print(f"Provider distribution: {stats['provider_distribution']}")
print(f"Capability distribution: {stats['capability_distribution']}")
```

### Search and Validation

#### `search_models(query: str)` → Dict[str, ModelConfig]
Search models by name, description, or provider.

```python
# Search for reasoning models
reasoning_models = ModelConfigurations.search_models('reasoning')

# Search for Qwen models
qwen_models = ModelConfigurations.search_models('qwen')

# Search for vision models
vision_models = ModelConfigurations.search_models('vision')
```

#### `validate_model_config(model_id: str)` → Dict[str, Any]
Validate a specific model configuration.

```python
validation = ModelConfigurations.validate_model_config('gemini')
if validation['valid']:
    print("Model configuration is valid")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
```

### Export Methods

#### `export_model_list(format_type: str)` → Any
Export model list in different formats.

```python
# Simple list of model IDs
simple_list = ModelConfigurations.export_model_list('simple')

# Detailed information
detailed = ModelConfigurations.export_model_list('detailed')

# JSON format
json_output = ModelConfigurations.export_model_list('json')

# Markdown documentation
markdown = ModelConfigurations.export_model_list('markdown')
```

## Usage Examples

### Example 1: Find Best Model for Task

```python
def find_best_model_for_task(task_type: str):
    if task_type == "vision":
        models = ModelConfigurations.get_models_by_capability('images')
        # Filter for free models
        free_vision = {k: v for k, v in models.items() 
                      if k in ModelConfigurations.get_free_models()}
        return free_vision
    
    elif task_type == "reasoning":
        return ModelConfigurations.search_models('reasoning')
    
    elif task_type == "coding":
        return ModelConfigurations.get_model_categories()['coding']
```

### Example 2: Model Recommendation System

```python
def recommend_models(user_requirements):
    recommendations = {}
    
    # Check if user needs vision capabilities
    if user_requirements.get('needs_vision'):
        vision_models = ModelConfigurations.get_models_by_capability('images')
        recommendations['vision'] = list(vision_models.keys())[:3]
    
    # Check if user wants free models only
    if user_requirements.get('free_only'):
        free_models = ModelConfigurations.get_free_models()
        recommendations['free'] = list(free_models.keys())[:5]
    
    # Check for specific capabilities
    if user_requirements.get('task_type'):
        task_models = ModelConfigurations.search_models(user_requirements['task_type'])
        recommendations['task_specific'] = list(task_models.keys())[:3]
    
    return recommendations
```

### Example 3: Model Comparison

```python
def compare_models(model_ids: List[str]):
    comparison = {}
    
    for model_id in model_ids:
        validation = ModelConfigurations.validate_model_config(model_id)
        all_models = ModelConfigurations.get_all_models()
        
        if model_id in all_models:
            config = all_models[model_id]
            comparison[model_id] = {
                'display_name': config.display_name,
                'provider': config.provider.value,
                'capabilities': {
                    'images': config.supports_images,
                    'documents': config.supports_documents,
                    'audio': config.supports_audio,
                    'video': config.supports_video
                },
                'max_tokens': config.max_tokens,
                'is_free': model_id in ModelConfigurations.get_free_models(),
                'validation': validation
            }
    
    return comparison
```

## Model Statistics Summary

Based on the current configuration:

- **Total Models**: 54
- **Free Models**: 52 (96% of all models)
- **Provider Distribution**:
  - OpenRouter: 52 models (96%)
  - Gemini: 1 model (2%)
  - DeepSeek: 1 model (2%)
- **Capability Distribution**:
  - Vision-capable: 8 models
  - Document support: 1 model
  - Audio support: 0 models
  - Video support: 0 models

## Model Categories Breakdown

- **Reasoning Models**: 13 models - Specialized in logical thinking and problem-solving
- **Efficient Models**: 14 models - Lightweight options (≤8B parameters)
- **Specialized Models**: 11 models - Domain-specific or unique capabilities
- **Vision Models**: 8 models - Image and visual content processing
- **Large Models**: 4 models - High-capacity models (≥70B parameters)
- **Coding Models**: 3 models - Programming and development focus
- **Conversation Models**: 1 model - Chat optimization

This enhanced system provides comprehensive model management capabilities for the Telegram bot, enabling intelligent model selection based on user needs and task requirements.
