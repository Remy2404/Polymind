#!/usr/bin/env python3
"""
Interactive Demo: How to Use Enhanced get_all_models() Features
Run this script to see live examples of all enhanced functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.model_handlers.model_configs import ModelConfigurations, Provider

def demo_basic_usage():
    """Demo 1: Basic model retrieval and information"""
    print("🔍 DEMO 1: Basic Model Usage")
    print("=" * 50)
    
    # Get all models
    all_models = ModelConfigurations.get_all_models()
    print(f"📊 Total models available: {len(all_models)}")
    
    # Show first few models
    print("\n📋 First 5 models:")
    for i, (model_id, config) in enumerate(list(all_models.items())[:5], 1):
        print(f"  {i}. {config.indicator_emoji} {config.display_name} ({model_id})")
    
    # Access specific model details
    print("\n🔎 Gemini model details:")
    gemini = all_models['gemini']
    print(f"  Display Name: {gemini.display_name}")
    print(f"  Provider: {gemini.provider.value}")
    print(f"  Supports Images: {gemini.supports_images}")
    print(f"  Max Tokens: {gemini.max_tokens}")

def demo_task_specific_search():
    """Demo 2: Finding models for specific tasks"""
    print("\n\n🎯 DEMO 2: Task-Specific Model Search")
    print("=" * 50)
    
    # Vision models
    print("👁️ Vision-capable models:")
    vision_models = ModelConfigurations.get_models_by_capability('images')
    for model_id, config in list(vision_models.items())[:3]:
        print(f"  • {config.indicator_emoji} {config.display_name}")
    
    # Reasoning models
    print("\n🧠 Reasoning models:")
    reasoning_models = ModelConfigurations.search_models('reasoning')
    for model_id in list(reasoning_models.keys())[:3]:
        config = reasoning_models[model_id]
        print(f"  • {config.indicator_emoji} {config.display_name}")
    
    # Coding models
    print("\n💻 Coding models:")
    coding_models = ModelConfigurations.search_models('code')
    for model_id in list(coding_models.keys())[:3]:
        config = coding_models[model_id]
        print(f"  • {config.indicator_emoji} {config.display_name}")

def demo_free_models():
    """Demo 3: Working with free models"""
    print("\n\n💰 DEMO 3: Free Models")
    print("=" * 50)
    
    free_models = ModelConfigurations.get_free_models()
    print(f"🆓 Total free models: {len(free_models)}")
    
    # Free vision models
    vision_models = ModelConfigurations.get_models_by_capability('images')
    free_vision = {k: v for k, v in vision_models.items() if k in free_models}
    print(f"\n👁️ Free vision models: {len(free_vision)}")
    for model_id in list(free_vision.keys())[:3]:
        print(f"  • {model_id}")
    
    # Show cost breakdown
    all_models = ModelConfigurations.get_all_models()
    paid_models = len(all_models) - len(free_models)
    free_percentage = (len(free_models) / len(all_models)) * 100
    print(f"\n📊 Cost breakdown:")
    print(f"  Free: {len(free_models)} models ({free_percentage:.1f}%)")
    print(f"  Paid: {paid_models} models ({100-free_percentage:.1f}%)")

def demo_categories():
    """Demo 4: Model categories"""
    print("\n\n📂 DEMO 4: Model Categories")
    print("=" * 50)
    
    categories = ModelConfigurations.get_model_categories()
    print("🏷️ Available categories:")
    
    for category, models in categories.items():
        if models:  # Only show non-empty categories
            print(f"\n  📋 {category.title().replace('_', ' ')}: {len(models)} models")
            # Show examples
            examples = models[:3]  # First 3 as examples
            print(f"     Examples: {', '.join(examples)}")

def demo_search_functionality():
    """Demo 5: Search capabilities"""
    print("\n\n🔍 DEMO 5: Search Functionality")
    print("=" * 50)
    
    search_terms = ['qwen', 'deepseek', 'llama', 'reasoning']
    
    for term in search_terms:
        results = ModelConfigurations.search_models(term)
        print(f"\n🔎 Search for '{term}': {len(results)} results")
        if results:
            # Show first 3 results
            for model_id in list(results.keys())[:3]:
                config = results[model_id]
                print(f"  • {config.display_name} ({model_id})")

def demo_statistics():
    """Demo 6: Model statistics"""
    print("\n\n📈 DEMO 6: Model Statistics")
    print("=" * 50)
    
    stats = ModelConfigurations.get_model_stats()
    
    print("📊 Overall Statistics:")
    print(f"  Total Models: {stats['total_models']}")
    print(f"  Free Models: {stats['free_models']}")
    
    print("\n🏢 Provider Distribution:")
    for provider, count in stats['provider_distribution'].items():
        percentage = (count / stats['total_models']) * 100
        print(f"  {provider.title()}: {count} ({percentage:.1f}%)")
    
    print("\n🎛️ Capability Distribution:")
    for capability, count in stats['capability_distribution'].items():
        if count > 0:
            print(f"  {capability.title()}: {count} models")
    
    print(f"\n📂 Category Breakdown:")
    for category, count in stats['categories'].items():
        if count > 0:
            print(f"  {category.replace('_', ' ').title()}: {count} models")

def demo_validation():
    """Demo 7: Model validation"""
    print("\n\n✅ DEMO 7: Model Validation")
    print("=" * 50)
    
    test_models = ['gemini', 'llama-3.3-8b', 'qwen3-32b', 'nonexistent-model']
    
    print("🔍 Validating models:")
    for model_id in test_models:
        validation = ModelConfigurations.validate_model_config(model_id)
        status = "✅" if validation['valid'] else "❌"
        print(f"  {status} {model_id}")
        
        if validation['valid']:
            info = validation.get('info', {})
            print(f"     Provider: {info.get('provider', 'Unknown')}")
            print(f"     Max Tokens: {info.get('max_tokens', 'Unknown')}")
            if validation.get('warnings'):
                print(f"     ⚠️ Warnings: {len(validation['warnings'])}")

def demo_export():
    """Demo 8: Export functionality"""
    print("\n\n📤 DEMO 8: Export Formats")
    print("=" * 50)
    
    # Simple export
    simple_list = ModelConfigurations.export_model_list('simple')
    print(f"📝 Simple format: {len(simple_list)} model IDs")
    print(f"   Examples: {', '.join(simple_list[:5])}")
    
    # Detailed export
    detailed = ModelConfigurations.export_model_list('detailed')
    print(f"\n📋 Detailed format: Full information for {len(detailed)} models")
    
    # Show example detailed info
    first_model = list(detailed.keys())[0]
    first_details = detailed[first_model]
    print(f"   Example ({first_model}):")
    print(f"     Display: {first_details['display_name']}")
    print(f"     Provider: {first_details['provider']}")
    print(f"     Capabilities: {sum(first_details['capabilities'].values())} types")

def demo_sorting():
    """Demo 9: Sorting capabilities"""
    print("\n\n🔄 DEMO 9: Sorting Models")
    print("=" * 50)
    
    # Sort by name
    sorted_by_name = ModelConfigurations.get_all_models_sorted('display_name')
    first_three_names = [config.display_name for config in list(sorted_by_name.values())[:3]]
    print("📝 First 3 models alphabetically:")
    for i, name in enumerate(first_three_names, 1):
        print(f"  {i}. {name}")
    
    # Sort by provider
    sorted_by_provider = ModelConfigurations.get_all_models_sorted('provider')
    provider_groups = {}
    for config in sorted_by_provider.values():
        provider = config.provider.value
        if provider not in provider_groups:
            provider_groups[provider] = 0
        provider_groups[provider] += 1
    
    print(f"\n🏢 Models grouped by provider:")
    for provider, count in provider_groups.items():
        print(f"  {provider.title()}: {count} models")

def demo_practical_example():
    """Demo 10: Practical usage example"""
    print("\n\n🛠️ DEMO 10: Practical Example - Smart Model Recommendation")
    print("=" * 50)
    
    def recommend_model(task_description: str):
        """Smart model recommendation based on task"""
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ['image', 'picture', 'visual', 'photo']):
            candidates = ModelConfigurations.get_models_by_capability('images')
            task_type = "Vision"
        elif any(word in task_lower for word in ['code', 'program', 'debug']):
            categories = ModelConfigurations.get_model_categories()
            all_models = ModelConfigurations.get_all_models()
            candidates = {k: all_models[k] for k in categories.get('coding', [])}
            task_type = "Coding"
        elif any(word in task_lower for word in ['math', 'solve', 'reason', 'logic']):
            candidates = ModelConfigurations.search_models('reasoning')
            task_type = "Reasoning"
        else:
            categories = ModelConfigurations.get_model_categories()
            all_models = ModelConfigurations.get_all_models()
            candidates = {k: all_models[k] for k in categories.get('efficient_models', [])[:5]}
            task_type = "General"
        
        # Filter for free models
        free_models = ModelConfigurations.get_free_models()
        free_candidates = {k: v for k, v in candidates.items() if k in free_models}
        
        if free_candidates:
            best_model = list(free_candidates.keys())[0]
            config = free_candidates[best_model]
            return best_model, config, task_type
        else:
            return None, None, task_type
    
    # Test with different tasks
    test_tasks = [
        "Help me analyze this image",
        "Debug my Python code", 
        "Solve this math problem",
        "Write a story"
    ]
    
    print("🎯 Smart recommendations:")
    for task in test_tasks:
        model_id, config, task_type = recommend_model(task)
        if model_id:
            print(f"\n  📋 Task: '{task}'")
            print(f"     Type: {task_type}")
            print(f"     Recommended: {config.indicator_emoji} {config.display_name}")
            print(f"     Model ID: {model_id}")
        else:
            print(f"\n  📋 Task: '{task}' - No suitable free model found")

def main():
    """Run all demos"""
    print("🚀 Enhanced get_all_models() Features - Interactive Demo")
    print("=" * 60)
    print("This demo shows you how to use all the enhanced model management features.")
    print()
    
    # Run all demos
    demo_basic_usage()
    demo_task_specific_search()
    demo_free_models()
    demo_categories()
    demo_search_functionality()
    demo_statistics()
    demo_validation()
    demo_export()
    demo_sorting()
    demo_practical_example()
    
    print("\n\n🎉 Demo Complete!")
    print("=" * 60)
    print("You can now use these features in your own code!")
    print("Check the documentation at: docs/HOW_TO_USE_ENHANCED_MODELS.md")

if __name__ == "__main__":
    main()
