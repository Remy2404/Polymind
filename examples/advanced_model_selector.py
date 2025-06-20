

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.model_handlers.model_configs import ModelConfigurations, Provider
from typing import Dict, List, Any, Optional
import json

class ModelSelector:
    """Intelligent model selection utility"""
    
    def __init__(self):
        self.all_models = ModelConfigurations.get_all_models()
        self.categories = ModelConfigurations.get_model_categories()
        self.free_models = ModelConfigurations.get_free_models()
        
    def recommend_for_task(self, task_type: str, requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """Recommend models based on task type and requirements"""
        if requirements is None:
            requirements = {}
            
        recommendations = {
            'primary': [],
            'alternatives': [],
            'reasoning': '',
            'task_type': task_type
        }
        
        # Task-specific recommendations
        if task_type.lower() in ['image', 'vision', 'visual', 'photo']:
            vision_models = ModelConfigurations.get_models_by_capability('images')
            primary_candidates = self._filter_by_requirements(vision_models, requirements)
            recommendations['primary'] = list(primary_candidates.keys())[:3]
            recommendations['reasoning'] = "Selected vision-capable models for image processing tasks"
            
        elif task_type.lower() in ['reasoning', 'logic', 'math', 'problem']:
            reasoning_models = {k: v for k, v in self.all_models.items() 
                              if k in self.categories.get('reasoning', [])}
            primary_candidates = self._filter_by_requirements(reasoning_models, requirements)
            recommendations['primary'] = list(primary_candidates.keys())[:3]
            recommendations['reasoning'] = "Selected reasoning-optimized models for complex problem solving"
            
        elif task_type.lower() in ['code', 'programming', 'development', 'coding']:
            coding_models = {k: v for k, v in self.all_models.items() 
                           if k in self.categories.get('coding', [])}
            primary_candidates = self._filter_by_requirements(coding_models, requirements)
            recommendations['primary'] = list(primary_candidates.keys())[:3]
            recommendations['reasoning'] = "Selected coding-specialized models for programming tasks"
            
        elif task_type.lower() in ['chat', 'conversation', 'dialogue']:
            conversation_models = {k: v for k, v in self.all_models.items() 
                                 if k in self.categories.get('conversation', [])}
            # Add some general-purpose models if conversation-specific are limited
            if len(conversation_models) < 3:
                efficient_models = {k: v for k, v in self.all_models.items() 
                                  if k in self.categories.get('efficient_models', [])}
                conversation_models.update(dict(list(efficient_models.items())[:5]))
            
            primary_candidates = self._filter_by_requirements(conversation_models, requirements)
            recommendations['primary'] = list(primary_candidates.keys())[:3]
            recommendations['reasoning'] = "Selected conversation-optimized and efficient models for dialogue"
            
        elif task_type.lower() in ['large', 'complex', 'comprehensive']:
            large_models = {k: v for k, v in self.all_models.items() 
                          if k in self.categories.get('large_models', [])}
            primary_candidates = self._filter_by_requirements(large_models, requirements)
            recommendations['primary'] = list(primary_candidates.keys())[:3]
            recommendations['reasoning'] = "Selected large-scale models for complex tasks"
            
        elif task_type.lower() in ['fast', 'quick', 'efficient', 'lightweight']:
            efficient_models = {k: v for k, v in self.all_models.items() 
                              if k in self.categories.get('efficient_models', [])}
            primary_candidates = self._filter_by_requirements(efficient_models, requirements)
            recommendations['primary'] = list(primary_candidates.keys())[:3]
            recommendations['reasoning'] = "Selected efficient models for fast response times"
            
        else:
            # General purpose recommendation
            search_results = ModelConfigurations.search_models(task_type)
            if search_results:
                primary_candidates = self._filter_by_requirements(search_results, requirements)
                recommendations['primary'] = list(primary_candidates.keys())[:3]
                recommendations['reasoning'] = f"Found models matching '{task_type}' in name or description"
            else:
                # Fallback to popular free models
                popular_models = ['llama-3.3-8b', 'qwen3-32b', 'deepseek-r1-zero', 'gemma-3-27b']
                available_popular = {k: v for k, v in self.all_models.items() if k in popular_models}
                primary_candidates = self._filter_by_requirements(available_popular, requirements)
                recommendations['primary'] = list(primary_candidates.keys())[:3]
                recommendations['reasoning'] = "Fallback to popular general-purpose models"
        
        # Add alternatives from other categories
        all_candidates = self._filter_by_requirements(self.all_models, requirements)
        alternatives = [k for k in all_candidates.keys() if k not in recommendations['primary']]
        recommendations['alternatives'] = alternatives[:5]
        
        return recommendations
    
    def _filter_by_requirements(self, models: Dict[str, Any], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Filter models based on user requirements"""
        filtered = models.copy()
        
        # Filter by free models only
        if requirements.get('free_only', True):  # Default to free models
            filtered = {k: v for k, v in filtered.items() if k in self.free_models}
        
        # Filter by provider
        if requirements.get('provider'):
            provider_filter = Provider(requirements['provider'])
            filtered = {k: v for k, v in filtered.items() if v.provider == provider_filter}
        
        # Filter by capabilities
        if requirements.get('needs_vision'):
            filtered = {k: v for k, v in filtered.items() if v.supports_images}
        
        if requirements.get('needs_documents'):
            filtered = {k: v for k, v in filtered.items() if v.supports_documents}
        
        # Filter by token limit
        if requirements.get('min_tokens'):
            min_tokens = requirements['min_tokens']
            filtered = {k: v for k, v in filtered.items() if v.max_tokens >= min_tokens}
        
        return filtered
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models side by side"""
        comparison = {}
        
        for model_id in model_ids:
            if model_id in self.all_models:
                config = self.all_models[model_id]
                comparison[model_id] = {
                    'display_name': config.display_name,
                    'provider': config.provider.value,
                    'emoji': config.indicator_emoji,
                    'capabilities': {
                        'vision': config.supports_images,
                        'documents': config.supports_documents,
                        'audio': config.supports_audio,
                        'video': config.supports_video
                    },
                    'specs': {
                        'max_tokens': config.max_tokens,
                        'temperature': config.default_temperature
                    },
                    'cost': 'Free' if model_id in self.free_models else 'Paid',
                    'category': self._get_model_category(model_id),
                    'description': config.description
                }
        
        return comparison
    
    def _get_model_category(self, model_id: str) -> str:
        """Get the primary category for a model"""
        for category, models in self.categories.items():
            if model_id in models:
                return category
        return 'general'
    
    def generate_model_report(self) -> str:
        """Generate a comprehensive model report"""
        stats = ModelConfigurations.get_model_stats()
        
        report = "# Telegram Bot Model Report\n\n"
        from datetime import datetime
        report += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Executive Summary
        report += "## Executive Summary\n\n"
        report += f"- **Total Models Available:** {stats['total_models']}\n"
        report += f"- **Free Models:** {stats['free_models']} ({stats['free_models']/stats['total_models']*100:.1f}%)\n"
        report += f"- **Vision-Capable Models:** {stats['capability_distribution']['images']}\n"
        report += f"- **Document-Capable Models:** {stats['capability_distribution']['documents']}\n\n"
        
        # Provider Distribution
        report += "## Provider Distribution\n\n"
        for provider, count in stats['provider_distribution'].items():
            percentage = count / stats['total_models'] * 100
            report += f"- **{provider.title()}:** {count} models ({percentage:.1f}%)\n"
        report += "\n"
        
        # Category Breakdown
        report += "## Model Categories\n\n"
        for category, count in stats['categories'].items():
            if count > 0:
                category_models = self.categories[category][:3]  # Show first 3 examples
                examples = ', '.join(category_models)
                report += f"- **{category.title().replace('_', ' ')}:** {count} models\n"
                report += f"  - Examples: {examples}\n"
        report += "\n"
        
        # Top Recommendations
        report += "## Top Recommendations by Use Case\n\n"
        use_cases = [
            ('General Chat', 'conversation'),
            ('Image Analysis', 'vision'),
            ('Code Generation', 'coding'),
            ('Complex Reasoning', 'reasoning'),
            ('Fast Responses', 'efficient')
        ]
        
        for use_case, task_type in use_cases:
            recommendations = self.recommend_for_task(task_type, {'free_only': True})
            if recommendations['primary']:
                report += f"### {use_case}\n"
                for i, model_id in enumerate(recommendations['primary'], 1):
                    config = self.all_models[model_id]
                    report += f"{i}. **{config.display_name}** {config.indicator_emoji} (`{model_id}`)\n"
                report += f"   - *{recommendations['reasoning']}*\n\n"
        
        return report

def main():
    """Demo the advanced model selection utility"""
    print("🔧 Advanced Model Selection Utility Demo\n")
    
    selector = ModelSelector()
    
    # Demo 1: Task-based recommendations
    print("1️⃣ Task-based Recommendations:")
    tasks = ['vision', 'coding', 'reasoning', 'conversation']
    
    for task in tasks:
        recommendations = selector.recommend_for_task(task, {'free_only': True})
        print(f"\n   📋 {task.title()} Task:")
        print(f"      Primary: {', '.join(recommendations['primary'][:2])}")
        print(f"      Reason: {recommendations['reasoning']}")
    
    # Demo 2: Model comparison
    print("\n2️⃣ Model Comparison:")
    models_to_compare = ['gemini', 'llama-3.3-8b', 'qwen3-32b', 'deepseek-r1-zero']
    comparison = selector.compare_models(models_to_compare)
    
    for model_id, details in comparison.items():
        print(f"\n   {details['emoji']} {details['display_name']}:")
        print(f"      Provider: {details['provider']}, Cost: {details['cost']}")
        print(f"      Category: {details['category']}, Tokens: {details['specs']['max_tokens']}")
        capabilities = [k for k, v in details['capabilities'].items() if v]
        if capabilities:
            print(f"      Capabilities: {', '.join(capabilities)}")
    
    # Demo 3: Advanced filtering
    print("\n3️⃣ Advanced Filtering Examples:")
    
    # Vision + free models
    vision_requirements = {'free_only': True, 'needs_vision': True}
    vision_rec = selector.recommend_for_task('general', vision_requirements)
    print(f"\n   🖼️ Free Vision Models: {', '.join(vision_rec['primary'][:3])}")
    
    # High token models
    high_token_req = {'free_only': True, 'min_tokens': 4000}
    high_token_rec = selector.recommend_for_task('general', high_token_req)
    print(f"   📄 High Token Models: {', '.join(high_token_rec['primary'][:3])}")
    
    # OpenRouter only
    openrouter_req = {'provider': 'openrouter', 'free_only': True}
    openrouter_rec = selector.recommend_for_task('general', openrouter_req)
    print(f"   🔄 OpenRouter Models: {', '.join(openrouter_rec['primary'][:3])}")
    
    print("\n4️⃣ Report Generation:")
    report = selector.generate_model_report()
    print(f"   📊 Generated comprehensive report ({len(report)} characters)")
    print(f"   Preview: {report[:200]}...")

if __name__ == "__main__":
    main()
