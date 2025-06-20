"""
Telegram Bot Integration Example
Demonstrates how to use enhanced ModelConfigurations in a Telegram bot context
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.model_handlers.model_configs import ModelConfigurations
from functools import lru_cache
import json

class TelegramBotModelManager:
    """
    Example class showing how to integrate enhanced model configurations
    into a Telegram bot for model management and selection
    """
    
    def __init__(self):
        self.model_configs = ModelConfigurations
    
    @lru_cache(maxsize=1)
    def get_cached_stats(self):
        """Cache model statistics for performance"""
        return self.model_configs.get_model_stats()
    
    def format_models_overview(self) -> str:
        """Format model overview for Telegram message"""
        stats = self.get_cached_stats()
        
        message = f"🤖 **Model Overview**\n\n"
        message += f"📊 **Statistics:**\n"
        message += f"• Total models: {stats['total_models']}\n"
        message += f"• Free models: {stats['free_models']}\n"
        message += f"• Providers: {len(stats['provider_distribution'])}\n\n"
        
        message += "🏢 **Top Providers:**\n"
        for provider, count in list(stats['provider_distribution'].items())[:5]:
            message += f"• {provider}: {count} models\n"
        
        return message
      def format_vision_models(self, limit: int = 10) -> str:
        """Format vision-capable models for Telegram"""
        vision_models = self.model_configs.get_models_by_capability('images')
        
        message = f"👁️ **Vision Models ({len(vision_models)} available)**\n\n"
        
        for model_id, config in list(vision_models.items())[:limit]:
            is_free = (config.provider.value == "openrouter" and 
                      config.openrouter_model_key and ":free" in config.openrouter_model_key)
            cost_info = "Free" if is_free else "Paid"
            
            message += f"{config.indicator_emoji} **{config.display_name}**\n"
            message += f"   ID: `{model_id}`\n"
            message += f"   Provider: {config.provider.value}\n"
            message += f"   Cost: {cost_info}\n\n"
        
        if len(vision_models) > limit:
            message += f"... and {len(vision_models) - limit} more models\n"
        
        return message
    
    def search_and_format(self, query: str, limit: int = 8) -> str:
        """Search models and format for Telegram"""
        results = self.model_configs.search_models(query)
        
        if not results:
            return f"🔍 No models found for: '{query}'"
        
        message = f"🔍 **Search Results for '{query}'** ({len(results)} found)\n\n"
        
        for model_id, config in list(results.items())[:limit]:
            message += f"{config.indicator_emoji} **{config.display_name}**\n"
            message += f"   ID: `{model_id}`\n"
            message += f"   Provider: {config.provider.value}\n\n"
        
        if len(results) > limit:
            message += f"... and {len(results) - limit} more results\n"
        
        return message
    
    def format_free_models(self, limit: int = 15) -> str:
        """Format free models for Telegram"""
        all_models = self.model_configs.get_all_models()
        
        free_models = {
            model_id: config for model_id, config in all_models.items()
            if config.cost == 0
        }
        
        message = f"💰 **Free Models ({len(free_models)} available)**\n\n"
        
        for model_id, config in list(free_models.items())[:limit]:
            message += f"{config.indicator_emoji} **{config.display_name}**\n"
            message += f"   ID: `{model_id}`\n"
            message += f"   Provider: {config.provider.value}\n\n"
        
        if len(free_models) > limit:
            message += f"... and {len(free_models) - limit} more free models\n"
        
        return message
    
    def format_model_categories(self) -> str:
        """Format model categories for Telegram"""
        categories = self.model_configs.get_model_categories()
        
        message = "📂 **Model Categories:**\n\n"
        
        for category, models in categories.items():
            if models:
                message += f"**{category.title()}** ({len(models)} models)\n"
                examples = models[:3]
                message += f"   Examples: {', '.join(f'`{m}`' for m in examples)}\n\n"
        
        return message
    
    def get_model_info(self, model_id: str) -> str:
        """Get detailed model information"""
        validation = self.model_configs.validate_model_config(model_id)
        
        if not validation['is_valid']:
            return f"❌ Model '{model_id}' not found or invalid."
        
        all_models = self.model_configs.get_all_models()
        config = all_models[model_id]
        
        message = f"🤖 **Model Information**\n\n"
        message += f"**Name:** {config.display_name}\n"
        message += f"**ID:** `{model_id}`\n"
        message += f"**Provider:** {config.provider.value}\n"
        message += f"**Cost:** {'Free' if config.cost == 0 else f'${config.cost:.4f}'}\n"
        message += f"**Capabilities:** {', '.join(config.capabilities)}\n"
        
        # Add context length if available
        if hasattr(config, 'context_length'):
            message += f"**Context Length:** {config.context_length}\n"
        
        return message
    
    def get_recommended_models(self, use_case: str) -> str:
        """Get recommended models for specific use cases"""
        recommendations = {}
        
        if use_case.lower() in ['vision', 'images', 'image']:
            models = self.model_configs.get_models_by_capability('images')
            # Get top 3 free vision models
            free_vision = {k: v for k, v in models.items() if v.cost == 0}
            recommendations = dict(list(free_vision.items())[:3])
            title = "👁️ **Recommended Vision Models (Free)**"
            
        elif use_case.lower() in ['coding', 'code', 'programming']:
            models = self.model_configs.get_models_by_capability('coding')
            # Get top 3 coding models
            recommendations = dict(list(models.items())[:3])
            title = "💻 **Recommended Coding Models**"
            
        elif use_case.lower() in ['reasoning', 'logic', 'analysis']:
            models = self.model_configs.get_models_by_capability('reasoning')
            recommendations = dict(list(models.items())[:3])
            title = "🧠 **Recommended Reasoning Models**"
            
        elif use_case.lower() in ['free', 'budget']:
            all_models = self.model_configs.get_all_models()
            free_models = {k: v for k, v in all_models.items() if v.cost == 0}
            # Get diverse free models
            recommendations = dict(list(free_models.items())[:5])
            title = "💰 **Recommended Free Models**"
            
        else:
            return f"❓ Unknown use case: '{use_case}'. Try: vision, coding, reasoning, or free"
        
        if not recommendations:
            return f"😔 No models found for use case: '{use_case}'"
        
        message = f"{title}\n\n"
        
        for model_id, config in recommendations.items():
            cost_info = "Free" if config.cost == 0 else f"${config.cost:.4f}"
            message += f"{config.indicator_emoji} **{config.display_name}**\n"
            message += f"   ID: `{model_id}`\n"
            message += f"   Cost: {cost_info}\n\n"
        
        return message

def demo_telegram_integration():
    """Demonstrate Telegram bot integration"""
    print("🤖 Telegram Bot Model Manager Demo\n")
    
    bot_manager = TelegramBotModelManager()
    
    # Simulate different Telegram bot commands
    test_cases = [
        ("Models Overview", lambda: bot_manager.format_models_overview()),
        ("Vision Models", lambda: bot_manager.format_vision_models(5)),
        ("Search 'reasoning'", lambda: bot_manager.search_and_format('reasoning', 3)),
        ("Free Models", lambda: bot_manager.format_free_models(5)),
        ("Categories", lambda: bot_manager.format_model_categories()),
        ("Model Info 'gpt-4o'", lambda: bot_manager.get_model_info('gpt-4o')),
        ("Recommendations for 'vision'", lambda: bot_manager.get_recommended_models('vision')),
        ("Recommendations for 'free'", lambda: bot_manager.get_recommended_models('free')),
    ]
    
    for title, func in test_cases:
        print(f"{'='*50}")
        print(f"🔹 {title}")
        print(f"{'='*50}")
        try:
            result = func()
            print(result)
        except Exception as e:
            print(f"❌ Error: {e}")
        print()

if __name__ == "__main__":
    demo_telegram_integration()
