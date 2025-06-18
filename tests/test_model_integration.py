import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.handlers.message_handlers import MessageHandlers
from src.services.model_handlers.model_configs import ModelConfigurations, Provider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_integration():
    """Test the model integration functionality."""
    logger.info("Starting model integration test...")
    
    # Mock the required dependencies
    class MockAPI:
        async def generate_response(self, prompt):
            return f"Mock response to: {prompt}"
    
    class MockUserDataManager:
        async def initialize_user(self, user_id):
            pass
        async def get_user_settings(self, user_id):
            return {"active_model": "gemini"}
        async def get_user_preference(self, user_id, key, default):
            return default
    
    class MockTelegramLogger:
        def log_message(self, message, user_id):
            pass
    
    # Create message handlers instance
    try:
        message_handlers = MessageHandlers(
            gemini_api=MockAPI(),
            user_data_manager=MockUserDataManager(),
            telegram_logger=MockTelegramLogger(),
            text_handler=None,
            deepseek_api=MockAPI(),
            openrouter_api=MockAPI()
        )
        
        logger.info("✅ MessageHandlers instance created successfully")
        
        # Test model statistics
        stats = message_handlers.get_model_stats()
        logger.info(f"Model statistics: {stats}")
        
        # Test specific model lookups
        test_models = ["llama-3.3-8b", "deepseek-r1-zero", "gemini", "deepseek"]
        for model_id in test_models:
            config = message_handlers.get_model_config(model_id)
            if config:
                logger.info(f"✅ Model '{model_id}' found: {config.display_name} ({config.provider.value})")
            else:
                logger.warning(f"❌ Model '{model_id}' not found")
        
        # Test model indicator generation
        for model_id in test_models:
            indicator, config = message_handlers.get_model_indicator_and_config(model_id)
            logger.info(f"Model '{model_id}' indicator: {indicator}")
        
        # Test provider-specific lookups
        for provider in Provider:
            models = message_handlers.get_models_by_provider(provider)
            logger.info(f"{provider.value.title()} provider has {len(models)} models")
        
        logger.info("✅ All model integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model integration test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_integration()
    sys.exit(0 if success else 1)
