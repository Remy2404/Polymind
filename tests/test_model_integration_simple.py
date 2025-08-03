import sys
import os
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.services.model_handlers.model_configs import ModelConfigurations, Provider

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_configurations():
    """Test the model configurations directly."""
    logger.info("Starting model configuration test...")

    try:
        # Test getting all models
        all_models = ModelConfigurations.get_all_models()
        logger.info(f"✅ Found {len(all_models)} total models")

        # Test specific models that were mentioned in the logs
        test_models = ["llama-3.3-8b", "deepseek-r1-zero", "gemini", "deepseek"]
        for model_id in test_models:
            if model_id in all_models:
                config = all_models[model_id]
                logger.info(f"✅ Model '{model_id}' found:")
                logger.info(f"  - Display name: {config.display_name}")
                logger.info(f"  - Provider: {config.provider.value}")
                logger.info(
                    f"  - OpenRouter key: {config.openrouter_model_key or 'N/A'}"
                )
                logger.info(f"  - Emoji: {config.indicator_emoji}")
            else:
                logger.warning(f"❌ Model '{model_id}' not found")

        # Test provider-specific lookups
        for provider in Provider:
            models = ModelConfigurations.get_models_by_provider(provider)
            logger.info(f"{provider.value.title()} provider has {len(models)} models")
            if models:
                # Show first few model names as examples
                example_names = list(models.keys())[:3]
                logger.info(f"  Examples: {example_names}")

        # Test free models
        free_models = ModelConfigurations.get_free_models()
        logger.info(f"Free OpenRouter models available: {len(free_models)}")

        # Specific test for llama-3.3-8b to see its OpenRouter mapping
        if "llama-3.3-8b" in all_models:
            llama_config = all_models["llama-3.3-8b"]
            logger.info(f"🦙 Llama 3.3 8B configuration:")
            logger.info(f"  - OpenRouter key: {llama_config.openrouter_model_key}")
            logger.info(f"  - Provider: {llama_config.provider.value}")
            logger.info(
                f"  - System message: {llama_config.system_message[:100] if llama_config.system_message else 'None'}..."
            )

        logger.info("✅ All model configuration tests passed!")
        return True

    except Exception as e:
        logger.error(f"❌ Model configuration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_configurations()
    sys.exit(0 if success else 1)
