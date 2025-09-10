"""
Tool Calling Support Detection for AI Models

This module provides functionality to identify which AI models support tool calling/function calling
and provides utilities for filtering and displaying tool-call capable models.
"""

import sys
import logging
from typing import Dict, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.services.model_handlers.simple_api_manager import SuperSimpleAPIManager
from src.services.model_handlers.model_configs import ModelConfigurations

logger = logging.getLogger(__name__)


class ToolCallSupportDetector:
    """Detects which AI models support tool calling/function calling."""

    def __init__(self, api_manager: Optional[SuperSimpleAPIManager] = None):
        """
        Initialize the tool call support detector.

        Args:
            api_manager: Optional API manager instance
        """
        self.api_manager = api_manager or SuperSimpleAPIManager()
        self._tool_call_models = None

    def reset_cache(self):
        """Reset the cached tool call models to force recalculation."""
        self._tool_call_models = None

    def get_tool_call_supported_models(self) -> Dict[str, Dict]:
        """
        Get all models that support tool calling.

        Returns:
            Dictionary mapping model IDs to their configurations
        """
        # Force cache reset to ensure we get fresh results with updated logic
        if self._tool_call_models is not None:
            return self._tool_call_models

        all_models = self.api_manager.get_all_models()
        tool_call_models = {}

        for model_id, config in all_models.items():
            # Skip models with None config
            if config is None:
                continue

            if self._supports_tool_calling(model_id, config):
                tool_call_models[model_id] = config

        self._tool_call_models = tool_call_models
        return tool_call_models

    def _supports_tool_calling(self, model_id: str, config) -> bool:
        """
        Determine if a model supports tool calling based on supported_parameters.
        
        Following OpenRouter documentation: only check if 'tools' is explicitly 
        listed in the supported_parameters array for each model.

        Args:
            model_id: The model identifier
            config: Model configuration object

        Returns:
            True if the model supports tool calling
        """
        if config is None:
            return False

        # Method 1: Check supported_parameters from ModelConfig (primary method - ChatGPT's suggestion)
        try:
            model_configs = ModelConfigurations.get_all_models()
            if model_id in model_configs:
                model_config = model_configs[model_id]
                if hasattr(model_config, 'supported_parameters') and model_config.supported_parameters:
                    # Strict check: only 'tools' in supported_parameters indicates tool calling support
                    return 'tools' in model_config.supported_parameters
        except Exception as e:
            logger.debug(f"Could not check supported_parameters for {model_id}: {e}")

        # Method 2: Check if supported_parameters is directly available in config (fallback)
        try:
            supported_params = getattr(config, 'supported_parameters', [])
            if supported_params:
                return 'tools' in supported_params
        except Exception as e:
            logger.debug(f"Could not access supported_parameters directly for {model_id}: {e}")

        # Method 3: DeepSeek models only (known to work reliably, very last resort)
        if hasattr(config, "provider"):
            if (
                hasattr(config.provider, "value")
                and config.provider.value == "deepseek"
            ):
                return True

        return False

    def get_tool_call_models_by_category(self) -> Dict[str, Dict]:
        """
        Get tool-call supported models organized by category.

        Returns:
            Dictionary with categories and their tool-call models
        """
        tool_call_models = self.get_tool_call_supported_models()
        categories = self.api_manager.get_models_by_category()

        # Filter categories to only include tool-call models
        filtered_categories = {}

        for category_id, category_info in categories.items():
            category_models = {}
            for model_id, config in category_info["models"].items():
                if model_id in tool_call_models:
                    category_models[model_id] = config

            if category_models:  # Only include categories with tool-call models
                filtered_categories[category_id] = {
                    "name": category_info["name"],
                    "emoji": category_info["emoji"],
                    "models": category_models,
                }

        return filtered_categories

    def get_tool_call_statistics(self) -> Dict[str, int]:
        """
        Get statistics about tool-call supported models.

        Returns:
            Dictionary with various statistics
        """
        tool_call_models = self.get_tool_call_supported_models()
        all_models = self.api_manager.get_all_models()

        return {
            "total_models": len(all_models),
            "tool_call_models": len(tool_call_models),
            "percentage": (
                round((len(tool_call_models) / len(all_models)) * 100, 1)
                if all_models
                else 0
            ),
        }

    def print_tool_call_models_report(self) -> str:
        """
        Generate a formatted report of tool-call supported models.

        Returns:
            Formatted string report
        """
        categories = self.get_tool_call_models_by_category()
        stats = self.get_tool_call_statistics()

        report_lines = [
            "🛠️ **Tool-Calling Models Report**",
            "=" * 50,
            "📊 **Statistics:**",
            f"   • Total Models: {stats['total_models']}",
            f"   • Tool-Call Models: {stats['tool_call_models']}",
            f"   • Support Rate: {stats['percentage']}%",
            "",
            "📂 **Models by Category:**",
        ]

        for category_id, category_info in categories.items():
            report_lines.append(
                f"\n{category_info['emoji']} **{category_info['name']}:**"
            )

            for model_id, config in category_info["models"].items():
                # Safely get emoji, fallback to default
                emoji = getattr(
                    config, "emoji", getattr(config, "indicator_emoji", "🤖")
                )
                display_name = getattr(config, "display_name", model_id)

                model_line = f"   • {emoji} {display_name}"
                if hasattr(config, "openrouter_key") and config.openrouter_key:
                    model_line += f" (`{config.openrouter_key}`)"
                report_lines.append(model_line)

        report_lines.extend(
            [
                "",
                "=" * 50,
                "💡 **Note:** Tool-calling support is detected based on model capabilities.",
                "   Some models may have limited or experimental tool-calling features.",
            ]
        )

        return "\n".join(report_lines)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Tool Calling Support Detector")
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed report"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    parser.add_argument(
        "--category", type=str, help="Show models for specific category"
    )

    args = parser.parse_args()

    detector = ToolCallSupportDetector()

    if args.stats:
        stats = detector.get_tool_call_statistics()
        print(
            f"Tool-call models: {stats['tool_call_models']}/{stats['total_models']} ({stats['percentage']}%)"
        )
    elif args.category:
        categories = detector.get_tool_call_models_by_category()
        if args.category in categories:
            category_info = categories[args.category]
            print(f"{category_info['emoji']} {category_info['name']}:")
            for model_id, config in category_info["models"].items():
                # Safely get emoji and display name
                emoji = getattr(
                    config, "emoji", getattr(config, "indicator_emoji", "🤖")
                )
                display_name = getattr(config, "display_name", model_id)
                print(f"  • {emoji} {display_name}")
        else:
            print(f"Category '{args.category}' not found")
    else:
        # Default: show report
        print(detector.print_tool_call_models_report())


if __name__ == "__main__":
    main()
