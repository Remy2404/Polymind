import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    TEXT = "text"
    IMAGES = "images"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENTS = "documents"
    CODE = "code"
    FUNCTION_CALLING = "function_calling"


@dataclass
class ModelConfig:
    """Configuration for an AI model."""

    model_id: str
    display_name: str
    provider: str
    description: str = ""
    api_required: bool = True
    max_tokens: int = 4096
    capabilities: Set[ModelCapability] = field(
        default_factory=lambda: {ModelCapability.TEXT}
    )
    default_params: Dict[str, Any] = field(default_factory=dict)
    indicator_emoji: str = "🤖"
    timeout_seconds: int = 60

    def get_model_indicator(self) -> str:
        """Get a formatted model indicator string."""
        return f"{self.indicator_emoji} {self.display_name}"


class ModelRegistry:
    """
    Central registry for managing available AI models.
    Provides methods for model registration, lookup, and capability-based filtering.
    """

    def __init__(self):
        """Initialize the ModelRegistry with an empty model collection."""
        self.models: Dict[str, ModelConfig] = {}
        self.default_model_id = "gemini"
        logger.info("ModelRegistry initialized")

    def register_model(self, model_config: ModelConfig) -> None:
        """
        Register a new AI model with the system.

        Args:
            model_config: Configuration for the AI model.
        """
        if model_config.model_id in self.models:
            logger.warning(
                f"Model {model_config.model_id} already registered, updating config"
            )

        self.models[model_config.model_id] = model_config
        logger.info(
            f"Registered model: {model_config.model_id} ({model_config.display_name})"
        )

    def register_models(self, model_configs: List[ModelConfig]) -> None:
        """
        Register multiple AI models at once.

        Args:
            model_configs: List of model configurations.
        """
        for config in model_configs:
            self.register_model(config)

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get the configuration for a specific model.

        Args:
            model_id: The identifier for the model.

        Returns:
            The model configuration if found, None otherwise.
        """
        return self.models.get(model_id)

    def get_all_models(self) -> Dict[str, ModelConfig]:
        """
        Get all registered models.

        Returns:
            Dictionary of model_id -> ModelConfig
        """
        return self.models

    def get_models_with_capability(
        self, capability: ModelCapability
    ) -> List[ModelConfig]:
        """
        Get all models that support a specific capability.

        Args:
            capability: Capability to filter by (e.g., TEXT, IMAGES)

        Returns:
            List of matching ModelConfig objects.
        """
        return [
            model for model in self.models.values() if capability in model.capabilities
        ]

    def set_default_model(self, model_id: str) -> bool:
        """
        Set the default model for new users.

        Args:
            model_id: The identifier for the model to use as default.

        Returns:
            True if successful, False if the model doesn't exist.
        """
        if model_id not in self.models:
            logger.warning(f"Attempted to set non-existent model {model_id} as default")
            return False

        self.default_model_id = model_id
        logger.info(f"Set default model to: {model_id}")
        return True


class UserModelManager:
    """
    Manages user model preferences and provides a clean interface
    for model selection state.
    """

    def __init__(self, model_registry: ModelRegistry):
        """
        Initialize UserModelManager with a reference to the central model registry.

        Args:
            model_registry: The ModelRegistry containing available models.
        """
        self.model_registry = model_registry
        self.user_model_selections: Dict[int, str] = {}
        self.user_model_history: Dict[int, List[str]] = {}
        logger.info("UserModelManager initialized")

    def set_user_model(self, user_id: int, model_id: str) -> bool:
        """
        Set a user's selected model.

        Args:
            user_id: The unique identifier for the user.
            model_id: The identifier for the model to select.

        Returns:
            True if successful, False if the model doesn't exist.
        """
        # Validate the model exists
        if model_id not in self.model_registry.models:
            logger.warning(
                f"Attempted to set non-existent model {model_id} for user {user_id}"
            )
            return False

        # Add current model to history if it's different
        current_model = self.get_user_model(user_id)
        if current_model != model_id:
            if user_id not in self.user_model_history:
                self.user_model_history[user_id] = []

            # Only add if different from last used
            if (
                not self.user_model_history[user_id]
                or self.user_model_history[user_id][-1] != current_model
            ):
                self.user_model_history[user_id].append(current_model)
                # Keep only last 5 models in history
                if len(self.user_model_history[user_id]) > 5:
                    self.user_model_history[user_id] = self.user_model_history[user_id][
                        -5:
                    ]

        # Set the new model
        self.user_model_selections[user_id] = model_id
        logger.info(f"User {user_id} selected model: {model_id}")
        return True

    def get_user_model(self, user_id: int) -> str:
        """
        Get a user's currently selected model ID.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            The model ID string.
        """
        return self.user_model_selections.get(
            user_id, self.model_registry.default_model_id
        )

    def get_user_model_config(self, user_id: int) -> ModelConfig:
        """
        Get the ModelConfig for a user's currently selected model.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            The ModelConfig object for the user's current model.
        """
        model_id = self.get_user_model(user_id)
        model_config = self.model_registry.get_model_config(model_id)

        # Fallback to default model if user's selected model isn't found
        if not model_config:
            logger.warning(
                f"Model {model_id} not found for user {user_id}, using default"
            )
            model_id = self.model_registry.default_model_id
            model_config = self.model_registry.get_model_config(model_id)

            # If still not found, raise an error
            if not model_config:
                logger.error(f"Default model {model_id} not found")
                raise ValueError(f"Default model {model_id} not found")

        return model_config

    def get_previous_model(self, user_id: int) -> Optional[str]:
        """
        Get the user's previously selected model ID.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            The previous model ID or None if no history exists.
        """
        if (
            user_id not in self.user_model_history
            or not self.user_model_history[user_id]
        ):
            return None

        return self.user_model_history[user_id][-1]

    def clear_history(self, user_id: int) -> None:
        """
        Clear the model selection history for a user.

        Args:
            user_id: The unique identifier for the user.
        """
        if user_id in self.user_model_history:
            self.user_model_history[user_id] = []
            logger.info(f"Cleared model history for user {user_id}")


# Create a global instance for convenience
model_registry = ModelRegistry()
