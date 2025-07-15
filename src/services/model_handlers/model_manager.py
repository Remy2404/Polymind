import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an AI model."""

    model_id: str
    display_name: str
    provider: str
    api_required: bool = True
    max_tokens: int = 4096
    supports_images: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_documents: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def capabilities(self) -> Set[str]:
        """Return a set of this model's capabilities."""
        capabilities = {"text"}
        if self.supports_images:
            capabilities.add("images")
        if self.supports_audio:
            capabilities.add("audio")
        if self.supports_video:
            capabilities.add("video")
        if self.supports_documents:
            capabilities.add("documents")
        return capabilities


class ModelManager:
    """
    Manages model registration, selection, and lookup.
    Provides a centralized registry of available AI models and
    handles user model selection preferences.
    """

    def __init__(self):
        """Initialize the ModelManager with an empty model registry."""
        self.models: Dict[str, ModelConfig] = {}
        self.user_model_selections: Dict[int, str] = {}
        self.default_model_id = "gemini"
        logger.info("ModelManager initialized")

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

    def set_user_model(self, user_id: int, model_id: str) -> bool:
        """
        Set a user's selected model.

        Args:
            user_id: The unique identifier for the user.
            model_id: The identifier for the model to select.

        Returns:
            True if successful, False if the model doesn't exist.
        """
        if model_id not in self.models:
            logger.warning(
                f"Attempted to set non-existent model {model_id} for user {user_id}"
            )
            return False

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
        return self.user_model_selections.get(user_id, self.default_model_id)

    def get_all_models(self) -> Dict[str, ModelConfig]:
        """
        Get all registered models.

        Returns:
            Dictionary of model_id -> ModelConfig
        """
        return self.models

    def get_models_with_capability(self, capability: str) -> List[ModelConfig]:
        """
        Get all models that support a specific capability.

        Args:
            capability: Capability to filter by (e.g., "images", "audio")

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
