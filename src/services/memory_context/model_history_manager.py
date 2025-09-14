import logging
from typing import List, Dict, Any, Optional
from src.services.memory_context.memory_manager import MemoryManager
from services.model_handlers.model_registry import UserModelManager, ModelRegistry
logger = logging.getLogger(__name__)
class ModelHistoryManager:
    """
    Manages conversation history for multiple AI models,
    centralizing the interaction with the memory system.
    Each model has its own separate conversation history per user.
    """
    def __init__(
        self,
        memory_manager: MemoryManager,
        user_model_manager: Optional[UserModelManager] = None,
        model_registry: Optional[ModelRegistry] = None,
    ):
        """
        Initialize the ModelHistoryManager.
        Args:
            memory_manager: MemoryManager instance to handle the underlying storage.
            user_model_manager: Optional UserModelManager for tracking model selection.
            model_registry: Optional ModelRegistry for access to available models.
        """
        self.memory_manager = memory_manager
        if not self.memory_manager:
            logger.error("MemoryManager instance is required for ModelHistoryManager")
            raise ValueError("MemoryManager instance cannot be None")
        self.user_model_manager = user_model_manager
        self.model_registry = model_registry
        self._user_model_selection = {}
        self._default_model = "gemini"
        logger.info(
            f"ModelHistoryManager initialized with {'external' if user_model_manager else 'internal'} model tracking"
        )
    def _get_conversation_id(self, user_id: int, model_id: Optional[str] = None) -> str:
        """
        Generate a unique conversation ID for a user and model.
        Args:
            user_id: User's unique identifier
            model_id: Optional model ID. If None, uses user's current model.
        Returns:
            A unique string identifier combining user and model.
        """
        if model_id is None:
            model_id = self.get_selected_model(user_id)
        return f"user_{user_id}_model_{model_id}"
    def get_selected_model(self, user_id: int) -> str:
        """
        Get the currently selected model for a user.
        Args:
            user_id: User's unique identifier
        Returns:
            Model ID string.
        """
        if self.user_model_manager:
            return self.user_model_manager.get_user_model(user_id)
        return self._user_model_selection.get(user_id, self._default_model)
    def set_selected_model(self, user_id: int, model_id: str) -> bool:
        """
        Set the user's selected model.
        Args:
            user_id: User's unique identifier
            model_id: Model ID to select
        Returns:
            True if successful, False if model was invalid.
        """
        if self.user_model_manager:
            return self.user_model_manager.set_user_model(user_id, model_id)
        if self.model_registry and model_id not in self.model_registry.get_all_models():
            logger.warning(
                f"Attempted to set invalid model {model_id} for user {user_id}"
            )
            return False
        self._user_model_selection[user_id] = model_id
        logger.info(f"User {user_id} switched to model: {model_id}")
        return True
    async def get_history(
        self, user_id: int, max_messages: int = 10, model_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get formatted conversation history for a user and model.
        Args:
            user_id: User's unique identifier
            max_messages: Maximum number of recent messages to retrieve
            model_id: Optional model ID. If None, uses user's current model.
        Returns:
            List of messages in format suitable for AI models.
        """
        if model_id is None:
            model_id = self.get_selected_model(user_id)
        conversation_id = self._get_conversation_id(user_id, model_id)
        logger.debug(
            f"Getting history for conversation_id: {conversation_id} (model: {model_id})"
        )
        try:
            messages = await self.memory_manager.get_short_term_memory(
                conversation_id, limit=max_messages
            )
            formatted_history = []
            for msg in messages:
                sender = msg.get("sender", "")
                content = msg.get("content", "")
                if not content or not content.strip():
                    continue
                role = (
                    "user"
                    if sender == str(user_id) or sender == "user"
                    else "assistant"
                )
                formatted_history.append({"role": role, "content": content})
            logger.info(
                f"Retrieved {len(formatted_history)} history messages for user {user_id} with model {model_id}"
            )
            return formatted_history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    async def save_message_pair(
        self,
        user_id: int,
        user_message: str,
        assistant_message: str,
        model_id: Optional[str] = None,
    ) -> None:
        """
        Save a user-assistant message exchange.
        Args:
            user_id: User's unique identifier
            user_message: Content of user's message
            assistant_message: Content of assistant's response
            model_id: Optional model ID. If None, uses user's current model.
        """
        if model_id is None:
            model_id = self.get_selected_model(user_id)
        conversation_id = self._get_conversation_id(user_id, model_id)
        logger.debug(
            f"Saving message pair for conversation_id: {conversation_id} (model: {model_id})"
        )
        try:
            await self.memory_manager.add_user_message(
                conversation_id, user_message, str(user_id)
            )
            await self.memory_manager.add_assistant_message(
                conversation_id, assistant_message
            )
            if hasattr(self.memory_manager, "_maybe_manage_context_window"):
                await self.memory_manager._maybe_manage_context_window(conversation_id)
            logger.info(f"Saved message pair for user {user_id} with model {model_id}")
            await self.verify_history_access(user_id, model_id)
        except Exception as e:
            logger.error(
                f"Failed to save message pair for user {user_id} with model {model_id}: {e}",
                exc_info=True,
            )
    async def save_image_interaction(
        self,
        user_id: int,
        caption: str,
        assistant_response: str,
        model_id: Optional[str] = None,
    ) -> None:
        """
        Save an image interaction in the conversation history.
        Args:
            user_id: User's unique identifier
            caption: Caption or description of the image
            assistant_response: Assistant's analysis or response to the image
            model_id: Optional model ID. If None, uses user's current model.
        """
        if model_id is None:
            model_id = self.get_selected_model(user_id)
        conversation_id = self._get_conversation_id(user_id, model_id)
        logger.debug(
            f"Saving image interaction for conversation_id: {conversation_id} (model: {model_id})"
        )
        user_content = f"[Image with caption: {caption}]"
        try:
            await self.memory_manager.add_user_message(
                conversation_id, user_content, str(user_id), message_type="image"
            )
            await self.memory_manager.add_assistant_message(
                conversation_id, assistant_response
            )
            if hasattr(self.memory_manager, "_maybe_manage_context_window"):
                await self.memory_manager._maybe_manage_context_window(conversation_id)
            logger.info(
                f"Saved image interaction for user {user_id} with model {model_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to save image interaction for user {user_id} with model {model_id}: {e}",
                exc_info=True,
            )
    async def verify_history_access(
        self, user_id: int, model_id: Optional[str] = None
    ) -> bool:
        """
        Verify that history is accessible for a user and model.
        Args:
            user_id: User's unique identifier
            model_id: Optional model ID. If None, uses user's current model.
        Returns:
            True if history is accessible and contains data.
        """
        try:
            if model_id is None:
                model_id = self.get_selected_model(user_id)
            conversation_id = self._get_conversation_id(user_id, model_id)
            messages = await self.memory_manager.get_short_term_memory(conversation_id)
            message_count = len(messages)
            logger.info(
                f"History verification for user {user_id} with model {model_id}: exists={message_count > 0}, message_count={message_count}"
            )
            history = await self.get_history(user_id, max_messages=5, model_id=model_id)
            logger.debug(
                f"Formatted history sample for user {user_id} with model {model_id}: {history}"
            )
            return message_count > 0
        except Exception as e:
            logger.error(
                f"History verification failed for user {user_id} with model {model_id}: {e}",
                exc_info=True,
            )
            return False
    async def clear_history(
        self,
        user_id: int,
        model_id: Optional[str] = None,
        clear_all_models: bool = False,
    ) -> None:
        """
        Clear conversation history for a user.
        Args:
            user_id: User's unique identifier
            model_id: Optional model ID. If None, uses user's current model.
            clear_all_models: If True, clears history for all models for this user.
        """
        if clear_all_models:
            model_ids = []
            if self.model_registry:
                model_ids = list(self.model_registry.get_all_models().keys())
            else:
                if self.user_model_manager:
                    current_model = self.user_model_manager.get_user_model(user_id)
                else:
                    current_model = self._user_model_selection.get(
                        user_id, self._default_model
                    )
                if (
                    hasattr(self, "_user_model_selection")
                    and user_id in self._user_model_selection
                ):
                    model_ids.append(self._user_model_selection[user_id])
                if self.user_model_manager and hasattr(
                    self.user_model_manager, "user_model_history"
                ):
                    if user_id in self.user_model_manager.user_model_history:
                        model_ids.extend(
                            self.user_model_manager.user_model_history[user_id]
                        )
                model_ids.append(current_model)
                model_ids.append(self._default_model)
                model_ids = list(set(model_ids))
            cleared_count = 0
            for mid in model_ids:
                conversation_id = self._get_conversation_id(user_id, mid)
                try:
                    cleared = await self.memory_manager.clear_conversation(
                        conversation_id
                    )
                    if cleared:
                        cleared_count += 1
                except Exception as e:
                    logger.error(
                        f"Error clearing history for user {user_id} with model {mid}: {e}"
                    )
            logger.info(
                f"Cleared {cleared_count}/{len(model_ids)} model histories for user {user_id}"
            )
        else:
            if model_id is None:
                model_id = self.get_selected_model(user_id)
            conversation_id = self._get_conversation_id(user_id, model_id)
            logger.info(
                f"Clearing history for conversation_id: {conversation_id} (model: {model_id})"
            )
            try:
                cleared = await self.memory_manager.clear_conversation(conversation_id)
                if cleared:
                    logger.info(
                        f"Successfully cleared history for user {user_id} with model {model_id}"
                    )
                else:
                    logger.warning(
                        f"Could not clear history for user {user_id} with model {model_id}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to clear history for user {user_id} with model {model_id}: {e}",
                    exc_info=True,
                )
