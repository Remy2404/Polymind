import logging
from typing import List, Dict, Any, Optional
from services.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class ModelHistoryManager:
    """
    Manages fetching and saving conversation history for AI models,
    centralizing the interaction with the memory system.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initializes the ModelHistoryManager.

        Args:
            memory_manager: An instance of MemoryManager to handle the underlying storage.
        """
        self.memory_manager = memory_manager
        if not self.memory_manager:
            logger.error("MemoryManager instance is required for ModelHistoryManager")
            raise ValueError("MemoryManager instance cannot be None")
        logger.info("ModelHistoryManager initialized.")

    def _get_conversation_id(self, user_id: int) -> str:
        """Generates a consistent conversation ID for a user."""
        return f"user_{user_id}"

    async def get_history(
        self, user_id: int, max_messages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the formatted conversation history for a given user,
        ready for model consumption.

        Args:
            user_id: The unique identifier for the user.
            max_messages: The maximum number of recent messages to retrieve.

        Returns:
            A list of message dictionaries (e.g., [{'role': 'user', 'content': '...'}, ...]).
        """
        conversation_id = self._get_conversation_id(user_id)
        logger.debug(f"Getting history for conversation_id: {conversation_id}")
        
        # Get messages from memory manager
        messages = self.memory_manager.short_term_memory.get(conversation_id, [])
        
        # Only take the most recent messages up to max_messages
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        # Format the messages for the AI model
        formatted_history = []
        for msg in recent_messages:
            # Get sender and content
            sender = msg.get("sender", "")
            content = msg.get("content", "")
            
            # Skip empty messages
            if not content.strip():
                continue
                
            # Map 'sender' to standard role names expected by AI models
            # Be explicit in the mapping to avoid ambiguity
            if sender == "user":
                role = "user"
            else:
                role = "assistant"
                
            formatted_history.append({"role": role, "content": content})
            
        logger.debug(f"Retrieved {len(formatted_history)} history messages for user {user_id}")
        return formatted_history

    async def verify_history_access(self, user_id: int) -> bool:
        """
        Verifies that we can access the conversation history for a user.
        Helpful for debugging memory issues.
        
        Args:
            user_id: The unique identifier for the user.
            
        Returns:
            Boolean indicating if history is accessible and contains expected data.
        """
        try:
            conversation_id = self._get_conversation_id(user_id)
            # Check if the conversation exists in memory
            has_short_term = conversation_id in self.memory_manager.short_term_memory
            # Get message count
            message_count = len(self.memory_manager.short_term_memory.get(conversation_id, []))
            # Log the results
            logger.info(f"History verification for user {user_id}: exists={has_short_term}, message_count={message_count}")
            
            # Get the actual formatted history
            history = await self.get_history(user_id, max_messages=5)
            logger.info(f"Formatted history sample for user {user_id}: {history}")
            
            return has_short_term and message_count > 0
        except Exception as e:
            logger.error(f"History verification failed for user {user_id}: {e}", exc_info=True)
            return False

    async def save_message_pair(
        self, user_id: int, user_message: str, assistant_message: str
    ) -> None:
        """
        Saves a pair of user and assistant messages to the conversation history.

        Args:
            user_id: The unique identifier for the user.
            user_message: The content of the user's message.
            assistant_message: The content of the assistant's response.
        """
        conversation_id = self._get_conversation_id(user_id)
        logger.debug(f"Saving message pair for conversation_id: {conversation_id}")
        try:
            # Use MemoryManager to add messages
            await self.memory_manager.add_user_message(
                conversation_id, user_message, str(user_id)
            )
            await self.memory_manager.add_assistant_message(
                conversation_id, assistant_message
            )
            # Optionally manage context window after adding messages
            await self.memory_manager._maybe_manage_context_window(conversation_id)
            logger.info(f"Saved message pair for user {user_id}")
            
            # Verify history was saved correctly
            await self.verify_history_access(user_id)
        except Exception as e:
            logger.error(
                f"Failed to save message pair for user {user_id}: {e}", exc_info=True
            )

    async def save_image_interaction(
        self, user_id: int, caption: str, assistant_response: str
    ) -> None:
        """
        Saves an image interaction (caption + analysis) to the history.

        Args:
            user_id: The unique identifier for the user.
            caption: The caption provided by the user for the image.
            assistant_response: The assistant's analysis or response related to the image.
        """
        conversation_id = self._get_conversation_id(user_id)
        logger.debug(f"Saving image interaction for conversation_id: {conversation_id}")
        user_content = f"[Image with caption: {caption}]"
        try:
            # Use MemoryManager to add messages
            await self.memory_manager.add_user_message(
                conversation_id, user_content, str(user_id), message_type="image"
            )
            await self.memory_manager.add_assistant_message(
                conversation_id, assistant_response
            )
            # Optionally manage context window
            await self.memory_manager._maybe_manage_context_window(conversation_id)
            logger.info(f"Saved image interaction for user {user_id}")
        except Exception as e:
            logger.error(
                f"Failed to save image interaction for user {user_id}: {e}",
                exc_info=True,
            )

    async def clear_history(self, user_id: int) -> None:
        """
        Clears the conversation history for a specific user.

        Args:
            user_id: The unique identifier for the user.
        """
        conversation_id = self._get_conversation_id(user_id)
        logger.info(f"Clearing history for conversation_id: {conversation_id}")
        try:
            # Use MemoryManager to clear the conversation
            cleared = await self.memory_manager.clear_conversation(conversation_id)
            if cleared:
                logger.info(f"Successfully cleared history for user {user_id}")
            else:
                logger.warning(f"Could not clear history for user {user_id}")
        except Exception as e:
            logger.error(
                f"Failed to clear history for user {user_id}: {e}", exc_info=True
            )