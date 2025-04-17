import logging
from typing import List, Dict, Any, Optional
from services.memory_manager import MemoryManager
from services.model_handlers.model_history_manager import ModelHistoryManager

class ConversationManager:
    """Manages conversation state and history."""
    
    def __init__(self, memory_manager: MemoryManager, model_history_manager: ModelHistoryManager):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = memory_manager
        self.model_history_manager = model_history_manager
        
    async def get_conversation_history(self, user_id: int, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get the conversation history for a user."""
        return await self.model_history_manager.get_history(user_id, max_messages=max_messages)
        
    async def save_message_pair(self, user_id: int, user_message: str, assistant_message: str, model_id: Optional[str] = None) -> None:
        """Save a user-assistant message pair to the conversation history."""
        await self.model_history_manager.save_message_pair(
            user_id, user_message, assistant_message, model_id
        )
        
    async def reset_conversation(self, user_id: int) -> None:
        """Clear the conversation history for a user."""
        await self.model_history_manager.clear_history(user_id)
        
    async def save_media_interaction(self, user_id: int, media_type: str, prompt: str, response: str) -> None:
        """Save an interaction with media (image, document, voice) to the conversation history."""
        conversation_id = f"user_{user_id}"
        
        # Add user message with media indicator
        await self.memory_manager.add_user_message(
            conversation_id,
            f"[{media_type.capitalize()} message: {prompt}]",
            str(user_id),
            is_media=True,
            media_type=media_type,
        )
        
        # Add assistant response
        await self.memory_manager.add_assistant_message(
            conversation_id,
            response
        )
        
    async def add_quoted_message_context(self, user_id: int, quoted_text: str, user_reply: str, assistant_response: str) -> None:
        """Add a quoted message interaction to the conversation history."""
        conversation_id = f"user_{user_id}"
        
        # Format the user message to include the quoted context
        formatted_user_message = f"[Replying to: \"{quoted_text}\"] {user_reply}"
        
        # Add user message with quote context
        await self.memory_manager.add_user_message(
            conversation_id,
            formatted_user_message,
            str(user_id),
            is_reply=True,
            quoted_text=quoted_text
        )
        
        # Add assistant response
        await self.memory_manager.add_assistant_message(
            conversation_id,
            assistant_response
        )
        
    async def get_short_term_memory(self, user_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent messages for quick context."""
        conversation_id = f"user_{user_id}"
        return await self.memory_manager.get_short_term_memory(conversation_id, limit)
        
    async def get_long_term_memory(self, user_id: int, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get relevant messages from long-term memory based on the query."""
        conversation_id = f"user_{user_id}"
        return await self.memory_manager.get_relevant_memory(conversation_id, query, limit)