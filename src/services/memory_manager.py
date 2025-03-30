import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import asyncio
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a message from dictionary data."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Conversation:
    messages: List[Message] = field(default_factory=list)
    system_prompt: str = ""
    id: str = field(default_factory=lambda: f"conv_{int(time.time())}")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, **metadata) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content, metadata=metadata)
        self.messages.append(message)
        return message

    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary format."""
        return {
            "id": self.id,
            "system_prompt": self.system_prompt,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create a conversation from dictionary data."""
        conv = cls(
            id=data.get("id", f"conv_{int(time.time())}"),
            system_prompt=data.get("system_prompt", ""),
            metadata=data.get("metadata", {}),
        )
        conv.messages = [Message.from_dict(msg) for msg in data.get("messages", [])]
        return conv


class MemoryManager:
    def __init__(self, db=None, max_history_size: int = 10):
        """Initialize the memory manager with limited history size"""
        self.conversations = {}
        self.max_history_size = max_history_size
        self._locks = {}  # For thread-safe operations
        self.db = db  # Database connection
        self.token_limit = 4096  # Default token limit
        self.tokenizer = None  # Will be set if using a specific tokenizer

    async def add_user_message(
        self,
        conversation_id: str,
        content: str,
        user_id: str = None,
        timestamp: float = None,
    ) -> None:
        """Add a user message to the conversation."""
        conversation = self.get_or_create_conversation(conversation_id)
        metadata = {"user_id": user_id} if user_id else {}
        if timestamp:
            metadata["timestamp"] = timestamp
        conversation.add_message("user", content, **metadata)

        # Trim history if needed
        if len(conversation.messages) > self.max_history_size:
            conversation.messages = conversation.messages[-self.max_history_size :]

    async def add_assistant_message(
        self, conversation_id: str, content: str, timestamp: float = None
    ) -> None:
        """Add an assistant message to the conversation."""
        conversation = self.get_or_create_conversation(conversation_id)
        metadata = {"timestamp": timestamp} if timestamp else {}
        conversation.add_message("assistant", content, **metadata)

        # Trim history if needed
        if len(conversation.messages) > self.max_history_size:
            conversation.messages = conversation.messages[-self.max_history_size :]

    def add_system_message(self, conversation_id: str, content: str) -> Message:
        """Add a system message to the conversation."""
        conversation = self.get_or_create_conversation(conversation_id)
        return conversation.add_message("system", content)

    def get_formatted_history(
        self, conversation_id: str, max_messages: int = None
    ) -> List[Dict[str, Any]]:
        """Get formatted conversation history in a format compatible with LLM APIs."""
        conversation = self.get_or_create_conversation(conversation_id)
        messages = []

        # Add system prompt if available
        if conversation.system_prompt:
            messages.append({"role": "system", "content": conversation.system_prompt})

        # Add conversation history, limiting to max_messages if specified
        msg_list = conversation.messages
        if max_messages is not None:
            msg_list = msg_list[-max_messages:]

        for message in msg_list:
            messages.append({"role": message.role, "content": message.content})

        return messages

    def get_messages(self, conversation_id: str, limit: int = None) -> List[Message]:
        """Get raw messages from the conversation."""
        conversation = self.get_or_create_conversation(conversation_id)
        if limit:
            return conversation.messages[-limit:]
        return conversation.messages

    def clear_conversation(self, conversation_id: str) -> None:
        """Clear all messages from a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id].messages = []
            # Save empty conversation to database if configured
            if self.db:
                self._safe_save_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation completely."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            # Delete from database if configured
            if self.db:
                self._safe_delete_conversation(conversation_id)
            # Clean up lock
            if conversation_id in self._locks:
                del self._locks[conversation_id]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback token counting (approximation)
            return len(text.split())

    async def _maybe_manage_context_window(self, conversation_id: str) -> None:
        """Check and potentially manage the context window if it exceeds the limit."""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return

        # Count total tokens
        total_tokens = self._calculate_conversation_tokens(conversation)

        if total_tokens > self.token_limit:
            await self._reduce_context(conversation_id, total_tokens)

    def _calculate_conversation_tokens(self, conversation: Conversation) -> int:
        """Calculate the total tokens in a conversation."""
        total = 0

        # Count system prompt
        if conversation.system_prompt:
            total += self.count_tokens(conversation.system_prompt)

        # Count messages
        for msg in conversation.messages:
            total += self.count_tokens(msg.content)
            # Add overhead for message format (role, formatting, etc.)
            total += 4  # Approximation for message metadata overhead

        return total

    async def _reduce_context(self, conversation_id: str, current_tokens: int) -> None:
        """Reduce the context window using various strategies."""
        conversation = self.conversations.get(conversation_id)
        if not conversation or len(conversation.messages) < 3:
            return

        # Strategy 1: Remove oldest messages until under limit
        target_token_count = int(
            self.token_limit * 0.7
        )  # Target 70% of limit after reduction

        # Keep at least the latest user-assistant exchange
        keep_count = 2
        preserve_messages = conversation.messages[-keep_count:]

        # First try: Generate a summary of older messages if possible
        if len(conversation.messages) > keep_count + 2:
            summary = await self._generate_summary(conversation_id)
            if summary:
                # Replace old messages with summary
                conversation.messages = [
                    Message(
                        role="system",
                        content=f"Previous conversation summary: {summary}",
                    )
                ] + preserve_messages
                return

        # If we can't summarize or it failed, remove oldest messages one by one
        while len(conversation.messages) > keep_count:
            # Remove oldest message
            conversation.messages.pop(0)

            # Recalculate tokens
            new_token_count = self._calculate_conversation_tokens(conversation)

            if new_token_count <= target_token_count:
                break

    async def _generate_summary(self, conversation_id: str) -> Optional[str]:
        """Generate a summary of the conversation using the assistant."""
        try:
            # This method would use the LLM to generate a summary
            # For now, returning a simple concatenation summary
            conversation = self.conversations.get(conversation_id)
            if not conversation or len(conversation.messages) < 3:
                return None

            # Exclude most recent messages that we'll preserve
            messages_to_summarize = conversation.messages[:-2]

            # Create a simple summary by extracting key information
            # In a real implementation, you'd call the LLM API here
            summary_parts = []
            for msg in messages_to_summarize:
                if msg.role == "user":
                    # Extract first sentence or portion of user queries
                    content = msg.content.strip()
                    if len(content) > 50:
                        content = content[:50] + "..."
                    summary_parts.append(f"User asked: {content}")
                elif msg.role == "assistant" and len(msg.content) > 100:
                    # For assistant responses, extract conclusion if possible
                    lines = msg.content.split("\n")
                    if len(lines) > 1:
                        summary_parts.append(f"Assistant explained about: {lines[-1]}")

            if not summary_parts:
                return None

            return " ".join(summary_parts)

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return None

    async def _safe_save_conversation(self, conversation_id: str) -> None:
        """Safely save conversation to database with multiple approaches."""
        if not self.db or conversation_id not in self.conversations:
            return

        try:
            conv_data = self.conversations[conversation_id].to_dict()

            # Try different approaches to save based on database interface

            # Approach 1: Direct method call
            if hasattr(self.db, "save_conversation") and callable(
                getattr(self.db, "save_conversation")
            ):
                await asyncio.to_thread(
                    self.db.save_conversation, conversation_id, conv_data
                )
                return

            # Approach 2: MongoDB collection
            if hasattr(self.db, "conversations"):
                collection = self.db.conversations
                if hasattr(collection, "update_one") and callable(
                    getattr(collection, "update_one")
                ):
                    await asyncio.to_thread(
                        collection.update_one,
                        {"id": conversation_id},
                        {"$set": conv_data},
                        upsert=True,
                    )
                    return

            # Approach 3: Direct update_one
            if hasattr(self.db, "update_one") and callable(
                getattr(self.db, "update_one")
            ):
                await asyncio.to_thread(
                    self.db.update_one,
                    {"id": conversation_id},
                    {"$set": conv_data},
                    upsert=True,
                )
                return

            logger.warning(
                f"Could not find a suitable method to save conversation {conversation_id}"
            )
        except Exception as e:
            logger.error(f"Failed to save conversation {conversation_id}: {str(e)}")

    def _safe_delete_conversation(self, conversation_id: str) -> None:
        """Safely delete conversation from database with multiple approaches."""
        if not self.db:
            return

        try:
            # Try different approaches to delete based on database interface

            # Approach 1: Direct method call
            if hasattr(self.db, "delete_conversation") and callable(
                getattr(self.db, "delete_conversation")
            ):
                self.db.delete_conversation(conversation_id)
                return

            # Approach 2: MongoDB collection
            if hasattr(self.db, "conversations"):
                collection = self.db.conversations
                if hasattr(collection, "delete_one") and callable(
                    getattr(collection, "delete_one")
                ):
                    collection.delete_one({"id": conversation_id})
                    return

            # Approach 3: Direct delete_one
            if hasattr(self.db, "delete_one") and callable(
                getattr(self.db, "delete_one")
            ):
                self.db.delete_one({"id": conversation_id})
                return

            logger.warning(
                f"Could not find a suitable method to delete conversation {conversation_id}"
            )
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {str(e)}")

    def _get_lock(self, conversation_id: str) -> asyncio.Lock:
        """Get the lock for a conversation, creating it if necessary."""
        if conversation_id not in self._locks:
            self._locks[conversation_id] = asyncio.Lock()
        return self._locks[conversation_id]

    def get_conversation_summary(
        self, conversation_id: str, max_length: int = 100
    ) -> str:
        """Get a short summary of the conversation for display purposes."""
        conversation = self.get_or_create_conversation(conversation_id)

        if not conversation.messages:
            return "No messages"

        # Get the most recent exchange
        last_messages = (
            conversation.messages[-2:]
            if len(conversation.messages) >= 2
            else conversation.messages
        )

        # Create a simple summary
        summary_parts = []
        for msg in last_messages:
            content = msg.content.strip()
            if len(content) > max_length:
                content = content[:max_length] + "..."
            summary_parts.append(
                f"{'User' if msg.role == 'user' else 'Assistant'}: {content}"
            )

        return "\n".join(summary_parts)

    def get_or_create_conversation(self, conversation_id: str) -> Conversation:
        """Get an existing conversation or create a new one."""
        if conversation_id not in self.conversations:
            # Convert old format to new format if it exists in old format
            old_messages = self.conversations.get(conversation_id, [])
            conversation = Conversation(id=conversation_id)

            # If there were messages in old format, convert them
            for msg in old_messages:
                conversation.add_message(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=msg.get("timestamp", time.time()),
                    **{
                        k: v
                        for k, v in msg.items()
                        if k not in ["role", "content", "timestamp"]
                    },
                )

            self.conversations[conversation_id] = conversation
            return conversation

        return self.conversations[conversation_id]
