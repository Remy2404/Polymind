import logging
import asyncio
from typing import Dict, List, Any, Optional
import time
import uuid
import re
from dataclasses import dataclass, field
from .user_profile_manager import UserProfileManager
from .persistence_manager import PersistenceManager
from .semantic_search_manager import SemanticSearchManager
from .group_memory_operations import GroupMemoryOperations
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from database.connection import get_database

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
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
    """Enhanced memory manager with modular components for better maintainability"""

    def __init__(self, db=None, client=None, storage_path=None):
        if db is None:
            try:
                self.db, self.client = get_database()
                if self.db is not None:
                    logger.info("Connected to MongoDB for memory management")
                else:
                    logger.warning(
                        "MongoDB connection failed, memory manager will not persist data"
                    )
                    self.client = None
            except Exception as e:
                logger.error(f"Error connecting to MongoDB: {e}")
                self.db = None
                self.client = None
        else:
            self.db = db
            self.client = client
        self.user_profile_manager = UserProfileManager(self.db)
        self.persistence_manager = PersistenceManager(self.db, storage_path)
        self.semantic_search_manager = SemanticSearchManager()
        self.group_operations = GroupMemoryOperations()
        self.memory_cache = {}
        self.group_memory_cache = {}
        self.conversation_summaries = {}
        self.group_summaries = {}
        self.lock = asyncio.Lock()
        self.importance_factors = {
            "recency": 0.3,
            "relevance": 0.4,
            "interaction": 0.2,
            "media": 0.1,
        }
        # Context management settings
        # Context management settings (Simplified)
        self.short_term_limit = 15 # Increased for better immediate context
        self.summary_min_messages = 20
        
        if self.db is not None:
            self.persistence_manager.ensure_indexes()
        logger.info("Lightweight MemoryManager initialized")

    async def add_user_message(
        self,
        conversation_id: str,
        content: str,
        user_id: str,
        message_type: str = "text",
        importance: float = 0.5,
        is_group: bool = False,
        group_id: Optional[str] = None,
        **metadata,
    ) -> None:
        """Add a user message with optimized storage."""
        message = {
            "role": "user",
            "content": content,
            "timestamp": time.time(),
            "user_id": user_id,
            "message_type": message_type,
            "importance": importance,
            "is_group": is_group,
            "group_id": group_id,
            "metadata": metadata,
        }
        cache_key = group_id if is_group and group_id else conversation_id
        self._ensure_message_id(cache_key, message, "user")
        async with self.lock:
            target_cache = self.group_memory_cache if is_group else self.memory_cache
            if cache_key not in target_cache:
                target_cache[cache_key] = []
            
            target_cache[cache_key].append(message)
            
            # Additional group ops if needed
            if is_group and group_id:
                await self.group_operations.update_group_context(group_id, message)

            # Optimisation: Removed vector storage call

            # Persist
            persist_key = group_id if is_group else conversation_id
            if persist_key:
                 await self.persistence_manager.persist_memory(
                    persist_key,
                    self._get_memory_data(persist_key, is_group),
                    is_group,
                 )

    async def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        message_type: str = "text",
        importance: float = 0.5,
        is_group: bool = False,
        group_id: Optional[str] = None,
        **metadata,
    ) -> None:
        """Add an assistant message with optimized storage."""
        message = {
            "role": "assistant",
            "content": content,
            "timestamp": time.time(),
            "message_type": message_type,
            "importance": importance,
            "is_group": is_group,
            "group_id": group_id,
            "metadata": metadata,
        }
        cache_key = group_id if is_group and group_id else conversation_id
        self._ensure_message_id(cache_key, message, "assistant")
        async with self.lock:
            target_cache = self.group_memory_cache if is_group else self.memory_cache
            if cache_key not in target_cache:
                target_cache[cache_key] = []
            
            target_cache[cache_key].append(message)

            if is_group and group_id:
                 await self.group_operations.update_group_context(group_id, message)
                 await self.group_operations.update_shared_knowledge(group_id, content)
            
            # Optimisation: Removed vector storage call

            # Auto-summary trigger
            msgs_count = len(target_cache.get(cache_key, []))
            if msgs_count % 20 == 0:
                 if cache_key:
                    await self._generate_conversation_summary(cache_key, is_group)

            # Persist
            persist_key = group_id if is_group else conversation_id
            if persist_key:
                await self.persistence_manager.persist_memory(
                    persist_key,
                    self._get_memory_data(persist_key, is_group),
                    is_group,
                )

    async def get_relevant_memory(
        self,
        conversation_id: str,
        query: str,
        limit: int = 5,
        min_relevance: float = 0.6,
        is_group: bool = False,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memory using simple keyword matching (replacing semantic search)."""
        cache_key = conversation_id
        await self.load_memory(cache_key, is_group)
        
        target_cache = self.group_memory_cache if is_group else self.memory_cache
        messages = target_cache.get(cache_key, [])
        
        if not messages:
            return []

        # Simple keyword matching
        query_words = set(query.lower().split())
        match_scores = []
        
        for msg in messages:
            content = msg.get("content", "").lower()
            if not content: continue
            
            score = 0
            for word in query_words:
                if word in content:
                    score += 1
            
            if score > 0:
                match_scores.append((msg, score))
        
        # Sort by score desc, then timestamp desc
        match_scores.sort(key=lambda x: (x[1], x[0].get("timestamp", 0)), reverse=True)
        
        return [item[0] for item in match_scores[:limit]]

    async def get_short_term_memory(
        self, conversation_id: str, limit: int = 10, is_group: bool = False
    ) -> List[Dict[str, Any]]:
        """Get most recent messages."""
        await self.load_memory(conversation_id, is_group)
        target_cache = self.group_memory_cache if is_group else self.memory_cache
        messages = target_cache.get(conversation_id, [])
        return messages[-limit:] if messages else []

    async def get_conversation_summary(
        self, conversation_id: str, is_group: bool = False
    ) -> Optional[str]:
        """Get the current conversation summary."""
        await self.load_memory(conversation_id, is_group)
        summary_cache = self.group_summaries if is_group else self.conversation_summaries
        return summary_cache.get(conversation_id)

    def clear_conversation(self, conversation_id: str, is_group: bool = False):
        """Clear conversation from memory."""
        target_cache = self.group_memory_cache if is_group else self.memory_cache
        target_cache.pop(conversation_id, None)
        
        summary_cache = self.group_summaries if is_group else self.conversation_summaries
        summary_cache.pop(conversation_id, None)
        
        if is_group:
            self.group_operations.clear_group_data(conversation_id)

    async def load_memory(self, cache_key: str, is_group: bool = False) -> None:
        """Load memory into cache if not present"""
        target_cache = self.group_memory_cache if is_group else self.memory_cache
        if cache_key in target_cache:
            return

        async with self.lock:
            # Double check pattern
            if cache_key in target_cache:
                return
            
            memory_data = await self.persistence_manager.load_memory(cache_key, is_group)
            if memory_data:
                self._populate_cache_from_data(cache_key, memory_data, is_group)
            else:
                 # Initialize empty
                 target_cache[cache_key] = []

    async def build_context_bundle(
        self,
        cache_key: str,
        limit: int = 15,
        include_summary: bool = True,
        include_highlights: bool = False, # Force False as heuristics removed
        is_group: bool = False,
    ) -> Dict[str, Any]:
        """Build a simplified context bundle: recent messages + summary."""
        key = cache_key
        if is_group and cache_key is None:
            key = "group"
        
        await self.load_memory(key, is_group)
        
        message_cache = (
            self.group_memory_cache.get(key, [])
            if is_group
            else self.memory_cache.get(key, [])
        )
        
        if not message_cache:
            return {"recent": [], "highlights": [], "summary": None}

        # Just get the recent messages
        recent_limit = max(limit, self.short_term_limit)
        recent_slice = message_cache[-recent_limit:]
        
        recent_messages = []
        for msg in recent_slice:
            if msg.get("content"):
                cloned = self._clone_message_for_context(
                    msg,
                    {"context_type": "recent", "message_id": msg.get("message_id")}
                )
                recent_messages.append(cloned)

        summary: Optional[str] = None
        if include_summary:
            summary_cache = (
                self.group_summaries if is_group else self.conversation_summaries
            )
            summary = summary_cache.get(key)
            if not summary and len(message_cache) >= self.summary_min_messages:
                 summary = await self._generate_conversation_summary(key, is_group)

        return {"recent": recent_messages, "highlights": [], "summary": summary}

    def _generate_message_id(self, cache_key: Optional[str], role: str) -> str:
        """Generate a stable unique message identifier"""
        key = (cache_key or "conversation").replace(" ", "_")
        return f"{key}:{role}:{uuid.uuid4().hex}"

    def _ensure_message_id(
        self, cache_key: Optional[str], message: Dict[str, Any], role: str
    ) -> str:
        """Ensure a message dictionary contains a message_id"""
        if "message_id" not in message or not message.get("message_id"):
            message["message_id"] = self._generate_message_id(cache_key, role)
        return message["message_id"]

    def _clone_message_for_context(
        self, message: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clone a message and apply overrides for context building"""
        msg_copy = message.copy()
        msg_copy.update(overrides)
        return msg_copy

    def _get_memory_data(self, cache_key: str, is_group: bool) -> Dict[str, Any]:
        """Get memory data for persistence"""
        memory_data = {
            "cache_key": cache_key,
            "messages": (
                self.group_memory_cache.get(cache_key, [])
                if is_group
                else self.memory_cache.get(cache_key, [])
            ),
            "summary": (
                self.group_summaries.get(cache_key)
                if is_group
                else self.conversation_summaries.get(cache_key)
            ),
            "is_group": is_group,
            "last_updated": time.time(),
        }
        if is_group and cache_key:
            memory_data["shared_knowledge"] = (
                self.group_operations.get_shared_knowledge_for_group(cache_key)
            )
        return memory_data

    def _populate_cache_from_data(
        self, cache_key: str, memory_data: Dict[str, Any], is_group: bool
    ):
        """Populate cache from loaded memory data"""
        if is_group:
            self.group_memory_cache[cache_key] = memory_data.get("messages", [])
            if memory_data.get("summary"):
                self.group_summaries[cache_key] = memory_data["summary"]
            if memory_data.get("shared_knowledge"):
                self.group_operations.shared_knowledge[cache_key] = memory_data[
                    "shared_knowledge"
                ]
        else:
            self.memory_cache[cache_key] = memory_data.get("messages", [])
            if memory_data.get("summary"):
                self.conversation_summaries[cache_key] = memory_data["summary"]

    async def _generate_conversation_summary(
        self, cache_key: str, is_group: bool = False
    ) -> str:
        """Generate a summary of the conversation"""
        try:
            from collections import defaultdict

            message_cache = (
                self.group_memory_cache.get(cache_key, [])
                if is_group
                else self.memory_cache.get(cache_key, [])
            )
            if not message_cache:
                return "No conversation history available."
            recent_messages = message_cache[-50:]
            topics = []
            user_contributions = defaultdict(list)
            for message in recent_messages:
                content = message.get("content", "")
                user_id = message.get("user_id", "assistant")
                if len(content) > 20:
                    topics.append(content)
                    user_contributions[user_id].append(content[:100])
            if is_group:
                participants = len(user_contributions)
                summary = f"Group conversation with {participants} participants. "
                summary += f"Total messages: {len(recent_messages)}. "
                if user_contributions:
                    most_active = max(
                        user_contributions.items(), key=lambda x: len(x[1])
                    )
                    summary += f"Most active participant: User {most_active[0]} ({len(most_active[1])} messages). "
            else:
                summary = f"Individual conversation with {len(recent_messages)} recent messages. "
            if topics:
                recent_topics = topics[-5:]
                summary += "Recent topics: " + "; ".join(
                    [t[:50] + "..." for t in recent_topics]
                )
            summary_cache = (
                self.group_summaries if is_group else self.conversation_summaries
            )
            summary_cache[cache_key] = summary
            return summary
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return "Summary generation failed."
