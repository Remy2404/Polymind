import logging
import asyncio
from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass, field
from sklearn.feature_extraction.text import TfidfVectorizer
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
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.importance_factors = {
            "recency": 0.3,
            "relevance": 0.4,
            "interaction": 0.2,
            "media": 0.1,
        }
        if self.db is not None:
            self.persistence_manager.ensure_indexes()
        logger.info("Enhanced MemoryManager initialized with modular components")
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
        """Add a user message with enhanced metadata and group support"""
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
        async with self.lock:
            if is_group and group_id:
                if group_id not in self.group_memory_cache:
                    self.group_memory_cache[group_id] = []
                self.group_memory_cache[group_id].append(message)
                await self.group_operations.update_group_context(group_id, message)
                await self.semantic_search_manager.store_group_message_vector(
                    group_id, content, len(self.group_memory_cache[group_id]) - 1
                )
            else:
                if conversation_id not in self.memory_cache:
                    self.memory_cache[conversation_id] = []
                self.memory_cache[conversation_id].append(message)
                await self.semantic_search_manager.store_message_vector(
                    conversation_id,
                    content,
                    len(self.memory_cache[conversation_id]) - 1,
                )
            persist_key = conversation_id if not is_group else group_id
            if persist_key is not None:
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
        """Add an assistant message with enhanced metadata and group support"""
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
        async with self.lock:
            if is_group and group_id:
                if group_id not in self.group_memory_cache:
                    self.group_memory_cache[group_id] = []
                self.group_memory_cache[group_id].append(message)
                await self.group_operations.update_group_context(group_id, message)
                await self.group_operations.update_shared_knowledge(group_id, content)
                await self.semantic_search_manager.store_group_message_vector(
                    group_id, content, len(self.group_memory_cache[group_id]) - 1
                )
            else:
                if conversation_id not in self.memory_cache:
                    self.memory_cache[conversation_id] = []
                self.memory_cache[conversation_id].append(message)
                await self.semantic_search_manager.store_message_vector(
                    conversation_id,
                    content,
                    len(self.memory_cache[conversation_id]) - 1,
                )
            cache_key = group_id if is_group else conversation_id
            if (
                len(
                    self.group_memory_cache.get(group_id, [])
                    if is_group
                    else self.memory_cache.get(conversation_id, [])
                )
                % 20
                == 0
            ):
                if cache_key is not None:
                    await self._generate_conversation_summary(cache_key, is_group)
            persist_key = conversation_id if not is_group else group_id
            if persist_key is not None:
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
        is_group: bool = False,
        group_id: Optional[str] = None,
        include_group_knowledge: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get relevant messages using semantic search with group support"""
        try:
            cache_key = group_id if is_group else conversation_id
            if cache_key is None:
                return []
            message_cache = (
                self.group_memory_cache.get(group_id, [])
                if is_group
                else self.memory_cache.get(conversation_id, [])
            )
            if not message_cache:
                return []
            relevant_messages = await self.semantic_search_manager.semantic_search(
                cache_key, query, is_group
            )
            if is_group and include_group_knowledge and group_id:
                shared_knowledge = await self.group_operations.get_shared_knowledge(
                    group_id, query
                )
                relevant_messages.extend(shared_knowledge)
            scored_messages = []
            for msg_idx, similarity in relevant_messages[: limit * 2]:
                if msg_idx < len(message_cache):
                    message = message_cache[msg_idx]
                    combined_score = (
                        self.semantic_search_manager.calculate_message_importance(
                            message, similarity, self.importance_factors
                        )
                    )
                    scored_messages.append((message, combined_score))
            scored_messages.sort(key=lambda x: x[1], reverse=True)
            return [msg for msg, score in scored_messages[:limit]]
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return await self.get_short_term_memory(
                conversation_id, limit, is_group, group_id
            )
    async def get_short_term_memory(
        self,
        conversation_id: str,
        limit: int = 5,
        is_group: bool = False,
        group_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent messages with group support and auto-loading from storage"""
        cache_key = group_id if is_group else conversation_id
        if cache_key is None:
            return []
        message_cache = (
            self.group_memory_cache.get(group_id, [])
            if is_group
            else self.memory_cache.get(conversation_id, [])
        )
        if not message_cache:
            await self.load_memory(cache_key, is_group)
            message_cache = (
                self.group_memory_cache.get(group_id, [])
                if is_group
                else self.memory_cache.get(conversation_id, [])
            )
        if not message_cache:
            return []
        return message_cache[-limit:]
    async def get_conversation_summary(
        self,
        conversation_id: str,
        is_group: bool = False,
        group_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get or generate conversation summary with group support"""
        cache_key = group_id if is_group else conversation_id
        if cache_key is None:
            return None
        summary_cache = (
            self.group_summaries if is_group else self.conversation_summaries
        )
        if cache_key in summary_cache:
            return summary_cache[cache_key]
        return await self._generate_conversation_summary(cache_key, is_group)
    async def clear_conversation(
        self,
        conversation_id: str,
        is_group: bool = False,
        group_id: Optional[str] = None,
    ) -> None:
        """Clear conversation memory with group support"""
        async with self.lock:
            if is_group:
                if group_id is not None:
                    self.group_memory_cache.pop(group_id, None)
                    self.group_summaries.pop(group_id, None)
                    self.group_operations.clear_group_data(group_id)
            else:
                self.memory_cache.pop(conversation_id, None)
                self.conversation_summaries.pop(conversation_id, None)
    async def get_group_participants(self, group_id: str) -> List[str]:
        """Get list of participants in a group conversation"""
        return await self.group_operations.get_group_participants(
            group_id, self.group_memory_cache
        )
    async def get_group_activity_summary(
        self, group_id: str, days: int = 7
    ) -> Dict[str, Any]:
        """Get group activity summary for specified days"""
        summary = await self.group_operations.get_group_activity_summary(
            group_id, self.group_memory_cache, days
        )
        if summary:
            summary["summary"] = await self.get_conversation_summary("", True, group_id)
        return summary
    async def save_user_profile(self, user_id: int, profile_data: Dict[str, Any]):
        """Save user profile information"""
        return await self.user_profile_manager.save_user_profile(user_id, profile_data)
    async def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve user profile information"""
        return await self.user_profile_manager.get_user_profile(user_id)
    async def update_user_profile_field(self, user_id: int, field: str, value: Any):
        """Update a specific field in user profile"""
        return await self.user_profile_manager.update_user_profile_field(
            user_id, field, value
        )
    async def extract_and_save_user_info(self, user_id: int, message_content: str):
        """Extract and save user information from message content"""
        return await self.user_profile_manager.extract_and_save_user_info(
            user_id, message_content
        )
    async def load_memory(self, cache_key: str, is_group: bool = False):
        """Load memory from storage"""
        memory_data = await self.persistence_manager.load_memory(cache_key, is_group)
        if memory_data:
            self._populate_cache_from_data(cache_key, memory_data, is_group)
    async def get_all_conversation_history(
        self,
        conversation_id: str,
        is_group: bool = False,
        group_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all conversation history for a user/group"""
        try:
            cache_key = group_id if is_group else conversation_id
            if cache_key is None:
                return []
            await self.load_memory(cache_key, is_group)
            messages = (
                self.group_memory_cache.get(cache_key, [])
                if is_group
                else self.memory_cache.get(cache_key, [])
            )
            return messages
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
    async def export_conversation_data(
        self,
        conversation_id: str,
        is_group: bool = False,
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export complete conversation data including summary"""
        try:
            cache_key = group_id if is_group else conversation_id
            if cache_key is None:
                return {}
            await self.load_memory(cache_key, is_group)
            messages = await self.get_all_conversation_history(
                conversation_id, is_group, group_id
            )
            summary = (
                self.group_summaries.get(cache_key)
                if is_group
                else self.conversation_summaries.get(cache_key)
            )
            export_data = {
                "conversation_id": conversation_id,
                "is_group": is_group,
                "group_id": group_id,
                "messages": messages,
                "summary": summary,
                "total_messages": len(messages),
                "exported_at": time.time(),
            }
            if is_group and cache_key:
                export_data["shared_knowledge"] = (
                    self.group_operations.get_shared_knowledge_for_group(cache_key)
                )
            return export_data
        except Exception as e:
            logger.error(f"Error exporting conversation data: {e}")
            return {}
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
