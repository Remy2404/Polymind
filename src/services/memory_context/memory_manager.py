import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import aiofiles
import os
import re
from collections import defaultdict
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
    """Enhanced memory manager with semantic search, summarization, and group support"""

    def __init__(self, db=None, storage_path="./data/memory"):
        self.db = db
        self.storage_path = storage_path
        self.memory_cache = {}
        self.group_memory_cache = {}  # Separate cache for group memories
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)

        # Enhanced memory features
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
        self.message_vectors = {}
        self.group_message_vectors = {}  # Separate vectors for group messages
        self.conversation_summaries = {}
        self.group_summaries = {}  # Group conversation summaries

        # Memory importance scoring
        self.importance_factors = {
            "recency": 0.3,  # How recent the message is
            "relevance": 0.4,  # How relevant to current context
            "interaction": 0.2,  # How much interaction it generated
            "media": 0.1,  # Bonus for media content
        }

        # Group collaboration features
        self.group_contexts = {}  # Active group conversation contexts
        self.shared_knowledge = {}  # Shared knowledge between group members

        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)

        logger.info(
            "Enhanced MemoryManager initialized with semantic search and group support"
        )

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
            # Store in appropriate cache based on group or individual
            if is_group and group_id:
                if group_id not in self.group_memory_cache:
                    self.group_memory_cache[group_id] = []
                self.group_memory_cache[group_id].append(message)

                # Update group context
                await self._update_group_context(group_id, message)

                # Store group message vectors for semantic search
                await self._store_group_message_vector(
                    group_id, content, len(self.group_memory_cache[group_id]) - 1
                )
            else:
                if conversation_id not in self.memory_cache:
                    self.memory_cache[conversation_id] = []
                self.memory_cache[conversation_id].append(message)

                # Store message vectors for semantic search
                await self._store_message_vector(
                    conversation_id,
                    content,
                    len(self.memory_cache[conversation_id]) - 1,
                )

            # Persist to storage
            await self._persist_memory(
                conversation_id if not is_group else group_id, is_group
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

                # Update group context and shared knowledge
                await self._update_group_context(group_id, message)
                await self._update_shared_knowledge(group_id, content)

                # Store group message vectors
                await self._store_group_message_vector(
                    group_id, content, len(self.group_memory_cache[group_id]) - 1
                )
            else:
                if conversation_id not in self.memory_cache:
                    self.memory_cache[conversation_id] = []
                self.memory_cache[conversation_id].append(message)

                # Store message vectors
                await self._store_message_vector(
                    conversation_id,
                    content,
                    len(self.memory_cache[conversation_id]) - 1,
                )

            # Auto-generate conversation summary if needed
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
                await self._generate_conversation_summary(cache_key, is_group)

            await self._persist_memory(
                conversation_id if not is_group else group_id, is_group
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
            message_cache = (
                self.group_memory_cache.get(group_id, [])
                if is_group
                else self.memory_cache.get(conversation_id, [])
            )

            if not message_cache:
                return []

            # Get semantic similarity scores
            relevant_messages = await self._semantic_search(cache_key, query, is_group)

            # If this is a group and we want to include shared knowledge
            if is_group and include_group_knowledge and group_id:
                shared_knowledge = await self._get_shared_knowledge(group_id, query)
                relevant_messages.extend(shared_knowledge)

            # Sort by combined relevance and importance score
            scored_messages = []
            for msg_idx, similarity in relevant_messages[
                : limit * 2
            ]:  # Get more candidates
                if msg_idx < len(message_cache):
                    message = message_cache[msg_idx]
                    combined_score = self._calculate_message_importance(
                        message, similarity
                    )
                    scored_messages.append((message, combined_score))

            # Sort by score and return top results
            scored_messages.sort(key=lambda x: x[1], reverse=True)
            return [msg for msg, score in scored_messages[:limit]]

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            # Fallback to recent messages
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
        message_cache = (
            self.group_memory_cache.get(group_id, [])
            if is_group
            else self.memory_cache.get(conversation_id, [])
        )

        # If cache is empty, try to load from storage
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
        summary_cache = (
            self.group_summaries if is_group else self.conversation_summaries
        )

        if cache_key in summary_cache:
            return summary_cache[cache_key]

        # Generate new summary
        return await self._generate_conversation_summary(cache_key, is_group)

    async def clear_conversation(
        self,
        conversation_id: str,
        is_group: bool = False,
        group_id: Optional[str] = None,
    ) -> None:
        """Clear conversation memory with group support"""
        async with self.lock:
            cache_key = group_id if is_group else conversation_id

            if is_group:
                self.group_memory_cache.pop(group_id, None)
                self.group_message_vectors.pop(group_id, None)
                self.group_summaries.pop(group_id, None)
                self.group_contexts.pop(group_id, None)
            else:
                self.memory_cache.pop(conversation_id, None)
                self.message_vectors.pop(conversation_id, None)
                self.conversation_summaries.pop(conversation_id, None)

            # Remove from persistent storage
            file_path = os.path.join(
                self.storage_path, f"{'group_' if is_group else ''}{cache_key}.json"
            )
            if os.path.exists(file_path):
                os.remove(file_path)

    # Enhanced group-specific methods

    async def get_group_participants(self, group_id: str) -> List[str]:
        """Get list of participants in a group conversation"""
        if group_id not in self.group_memory_cache:
            return []

        participants = set()
        for message in self.group_memory_cache[group_id]:
            if message.get("user_id"):
                participants.add(message["user_id"])

        return list(participants)

    async def get_group_activity_summary(
        self, group_id: str, days: int = 7
    ) -> Dict[str, Any]:
        """Get group activity summary for specified days"""
        if group_id not in self.group_memory_cache:
            return {}

        cutoff_time = time.time() - (days * 24 * 60 * 60)
        recent_messages = [
            msg
            for msg in self.group_memory_cache[group_id]
            if msg.get("timestamp", 0) > cutoff_time
        ]

        # Analyze activity patterns
        user_activity = defaultdict(int)
        message_types = defaultdict(int)
        topics = []

        for message in recent_messages:
            if message.get("user_id"):
                user_activity[message["user_id"]] += 1
            message_types[message.get("message_type", "text")] += 1

            # Extract key topics (simplified)
            content = message.get("content", "")
            if len(content) > 20:  # Meaningful content
                topics.append(content[:100])  # First 100 chars as topic indicator

        return {
            "total_messages": len(recent_messages),
            "active_users": len(user_activity),
            "user_activity": dict(user_activity),
            "message_types": dict(message_types),
            "most_active_user": (
                max(user_activity.items(), key=lambda x: x[1])[0]
                if user_activity
                else None
            ),
            "summary": await self.get_conversation_summary("", True, group_id),
        }

    # Private helper methods

    async def _store_message_vector(
        self, conversation_id: str, content: str, message_index: int
    ):
        """Store message vector for semantic search"""
        try:
            if conversation_id not in self.message_vectors:
                self.message_vectors[conversation_id] = {}

            # Simple TF-IDF vectorization (in production, use better embeddings)
            words = re.findall(r"\w+", content.lower())
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1

            self.message_vectors[conversation_id][message_index] = {
                "content": content,
                "words": dict(word_freq),
                "length": len(content),
            }
        except Exception as e:
            logger.error(f"Error storing message vector: {e}")

    async def _store_group_message_vector(
        self, group_id: str, content: str, message_index: int
    ):
        """Store group message vector for semantic search"""
        try:
            if group_id not in self.group_message_vectors:
                self.group_message_vectors[group_id] = {}

            words = re.findall(r"\w+", content.lower())
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1

            self.group_message_vectors[group_id][message_index] = {
                "content": content,
                "words": dict(word_freq),
                "length": len(content),
            }
        except Exception as e:
            logger.error(f"Error storing group message vector: {e}")

    async def _semantic_search(
        self, cache_key: str, query: str, is_group: bool = False
    ) -> List[Tuple[int, float]]:
        """Perform semantic search on messages"""
        try:
            vector_cache = (
                self.group_message_vectors if is_group else self.message_vectors
            )

            if cache_key not in vector_cache:
                return []

            query_words = set(re.findall(r"\w+", query.lower()))
            similarities = []

            for msg_idx, vector_data in vector_cache[cache_key].items():
                # Simple cosine similarity based on word overlap
                msg_words = set(vector_data["words"].keys())
                intersection = len(query_words & msg_words)
                union = len(query_words | msg_words)

                if union > 0:
                    similarity = intersection / union
                    # Boost similarity for longer, more detailed messages
                    length_boost = min(vector_data["length"] / 200, 1.5)
                    final_similarity = similarity * length_boost

                    if final_similarity > 0.1:  # Minimum threshold
                        similarities.append((msg_idx, final_similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def _calculate_message_importance(
        self, message: Dict[str, Any], relevance_score: float
    ) -> float:
        """Calculate combined importance score for a message"""
        current_time = time.time()
        message_time = message.get("timestamp", current_time)

        # Recency factor (newer messages get higher scores)
        time_diff = current_time - message_time
        recency_score = max(0, 1 - (time_diff / (7 * 24 * 3600)))  # Decay over 7 days

        # Base importance from message metadata
        base_importance = message.get("importance", 0.5)

        # Media bonus
        media_bonus = 0.2 if message.get("message_type") != "text" else 0

        # Calculate weighted score
        final_score = (
            relevance_score * self.importance_factors["relevance"]
            + recency_score * self.importance_factors["recency"]
            + base_importance * self.importance_factors["interaction"]
            + media_bonus * self.importance_factors["media"]
        )

        return final_score

    async def _generate_conversation_summary(
        self, cache_key: str, is_group: bool = False
    ) -> str:
        """Generate a summary of the conversation"""
        try:
            message_cache = (
                self.group_memory_cache.get(cache_key, [])
                if is_group
                else self.memory_cache.get(cache_key, [])
            )

            if not message_cache:
                return "No conversation history available."

            # Get recent messages for summary
            recent_messages = message_cache[-50:]  # Last 50 messages

            # Extract key topics and themes
            topics = []
            user_contributions = defaultdict(list)

            for message in recent_messages:
                content = message.get("content", "")
                user_id = message.get("user_id", "assistant")

                if len(content) > 20:  # Meaningful content
                    topics.append(content)
                    user_contributions[user_id].append(content[:100])

            # Generate simple summary
            if is_group:
                participants = len(user_contributions)
                summary = f"Group conversation with {participants} participants. "
                summary += f"Total messages: {len(recent_messages)}. "

                # Add top contributors
                if user_contributions:
                    most_active = max(
                        user_contributions.items(), key=lambda x: len(x[1])
                    )
                    summary += f"Most active participant: User {most_active[0]} ({len(most_active[1])} messages). "
            else:
                summary = f"Individual conversation with {len(recent_messages)} recent messages. "

            # Add recent topics (simplified)
            if topics:
                recent_topics = topics[-5:]  # Last 5 topics
                summary += "Recent topics: " + "; ".join(
                    [t[:50] + "..." for t in recent_topics]
                )

            # Cache the summary
            summary_cache = (
                self.group_summaries if is_group else self.conversation_summaries
            )
            summary_cache[cache_key] = summary

            return summary

        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            return "Summary generation failed."

    async def _update_group_context(self, group_id: str, message: Dict[str, Any]):
        """Update active group conversation context"""
        if group_id not in self.group_contexts:
            self.group_contexts[group_id] = {
                "active_users": set(),
                "current_topic": None,
                "last_activity": time.time(),
                "message_count": 0,
            }

        context = self.group_contexts[group_id]

        # Update active users
        if message.get("user_id"):
            context["active_users"].add(message["user_id"])

        # Update activity timestamp
        context["last_activity"] = time.time()
        context["message_count"] += 1

        # Simple topic detection (can be enhanced with NLP)
        content = message.get("content", "").lower()
        if any(
            keyword in content for keyword in ["project", "task", "deadline", "meeting"]
        ):
            context["current_topic"] = "work_discussion"
        elif any(
            keyword in content for keyword in ["help", "question", "how", "what", "why"]
        ):
            context["current_topic"] = "help_request"
        else:
            context["current_topic"] = "general_chat"

    async def _update_shared_knowledge(self, group_id: str, content: str):
        """Update shared knowledge base for the group"""
        if group_id not in self.shared_knowledge:
            self.shared_knowledge[group_id] = []

        # Extract potential knowledge items (simplified)
        # In production, use NLP to extract entities, facts, etc.
        if len(content) > 50:  # Substantial content
            knowledge_item = {
                "content": content,
                "timestamp": time.time(),
                "importance": (
                    0.7
                    if any(
                        keyword in content.lower()
                        for keyword in [
                            "remember",
                            "important",
                            "note",
                            "fact",
                            "definition",
                        ]
                    )
                    else 0.5
                ),
            }

            self.shared_knowledge[group_id].append(knowledge_item)

            # Keep only the most recent/important knowledge items
            if len(self.shared_knowledge[group_id]) > 100:
                # Sort by importance and recency, keep top 100
                self.shared_knowledge[group_id].sort(
                    key=lambda x: (x["importance"], x["timestamp"]),
                    reverse=True,
                )
                self.shared_knowledge[group_id] = self.shared_knowledge[group_id][:100]

    async def _get_shared_knowledge(
        self, group_id: str, query: str
    ) -> List[Tuple[int, float]]:
        """Get relevant shared knowledge for a query"""
        if group_id not in self.shared_knowledge:
            return []

        query_words = set(re.findall(r"\w+", query.lower()))
        relevant_knowledge = []

        for idx, knowledge_item in enumerate(self.shared_knowledge[group_id]):
            content_words = set(re.findall(r"\w+", knowledge_item["content"].lower()))
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)

            if union > 0:
                similarity = intersection / union * knowledge_item["importance"]
                if similarity > 0.2:
                    relevant_knowledge.append(
                        (-(idx + 1000), similarity)
                    )  # Negative index to distinguish from messages

        return relevant_knowledge

    async def _persist_memory(self, cache_key: str, is_group: bool = False):
        """Persist memory to MongoDB storage"""
        try:
            if self.db is None:
                # Fallback to file storage if no MongoDB connection
                await self._persist_memory_file(cache_key, is_group)
                return

            # Use MongoDB for persistence
            collection = (
                self.db.group_conversations if is_group else self.db.conversations
            )

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

            if is_group and cache_key in self.shared_knowledge:
                memory_data["shared_knowledge"] = self.shared_knowledge[cache_key]

            # Update or insert conversation data
            collection.update_one(
                {"cache_key": cache_key}, {"$set": memory_data}, upsert=True
            )

            logger.info(
                f"Persisted {'group' if is_group else 'conversation'} memory to MongoDB for {cache_key}"
            )

        except Exception as e:
            logger.error(f"Error persisting memory to MongoDB: {e}")
            # Fallback to file storage on error
            await self._persist_memory_file(cache_key, is_group)

    async def _persist_memory_file(self, cache_key: str, is_group: bool = False):
        """Fallback file-based persistence"""
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            file_path = os.path.join(
                self.storage_path, f"{'group_' if is_group else ''}{cache_key}.json"
            )

            memory_data = {
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

            if is_group and cache_key in self.shared_knowledge:
                memory_data["shared_knowledge"] = self.shared_knowledge[cache_key]

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(memory_data, ensure_ascii=False, indent=2))

        except Exception as e:
            logger.error(f"Error persisting memory to file: {e}")

    async def load_memory(self, cache_key: str, is_group: bool = False):
        """Load memory from MongoDB or file storage"""
        try:
            if self.db is not None:
                # Try MongoDB first
                collection = (
                    self.db.group_conversations if is_group else self.db.conversations
                )
                memory_data = collection.find_one({"cache_key": cache_key})

                if memory_data:
                    if is_group:
                        self.group_memory_cache[cache_key] = memory_data.get(
                            "messages", []
                        )
                        if memory_data.get("summary"):
                            self.group_summaries[cache_key] = memory_data["summary"]
                        if memory_data.get("shared_knowledge"):
                            self.shared_knowledge[cache_key] = memory_data[
                                "shared_knowledge"
                            ]
                    else:
                        self.memory_cache[cache_key] = memory_data.get("messages", [])
                        if memory_data.get("summary"):
                            self.conversation_summaries[cache_key] = memory_data[
                                "summary"
                            ]

                    logger.info(
                        f"Loaded memory from MongoDB for {'group' if is_group else 'conversation'} {cache_key}"
                    )
                    return

            # Fallback to file storage
            await self._load_memory_file(cache_key, is_group)

        except Exception as e:
            logger.error(f"Error loading memory from MongoDB: {e}")
            # Fallback to file storage on error
            await self._load_memory_file(cache_key, is_group)

    async def _load_memory_file(self, cache_key: str, is_group: bool = False):
        """Fallback file-based loading"""
        try:
            file_path = os.path.join(
                self.storage_path, f"{'group_' if is_group else ''}{cache_key}.json"
            )

            if not os.path.exists(file_path):
                return

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                memory_data = json.loads(content)

            if is_group:
                self.group_memory_cache[cache_key] = memory_data.get("messages", [])
                if memory_data.get("summary"):
                    self.group_summaries[cache_key] = memory_data["summary"]
                if memory_data.get("shared_knowledge"):
                    self.shared_knowledge[cache_key] = memory_data["shared_knowledge"]
            else:
                self.memory_cache[cache_key] = memory_data.get("messages", [])
                if memory_data.get("summary"):
                    self.conversation_summaries[cache_key] = memory_data["summary"]

            logger.info(
                f"Loaded memory from file for {'group' if is_group else 'conversation'} {cache_key}"
            )

        except Exception as e:
            logger.error(f"Error loading memory from file: {e}")

    async def save_user_profile(self, user_id: int, profile_data: Dict[str, Any]):
        """Save user profile information to MongoDB"""
        try:
            if self.db is None:
                logger.warning("No database connection for user profile storage")
                return

            # Store user profile data including name and other personal info
            profile_document = {
                "user_id": user_id,
                "profile_data": profile_data,
                "last_updated": time.time(),
                "created_at": profile_data.get("created_at", time.time()),
            }

            # Update or insert user profile
            self.db.user_profiles.update_one(
                {"user_id": user_id}, {"$set": profile_document}, upsert=True
            )

            logger.info(f"Saved user profile for user {user_id}")

        except Exception as e:
            logger.error(f"Error saving user profile for {user_id}: {e}")

    async def get_user_profile(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve user profile information from MongoDB"""
        try:
            if self.db is None:
                logger.warning("No database connection for user profile retrieval")
                return None

            profile_doc = self.db.user_profiles.find_one({"user_id": user_id})

            if profile_doc:
                return profile_doc.get("profile_data", {})

            return None

        except Exception as e:
            logger.error(f"Error retrieving user profile for {user_id}: {e}")
            return None

    async def update_user_profile_field(self, user_id: int, field: str, value: Any):
        """Update a specific field in user profile"""
        try:
            if self.db is None:
                logger.warning("No database connection for user profile update")
                return

            # Update specific field in profile data
            self.db.user_profiles.update_one(
                {"user_id": user_id},
                {"$set": {f"profile_data.{field}": value, "last_updated": time.time()}},
                upsert=True,
            )

            logger.info(f"Updated {field} for user {user_id}")

        except Exception as e:
            logger.error(
                f"Error updating user profile field {field} for {user_id}: {e}"
            )

    async def extract_and_save_user_info(self, user_id: int, message_content: str):
        """Extract and save user information from message content"""
        try:
            # Simple pattern matching for name extraction
            name_patterns = [
                r"my name is (\w+)",
                r"i'm (\w+)",
                r"i am (\w+)",
                r"call me (\w+)",
                r"name's (\w+)",
            ]

            import re

            for pattern in name_patterns:
                match = re.search(pattern, message_content.lower())
                if match:
                    name = match.group(1).capitalize()
                    await self.update_user_profile_field(user_id, "name", name)
                    logger.info(f"Extracted and saved name '{name}' for user {user_id}")
                    break

        except Exception as e:
            logger.error(f"Error extracting user info from message: {e}")
