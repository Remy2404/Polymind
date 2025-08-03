"""
Advanced Group Chat Integration System
Handles group conversations, shared memory, and collaborative features.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from telegram import Update, Chat, User, Message
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class GroupRole(Enum):
    """User roles within a group."""

    MEMBER = "member"
    ADMIN = "admin"
    MODERATOR = "moderator"
    BOT_CONTROLLER = "bot_controller"


class ContextScope(Enum):
    """Scope of conversation context."""

    PRIVATE = "private"
    GROUP_SHARED = "group_shared"
    THREAD_LOCAL = "thread_local"
    GLOBAL = "global"


@dataclass
class GroupParticipant:
    """Represents a participant in a group chat."""

    user_id: int
    username: Optional[str]
    full_name: str
    role: GroupRole
    join_date: datetime
    last_active: datetime
    message_count: int = 0
    is_active: bool = True
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupThread:
    """Represents a conversation thread within a group."""

    thread_id: str
    group_id: int
    topic: Optional[str]
    participants: Set[int]
    created_at: datetime
    last_message_at: datetime
    message_count: int = 0
    is_active: bool = True
    context_scope: ContextScope = ContextScope.THREAD_LOCAL


@dataclass
class GroupConversationContext:
    """Shared conversation context for a group."""

    group_id: int
    group_name: str
    shared_memory: Dict[str, Any]
    active_topics: List[str]
    recent_messages: List[Dict[str, Any]]
    participants: Dict[int, GroupParticipant]
    threads: Dict[str, GroupThread]
    created_at: datetime
    updated_at: datetime
    settings: Dict[str, Any] = field(default_factory=dict)


class GroupManager:
    """
    Manages group chat interactions, shared memory, and collaborative features.

    Features:
    - Shared conversation memory across group members
    - Thread-based conversations
    - Role-based permissions
    - Intelligent context switching
    - Group analytics and insights
    """

    def __init__(self, user_data_manager, conversation_manager):
        self.user_data_manager = user_data_manager
        self.conversation_manager = conversation_manager
        self.logger = logging.getLogger(__name__)

        # Group contexts and caches
        self.group_contexts: Dict[int, GroupConversationContext] = {}
        self.active_threads: Dict[str, GroupThread] = {}
        self.participant_cache: Dict[int, Dict[int, GroupParticipant]] = defaultdict(
            dict
        )

        # Activity tracking
        self.message_buffers: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.typing_indicators: Dict[int, Set[int]] = defaultdict(set)

        # Settings
        self.max_shared_memory_size = 50
        self.thread_timeout_hours = 24
        self.max_recent_messages = 100

        self.logger.info("GroupManager initialized")

    async def handle_group_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, message_text: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Handle incoming group messages with intelligent context management.

        Returns:
            Tuple of (processed_message, context_metadata)
        """
        try:
            chat = update.effective_chat
            user = update.effective_user
            message = update.message

            if not chat or chat.type not in ["group", "supergroup"]:
                return message_text, {}

            # Get or create group context
            group_context = await self._get_or_create_group_context(chat, user)

            # Update participant activity
            await self._update_participant_activity(group_context, user, message)

            # Determine conversation thread
            thread = await self._get_conversation_thread(group_context, message)

            # Build enhanced context
            enhanced_context = await self._build_enhanced_context(
                group_context, thread, message, message_text
            )

            # Process message with group context
            processed_message = await self._process_group_message(
                message_text, enhanced_context, group_context, thread
            )

            # Update shared memory
            await self._update_shared_memory(
                group_context, thread, message, processed_message
            )

            # Generate context metadata
            metadata = {
                "group_id": chat.id,
                "thread_id": thread.thread_id if thread else None,
                "participant_count": len(group_context.participants),
                "context_scope": (
                    thread.context_scope.value if thread else "group_shared"
                ),
                "shared_topics": group_context.active_topics[:5],
                "response_context": enhanced_context,
            }

            return processed_message, metadata

        except Exception as e:
            self.logger.error(f"Error handling group message: {e}")
            return message_text, {}

    async def _get_or_create_group_context(
        self, chat: Chat, user: User
    ) -> GroupConversationContext:
        """Get or create group conversation context."""
        group_id = chat.id

        if group_id not in self.group_contexts:
            # Create new group context
            self.group_contexts[group_id] = GroupConversationContext(
                group_id=group_id,
                group_name=chat.title or f"Group {group_id}",
                shared_memory={},
                active_topics=[],
                recent_messages=[],
                participants={},
                threads={},
                created_at=datetime.now(),
                updated_at=datetime.now(),
                settings=await self._get_group_settings(group_id),
            )

            self.logger.info(f"Created new group context for {group_id}")

        group_context = self.group_contexts[group_id]

        # Add or update participant
        if user.id not in group_context.participants:
            participant = GroupParticipant(
                user_id=user.id,
                username=user.username,
                full_name=user.full_name or user.first_name or "Unknown",
                role=GroupRole.MEMBER,
                join_date=datetime.now(),
                last_active=datetime.now(),
            )
            group_context.participants[user.id] = participant
            self.logger.info(f"Added new participant {user.id} to group {group_id}")

        return group_context

    async def _get_conversation_thread(
        self, group_context: GroupConversationContext, message: Message
    ) -> Optional[GroupThread]:
        """Determine or create conversation thread for the message."""
        # Check if replying to a message (creates thread context)
        if message.reply_to_message:
            # Create thread ID based on original message
            original_msg_id = message.reply_to_message.message_id
            thread_id = f"{group_context.group_id}_{original_msg_id}"

            if thread_id not in group_context.threads:
                # Create new thread
                group_context.threads[thread_id] = GroupThread(
                    thread_id=thread_id,
                    group_id=group_context.group_id,
                    topic=None,  # Will be inferred from conversation
                    participants={message.from_user.id},
                    created_at=datetime.now(),
                    last_message_at=datetime.now(),
                    context_scope=ContextScope.THREAD_LOCAL,
                )

            thread = group_context.threads[thread_id]
            thread.participants.add(message.from_user.id)
            thread.last_message_at = datetime.now()
            thread.message_count += 1

            return thread

        # Check for ongoing threads based on message timing and participants
        recent_cutoff = datetime.now() - timedelta(minutes=10)
        for thread in group_context.threads.values():
            if (
                thread.last_message_at > recent_cutoff
                and message.from_user.id in thread.participants
            ):
                thread.last_message_at = datetime.now()
                thread.message_count += 1
                return thread

        return None

    async def _build_enhanced_context(
        self,
        group_context: GroupConversationContext,
        thread: Optional[GroupThread],
        message: Message,
        message_text: str,
    ) -> str:
        """Build enhanced context for AI processing."""
        context_parts = []

        # Group information
        context_parts.append(f"📍 Group: {group_context.group_name}")
        context_parts.append(
            f"👥 Active Participants: {len(group_context.participants)}"
        )

        # Thread context if available
        if thread:
            context_parts.append(f"🧵 Thread: {thread.topic or 'Conversation thread'}")
            context_parts.append(f"💬 Thread participants: {len(thread.participants)}")

        # Shared topics and memory
        if group_context.active_topics:
            topics = ", ".join(group_context.active_topics[:3])
            context_parts.append(f"🔖 Active topics: {topics}")

        # Recent conversation context
        if group_context.recent_messages:
            context_parts.append("\n📚 Recent conversation context:")
            for msg in group_context.recent_messages[-3:]:
                user_name = msg.get("user_name", "Unknown")
                content = msg.get("content", "")[:100]
                context_parts.append(f"  {user_name}: {content}")

        # Shared memory highlights
        if group_context.shared_memory:
            context_parts.append("\n🧠 Shared group knowledge:")
            for key, value in list(group_context.shared_memory.items())[:3]:
                context_parts.append(f"  {key}: {str(value)[:80]}")

        return "\n".join(context_parts)

    async def _process_group_message(
        self,
        message_text: str,
        enhanced_context: str,
        group_context: GroupConversationContext,
        thread: Optional[GroupThread],
    ) -> str:
        """Process message with group-aware AI context."""
        # Enhance the message with group context
        scope = thread.context_scope.value if thread else "group_shared"

        enhanced_message = f"""
{enhanced_context}

📝 Current message: {message_text}
🎯 Context scope: {scope}

Please respond considering the group conversation context and shared knowledge.
"""

        return enhanced_message

    async def _update_shared_memory(
        self,
        group_context: GroupConversationContext,
        thread: Optional[GroupThread],
        message: Message,
        processed_message: str,
    ):
        """Update shared conversation memory."""
        # Add to recent messages
        message_data = {
            "user_id": message.from_user.id,
            "user_name": message.from_user.full_name or message.from_user.first_name,
            "content": message.text or "",
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread.thread_id if thread else None,
        }

        group_context.recent_messages.append(message_data)

        # Maintain recent messages limit
        if len(group_context.recent_messages) > self.max_recent_messages:
            group_context.recent_messages = group_context.recent_messages[
                -self.max_recent_messages :
            ]

        # Extract and update topics (simple keyword extraction)
        await self._extract_and_update_topics(group_context, message.text or "")

        # Update group context timestamp
        group_context.updated_at = datetime.now()

    async def _extract_and_update_topics(
        self, group_context: GroupConversationContext, text: str
    ):
        """Extract and update active topics from conversation."""
        # Simple topic extraction (can be enhanced with NLP)
        words = text.lower().split()

        # Filter for potential topics (words longer than 3 chars, not common words)
        common_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "its",
            "may",
            "new",
            "now",
            "old",
            "see",
            "two",
            "who",
            "boy",
            "did",
            "man",
            "end",
            "few",
            "got",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
        }

        potential_topics = [
            word
            for word in words
            if len(word) > 3 and word not in common_words and word.isalpha()
        ]

        # Update active topics
        for topic in potential_topics[:5]:  # Limit new topics
            if topic not in group_context.active_topics:
                group_context.active_topics.append(topic)

        # Maintain topic list size
        if len(group_context.active_topics) > 20:
            group_context.active_topics = group_context.active_topics[-20:]

    async def _update_participant_activity(
        self, group_context: GroupConversationContext, user: User, message: Message
    ):
        """Update participant activity and statistics."""
        if user.id in group_context.participants:
            participant = group_context.participants[user.id]
            participant.last_active = datetime.now()
            participant.message_count += 1
            participant.is_active = True

    async def _get_group_settings(self, group_id: int) -> Dict[str, Any]:
        """Get group-specific settings."""
        try:
            # Try to load from database
            group_data = await self.user_data_manager.get_user_data(group_id)
            return group_data.get(
                "group_settings",
                {
                    "shared_memory_enabled": True,
                    "thread_tracking_enabled": True,
                    "context_scope": "group_shared",
                    "max_context_messages": 10,
                    "ai_participation_level": "moderate",
                },
            )
        except Exception as e:
            self.logger.error(f"Error loading group settings for {group_id}: {e}")
            return {
                "shared_memory_enabled": True,
                "thread_tracking_enabled": True,
                "context_scope": "group_shared",
                "max_context_messages": 10,
                "ai_participation_level": "moderate",
            }

    async def get_group_analytics(self, group_id: int) -> Dict[str, Any]:
        """Get analytics for a group."""
        if group_id not in self.group_contexts:
            return {}

        group_context = self.group_contexts[group_id]

        # Calculate statistics
        total_messages = sum(
            p.message_count for p in group_context.participants.values()
        )
        active_participants = sum(
            1 for p in group_context.participants.values() if p.is_active
        )

        return {
            "group_name": group_context.group_name,
            "total_participants": len(group_context.participants),
            "active_participants": active_participants,
            "total_messages": total_messages,
            "active_topics": group_context.active_topics[:10],
            "active_threads": len(
                [t for t in group_context.threads.values() if t.is_active]
            ),
            "shared_memory_items": len(group_context.shared_memory),
            "created_at": group_context.created_at.isoformat(),
            "last_activity": group_context.updated_at.isoformat(),
        }

    async def cleanup_inactive_threads(self, group_id: int):
        """Clean up inactive threads to maintain performance."""
        if group_id not in self.group_contexts:
            return

        group_context = self.group_contexts[group_id]
        cutoff_time = datetime.now() - timedelta(hours=self.thread_timeout_hours)

        inactive_threads = [
            thread_id
            for thread_id, thread in group_context.threads.items()
            if thread.last_message_at < cutoff_time
        ]

        for thread_id in inactive_threads:
            group_context.threads[thread_id].is_active = False
            self.logger.info(f"Deactivated inactive thread {thread_id}")

    async def set_group_context_scope(self, group_id: int, scope: ContextScope) -> bool:
        """Set the context scope for a group."""
        try:
            if group_id in self.group_contexts:
                self.group_contexts[group_id].settings["context_scope"] = scope.value

                # Save to database
                await self.user_data_manager.update_user_data(
                    group_id, {"group_settings.context_scope": scope.value}
                )

                self.logger.info(
                    f"Updated context scope for group {group_id} to {scope.value}"
                )
                return True
        except Exception as e:
            self.logger.error(f"Error setting context scope for group {group_id}: {e}")

        return False

    def get_active_groups(self) -> List[int]:
        """Get list of active group IDs."""
        return list(self.group_contexts.keys())

    async def export_group_context(self, group_id: int) -> Optional[Dict[str, Any]]:
        """Export group context for backup or analysis."""
        if group_id not in self.group_contexts:
            return None

        group_context = self.group_contexts[group_id]

        return {
            "group_id": group_context.group_id,
            "group_name": group_context.group_name,
            "participants": {
                str(uid): {
                    "username": p.username,
                    "full_name": p.full_name,
                    "role": p.role.value,
                    "message_count": p.message_count,
                    "join_date": p.join_date.isoformat(),
                    "last_active": p.last_active.isoformat(),
                }
                for uid, p in group_context.participants.items()
            },
            "active_topics": group_context.active_topics,
            "shared_memory": group_context.shared_memory,
            "threads": {
                tid: {
                    "topic": t.topic,
                    "participant_count": len(t.participants),
                    "message_count": t.message_count,
                    "created_at": t.created_at.isoformat(),
                    "last_message_at": t.last_message_at.isoformat(),
                }
                for tid, t in group_context.threads.items()
            },
            "created_at": group_context.created_at.isoformat(),
            "updated_at": group_context.updated_at.isoformat(),
        }
