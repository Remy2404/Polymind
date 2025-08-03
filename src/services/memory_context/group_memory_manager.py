"""
🧠 Advanced Group Memory & Conversation Intelligence System
Provides shared memory for team collaboration with contextual awareness
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import hashlib
import sys
import os

# MongoDB imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from database.connection import get_database
from pymongo.database import Database

from ..user_data_manager import UserDataManager
from ..utils import Utils

logger = logging.getLogger(__name__)


@dataclass
class GroupConversationContext:
    """Enhanced context for group conversations"""

    group_id: int
    group_title: str
    active_participants: List[int]
    conversation_topic: Optional[str]
    current_discussion_thread: str
    shared_context: Dict[str, Any]
    group_preferences: Dict[str, Any]
    last_activity: datetime
    conversation_summary: str
    key_decisions: List[str]
    action_items: List[Dict[str, Any]]
    expertise_map: Dict[int, List[str]]  # user_id -> areas of expertise


@dataclass
class ConversationThread:
    """Individual conversation thread within a group"""

    thread_id: str
    topic: str
    participants: Set[int]
    messages: deque
    created_at: datetime
    last_active: datetime
    status: str  # 'active', 'paused', 'resolved'
    priority: str  # 'low', 'medium', 'high', 'urgent'


class GroupMemoryManager:
    """
    🧠 Advanced Group Memory System

    Features:
    - Shared group memory with intelligent context switching
    - Conversation thread management
    - Participant expertise tracking
    - Smart notification system
    - Decision tracking and action items"""

    def __init__(self, db=None):
        # MongoDB initialization
        if db is None:
            try:
                self.db, self.client = get_database()
                if self.db is not None:
                    logger.info("Connected to MongoDB for group memory management")
                    # Initialize collections
                    self.group_contexts_collection = self.db.group_contexts
                    self.conversation_threads_collection = self.db.conversation_threads
                    self.group_analytics_collection = self.db.group_analytics

                    # Ensure indexes for better performance
                    self._ensure_indexes()
                else:
                    logger.warning(
                        "MongoDB connection failed, group memory manager will not persist data"
                    )
                    self.client = None
                    self.group_contexts_collection = None
                    self.conversation_threads_collection = None
                    self.group_analytics_collection = None
            except Exception as e:
                logger.error(f"Error connecting to MongoDB: {e}")
                self.db = None
                self.client = None
                self.group_contexts_collection = None
                self.conversation_threads_collection = None
                self.group_analytics_collection = None
        else:
            self.db = db
            self.client = None  # Will be managed externally
            if db is not None:
                self.group_contexts_collection = db.group_contexts
                self.conversation_threads_collection = db.conversation_threads
                self.group_analytics_collection = db.group_analytics
                self._ensure_indexes()
            else:
                self.group_contexts_collection = None
                self.conversation_threads_collection = None
                self.group_analytics_collection = None

        # Group conversation contexts
        self.group_contexts: Dict[int, GroupConversationContext] = {}

        # Active conversation threads
        self.conversation_threads: Dict[str, ConversationThread] = {}

        # Group analytics
        self.group_analytics = defaultdict(
            lambda: {
                "message_count": 0,
                "active_hours": set(),
                "popular_topics": defaultdict(int),
                "collaboration_score": 0.0,
            }
        )

        # Smart notification queue
        self.notification_queue = deque()

        # Load existing data
        self._load_group_data()

        logger.info("🧠 Group Memory Manager initialized with advanced intelligence")

    async def initialize_group_context(
        self, group_id: int, group_title: str, participants: List[int]
    ) -> GroupConversationContext:
        """Initialize or update group conversation context"""
        try:
            if group_id not in self.group_contexts:
                context = GroupConversationContext(
                    group_id=group_id,
                    group_title=group_title,
                    active_participants=participants,
                    conversation_topic=None,
                    current_discussion_thread="general",
                    shared_context={},
                    group_preferences={
                        "memory_retention_days": 30,
                        "auto_summarize": True,
                        "smart_notifications": True,
                        "expertise_tracking": True,
                    },
                    last_activity=datetime.now(),
                    conversation_summary="",
                    key_decisions=[],
                    action_items=[],
                    expertise_map=defaultdict(list),
                )
                self.group_contexts[group_id] = context
                logger.info(f"🏗️ Initialized new group context for {group_title}")
            else:
                # Update existing context
                context = self.group_contexts[group_id]
                context.active_participants = list(
                    set(context.active_participants + participants)
                )
                context.last_activity = datetime.now()

            await self._save_group_data()
            return self.group_contexts[group_id]

        except Exception as e:
            logger.error(f"❌ Error initializing group context: {e}")
            raise

    async def add_group_message(
        self,
        group_id: int,
        user_id: int,
        message: str,
        message_type: str = "text",
        metadata: Dict = None,
    ) -> Dict[str, Any]:
        """Add message to group memory with intelligent processing"""
        try:
            if group_id not in self.group_contexts:
                logger.warning(f"⚠️ Group {group_id} not initialized")
                return {}

            context = self.group_contexts[group_id]
            timestamp = datetime.now()

            # Create enhanced message object
            enhanced_message = {
                "user_id": user_id,
                "message": message,
                "type": message_type,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata or {},
                "thread_id": context.current_discussion_thread,
                "sentiment": await self._analyze_sentiment(message),
                "topics": await self._extract_topics(message),
                "mentions": await self._extract_mentions(message),
                "action_items": await self._extract_action_items(message),
            }

            # Update group analytics
            self._update_analytics(group_id, enhanced_message)

            # Update conversation thread
            await self._update_conversation_thread(group_id, enhanced_message)

            # Check for smart notifications
            await self._process_smart_notifications(group_id, enhanced_message)

            # Update expertise map
            await self._update_expertise_map(group_id, user_id, message)

            # Save context
            context.last_activity = timestamp
            await self._save_group_data()

            return enhanced_message

        except Exception as e:
            logger.error(f"❌ Error adding group message: {e}")
            return {}

    async def get_group_context(
        self, group_id: int, user_id: int, context_length: int = 20
    ) -> Dict[str, Any]:
        """Get intelligent group context for response generation"""
        try:
            if group_id not in self.group_contexts:
                return {"context": [], "summary": "No group context available"}

            context = self.group_contexts[group_id]

            # Get current thread messages
            thread_id = context.current_discussion_thread
            thread_messages = []

            if thread_id in self.conversation_threads:
                thread = self.conversation_threads[thread_id]
                thread_messages = list(thread.messages)[-context_length:]

            # Generate intelligent summary
            summary = await self._generate_context_summary(group_id, user_id)

            # Get relevant expertise
            relevant_experts = await self._get_relevant_experts(
                group_id, thread_messages
            )

            # Prepare enhanced context
            enhanced_context = {
                "group_info": {
                    "title": context.group_title,
                    "participants": len(context.active_participants),
                    "current_topic": context.conversation_topic,
                    "thread": context.current_discussion_thread,
                },
                "conversation_context": thread_messages,
                "summary": summary,
                "recent_decisions": context.key_decisions[-5:],
                "pending_actions": context.action_items,
                "relevant_experts": relevant_experts,
                "group_preferences": context.group_preferences,
                "analytics": self.group_analytics[group_id],
            }

            return enhanced_context

        except Exception as e:
            logger.error(f"❌ Error getting group context: {e}")
            return {"context": [], "summary": "Error retrieving context"}

    async def create_conversation_thread(
        self, group_id: int, topic: str, initiator_id: int, priority: str = "medium"
    ) -> str:
        """Create a new conversation thread"""
        try:
            thread_id = f"{group_id}_{datetime.now().timestamp()}_{hashlib.md5(topic.encode()).hexdigest()[:8]}"

            thread = ConversationThread(
                thread_id=thread_id,
                topic=topic,
                participants={initiator_id},
                messages=deque(maxlen=100),
                created_at=datetime.now(),
                last_active=datetime.now(),
                status="active",
                priority=priority,
            )

            self.conversation_threads[thread_id] = thread

            # Update group context
            if group_id in self.group_contexts:
                self.group_contexts[group_id].current_discussion_thread = thread_id
                self.group_contexts[group_id].conversation_topic = topic

            await self._save_group_data()
            logger.info(f"🧵 Created new conversation thread: {topic}")

            return thread_id

        except Exception as e:
            logger.error(f"❌ Error creating conversation thread: {e}")
            return ""

    async def switch_conversation_thread(self, group_id: int, thread_id: str) -> bool:
        """Switch to a different conversation thread"""
        try:
            if (
                thread_id in self.conversation_threads
                and group_id in self.group_contexts
            ):
                context = self.group_contexts[group_id]
                context.current_discussion_thread = thread_id
                context.conversation_topic = self.conversation_threads[thread_id].topic

                await self._save_group_data()
                logger.info(f"🔄 Switched to thread: {thread_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error switching thread: {e}")
            return False

    async def get_group_summary(
        self, group_id: int, timeframe_hours: int = 24
    ) -> Dict[str, Any]:
        """Generate comprehensive group activity summary"""
        try:
            if group_id not in self.group_contexts:
                return {}

            context = self.group_contexts[group_id]
            cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)

            # Gather recent activity
            recent_threads = []
            total_messages = 0

            for thread_id, thread in self.conversation_threads.items():
                if (
                    thread_id.startswith(str(group_id))
                    and thread.last_active > cutoff_time
                ):
                    recent_threads.append(
                        {
                            "id": thread_id,
                            "topic": thread.topic,
                            "messages": len(thread.messages),
                            "participants": len(thread.participants),
                            "status": thread.status,
                            "priority": thread.priority,
                        }
                    )
                    total_messages += len(thread.messages)

            # Generate summary
            summary = {
                "group_info": {
                    "title": context.group_title,
                    "participants": len(context.active_participants),
                    "last_activity": context.last_activity.isoformat(),
                },
                "activity_stats": {
                    "total_messages": total_messages,
                    "active_threads": len(recent_threads),
                    "timeframe_hours": timeframe_hours,
                },
                "conversation_threads": recent_threads,
                "key_decisions": context.key_decisions[-10:],
                "pending_actions": context.action_items,
                "top_contributors": await self._get_top_contributors(
                    group_id, timeframe_hours
                ),
                "trending_topics": await self._get_trending_topics(
                    group_id, timeframe_hours
                ),
                "collaboration_score": self.group_analytics[group_id][
                    "collaboration_score"
                ],
            }

            return summary

        except Exception as e:
            logger.error(f"❌ Error generating group summary: {e}")
            return {}

    async def add_decision(
        self, group_id: int, decision: str, participants: List[int]
    ) -> bool:
        """Record a key decision made by the group"""
        try:
            if group_id in self.group_contexts:
                context = self.group_contexts[group_id]
                decision_record = {
                    "decision": decision,
                    "timestamp": datetime.now().isoformat(),
                    "participants": participants,
                    "thread_id": context.current_discussion_thread,
                }
                context.key_decisions.append(decision_record)

                await self._save_group_data()
                logger.info(f"📝 Recorded group decision: {decision}")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error adding decision: {e}")
            return False

    async def add_action_item(
        self,
        group_id: int,
        action: str,
        assignee_id: int,
        due_date: Optional[datetime] = None,
    ) -> bool:
        """Add an action item to the group"""
        try:
            if group_id in self.group_contexts:
                context = self.group_contexts[group_id]
                action_item = {
                    "action": action,
                    "assignee_id": assignee_id,
                    "created_at": datetime.now().isoformat(),
                    "due_date": due_date.isoformat() if due_date else None,
                    "status": "pending",
                    "thread_id": context.current_discussion_thread,
                }
                context.action_items.append(action_item)

                # Add to notification queue
                await self._queue_action_notification(group_id, action_item)

                await self._save_group_data()
                logger.info(f"✅ Added action item: {action}")
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error adding action item: {e}")
            return False

    async def _analyze_sentiment(self, message: str) -> str:
        """Analyze message sentiment"""
        # Simplified sentiment analysis
        positive_words = [
            "great",
            "good",
            "excellent",
            "awesome",
            "love",
            "like",
            "perfect",
        ]
        negative_words = [
            "bad",
            "terrible",
            "hate",
            "dislike",
            "wrong",
            "problem",
            "issue",
        ]

        message_lower = message.lower()
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    async def _extract_topics(self, message: str) -> List[str]:
        """Extract topics from message"""
        # Simple topic extraction based on keywords
        topics = []
        message_lower = message.lower()

        topic_keywords = {
            "development": ["code", "programming", "development", "bug", "feature"],
            "meeting": ["meeting", "call", "discuss", "agenda"],
            "deadline": ["deadline", "due", "urgent", "asap"],
            "decision": ["decide", "choose", "decision", "vote"],
            "question": ["?", "question", "how", "what", "why", "when"],
        }

        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(topic)

        return topics

    async def _extract_mentions(self, message: str) -> List[str]:
        """Extract @mentions from message"""
        import re

        mentions = re.findall(r"@\w+", message)
        return mentions

    async def _extract_action_items(self, message: str) -> List[str]:
        """Extract potential action items from message"""
        action_patterns = [
            r"(?:need to|should|must|will) ([^.!?]+)",
            r"(?:todo|to do|action|task): ([^.!?]+)",
            r"(?:reminder|remind): ([^.!?]+)",
        ]

        actions = []
        for pattern in action_patterns:
            import re

            matches = re.findall(pattern, message.lower())
            actions.extend(matches)

        return actions

    def _update_analytics(self, group_id: int, message: Dict[str, Any]):
        """Update group analytics"""
        analytics = self.group_analytics[group_id]
        analytics["message_count"] += 1

        # Track active hours
        hour = datetime.now().hour
        analytics["active_hours"].add(hour)

        # Track popular topics
        for topic in message.get("topics", []):
            analytics["popular_topics"][topic] += 1

        # Update collaboration score
        analytics["collaboration_score"] = min(
            100.0, analytics["collaboration_score"] + 0.1
        )

    async def _update_conversation_thread(self, group_id: int, message: Dict[str, Any]):
        """Update the current conversation thread"""
        context = self.group_contexts[group_id]
        thread_id = context.current_discussion_thread

        if thread_id in self.conversation_threads:
            thread = self.conversation_threads[thread_id]
            thread.messages.append(message)
            thread.participants.add(message["user_id"])
            thread.last_active = datetime.now()

    async def _process_smart_notifications(
        self, group_id: int, message: Dict[str, Any]
    ):
        """Process smart notifications based on message content"""
        # Check for urgent keywords
        urgent_keywords = ["urgent", "asap", "emergency", "critical"]
        message_text = message["message"].lower()

        if any(keyword in message_text for keyword in urgent_keywords):
            notification = {
                "type": "urgent_message",
                "group_id": group_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
            self.notification_queue.append(notification)

    async def _update_expertise_map(self, group_id: int, user_id: int, message: str):
        """Update user expertise based on message content"""
        context = self.group_contexts[group_id]

        # Simple expertise detection
        expertise_keywords = {
            "programming": ["code", "programming", "python", "javascript", "api"],
            "design": ["design", "ui", "ux", "interface", "mockup"],
            "project_management": ["deadline", "schedule", "planning", "milestone"],
            "marketing": ["marketing", "promotion", "campaign", "audience"],
            "data_analysis": ["data", "analysis", "statistics", "metrics"],
        }

        message_lower = message.lower()
        for expertise, keywords in expertise_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                if expertise not in context.expertise_map[user_id]:
                    context.expertise_map[user_id].append(expertise)

    async def _generate_context_summary(self, group_id: int, user_id: int) -> str:
        """Generate intelligent context summary"""
        context = self.group_contexts[group_id]

        # Simple summary generation
        summary_parts = []

        if context.conversation_topic:
            summary_parts.append(f"Currently discussing: {context.conversation_topic}")

        if context.key_decisions:
            latest_decision = context.key_decisions[-1]
            summary_parts.append(
                f"Latest decision: {latest_decision.get('decision', 'N/A')}"
            )

        if context.action_items:
            pending_actions = [
                item for item in context.action_items if item.get("status") == "pending"
            ]
            if pending_actions:
                summary_parts.append(f"Pending actions: {len(pending_actions)}")

        return (
            " | ".join(summary_parts) if summary_parts else "Active group conversation"
        )

    async def _get_relevant_experts(
        self, group_id: int, messages: List[Dict]
    ) -> List[Dict]:
        """Get relevant experts for current discussion"""
        context = self.group_contexts[group_id]
        relevant_experts = []

        # Extract topics from recent messages
        recent_topics = set()
        for message in messages[-5:]:  # Last 5 messages
            recent_topics.update(message.get("topics", []))

        # Find users with relevant expertise
        for user_id, expertise_list in context.expertise_map.items():
            relevance_score = len(set(expertise_list) & recent_topics)
            if relevance_score > 0:
                relevant_experts.append(
                    {
                        "user_id": user_id,
                        "expertise": expertise_list,
                        "relevance_score": relevance_score,
                    }
                )

        return sorted(
            relevant_experts, key=lambda x: x["relevance_score"], reverse=True
        )

    async def _get_top_contributors(
        self, group_id: int, timeframe_hours: int
    ) -> List[Dict]:
        """Get top contributors in the timeframe"""
        # Simplified implementation
        contributor_stats = defaultdict(int)

        for thread_id, thread in self.conversation_threads.items():
            if thread_id.startswith(str(group_id)):
                for participant in thread.participants:
                    contributor_stats[participant] += 1

        top_contributors = []
        for user_id, count in contributor_stats.items():
            top_contributors.append({"user_id": user_id, "message_count": count})

        return sorted(top_contributors, key=lambda x: x["message_count"], reverse=True)[
            :5
        ]

    async def _get_trending_topics(
        self, group_id: int, timeframe_hours: int
    ) -> List[Dict]:
        """Get trending topics in the timeframe"""
        analytics = self.group_analytics[group_id]
        trending = []

        for topic, count in analytics["popular_topics"].items():
            trending.append({"topic": topic, "mention_count": count})

        return sorted(trending, key=lambda x: x["mention_count"], reverse=True)[:5]

    async def _queue_action_notification(self, group_id: int, action_item: Dict):
        """Queue notification for action item"""
        notification = {
            "type": "action_item",
            "group_id": group_id,
            "action_item": action_item,
            "timestamp": datetime.now().isoformat(),
        }
        self.notification_queue.append(notification)

    def _load_group_data(self):
        """Load group data from MongoDB"""
        try:
            if self.group_contexts_collection is None:
                logger.warning("No MongoDB collection available for loading group data")
                return

            # Load group contexts
            contexts = self.group_contexts_collection.find()
            for context_doc in contexts:
                group_id = context_doc["group_id"]
                # Convert document back to GroupConversationContext
                context_data = context_doc.copy()
                context_data.pop("_id", None)
                # Convert datetime strings back to datetime objects
                if "last_activity" in context_data and isinstance(
                    context_data["last_activity"], str
                ):
                    context_data["last_activity"] = datetime.fromisoformat(
                        context_data["last_activity"]
                    )

                self.group_contexts[group_id] = GroupConversationContext(**context_data)
            # Load conversation threads
            if self.conversation_threads_collection is not None:
                threads = self.conversation_threads_collection.find()
                for thread_doc in threads:
                    thread_id = thread_doc["thread_id"]
                    # Convert document back to ConversationThread
                    thread_data = thread_doc.copy()
                    thread_data.pop("_id", None)
                    # Convert datetime strings and sets
                    if "created_at" in thread_data and isinstance(
                        thread_data["created_at"], str
                    ):
                        thread_data["created_at"] = datetime.fromisoformat(
                            thread_data["created_at"]
                        )
                    if "last_active" in thread_data and isinstance(
                        thread_data["last_active"], str
                    ):
                        thread_data["last_active"] = datetime.fromisoformat(
                            thread_data["last_active"]
                        )
                    if "participants" in thread_data and isinstance(
                        thread_data["participants"], list
                    ):
                        thread_data["participants"] = set(thread_data["participants"])
                    if "messages" in thread_data and isinstance(
                        thread_data["messages"], list
                    ):
                        thread_data["messages"] = deque(thread_data["messages"])

                    self.conversation_threads[thread_id] = ConversationThread(
                        **thread_data
                    )
            # Load analytics
            if self.group_analytics_collection is not None:
                analytics = self.group_analytics_collection.find()
                for analytics_doc in analytics:
                    group_id = analytics_doc["group_id"]
                    analytics_data = analytics_doc.copy()
                    analytics_data.pop("_id", None)
                    analytics_data.pop("group_id", None)
                    self.group_analytics[group_id] = analytics_data

            logger.info("📁 Loaded group data from MongoDB")

        except Exception as e:
            logger.error(f"❌ Error loading group data: {e}")

    async def _save_group_data(self):
        """Save group data to MongoDB"""
        try:
            if self.group_contexts_collection is None:
                logger.warning("No MongoDB collection available for saving group data")
                return

            # Save group contexts
            for group_id, context in self.group_contexts.items():
                context_data = asdict(context)
                # Convert datetime objects to strings for MongoDB storage
                if "last_activity" in context_data and isinstance(
                    context_data["last_activity"], datetime
                ):
                    context_data["last_activity"] = context_data[
                        "last_activity"
                    ].isoformat()

                self.group_contexts_collection.update_one(
                    {"group_id": group_id}, {"$set": context_data}, upsert=True
                )
            # Save conversation threads
            if self.conversation_threads_collection is not None:
                for thread_id, thread in self.conversation_threads.items():
                    thread_data = asdict(thread)
                    # Convert datetime objects and collections for MongoDB storage
                    if "created_at" in thread_data and isinstance(
                        thread_data["created_at"], datetime
                    ):
                        thread_data["created_at"] = thread_data[
                            "created_at"
                        ].isoformat()
                    if "last_active" in thread_data and isinstance(
                        thread_data["last_active"], datetime
                    ):
                        thread_data["last_active"] = thread_data[
                            "last_active"
                        ].isoformat()
                    if "participants" in thread_data and isinstance(
                        thread_data["participants"], set
                    ):
                        thread_data["participants"] = list(thread_data["participants"])
                    if "messages" in thread_data and isinstance(
                        thread_data["messages"], deque
                    ):
                        thread_data["messages"] = list(thread_data["messages"])

                    self.conversation_threads_collection.update_one(
                        {"thread_id": thread_id}, {"$set": thread_data}, upsert=True
                    )
            # Save analytics
            if self.group_analytics_collection is not None:
                for group_id, analytics_data in self.group_analytics.items():
                    # Convert sets to lists for MongoDB storage
                    analytics_data_copy = analytics_data.copy()
                    if "active_hours" in analytics_data_copy and isinstance(
                        analytics_data_copy["active_hours"], set
                    ):
                        analytics_data_copy["active_hours"] = list(
                            analytics_data_copy["active_hours"]
                        )

                    self.group_analytics_collection.update_one(
                        {"group_id": group_id},
                        {"$set": {"group_id": group_id, **analytics_data_copy}},
                        upsert=True,
                    )

            logger.debug("💾 Saved group data to MongoDB")

        except Exception as e:
            logger.error(f"❌ Error saving group data: {e}")

    async def cleanup_old_data(self, days_old: int = 30):
        """Clean up old conversation data from MongoDB"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_iso = cutoff_date.isoformat()
            # Clean old threads from MongoDB
            if self.conversation_threads_collection is not None:
                result = self.conversation_threads_collection.delete_many(
                    {"last_active": {"$lt": cutoff_iso}}
                )
                logger.info(
                    f"🧹 Removed {result.deleted_count} old conversation threads from MongoDB"
                )

            # Clean old threads from memory
            threads_to_remove = []
            for thread_id, thread in self.conversation_threads.items():
                if thread.last_active < cutoff_date:
                    threads_to_remove.append(thread_id)

            for thread_id in threads_to_remove:
                del self.conversation_threads[thread_id]

            # Clean old notifications
            while (
                self.notification_queue
                and datetime.fromisoformat(self.notification_queue[0]["timestamp"])
                < cutoff_date
            ):
                self.notification_queue.popleft()

            await self._save_group_data()
            logger.info(
                f"🧹 Cleaned up {len(threads_to_remove)} old conversation threads from memory"
            )

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

    def _ensure_indexes(self):
        """Ensure database indexes are created for better performance"""
        try:
            if self.group_contexts_collection is not None:
                # Index for group_id lookups
                self.group_contexts_collection.create_index("group_id", unique=True)
                self.group_contexts_collection.create_index(
                    [("group_id", 1), ("last_activity", -1)]
                )

            if self.conversation_threads_collection is not None:
                # Indexes for conversation threads
                self.conversation_threads_collection.create_index(
                    "thread_id", unique=True
                )
                self.conversation_threads_collection.create_index("group_id")
                self.conversation_threads_collection.create_index(
                    [("group_id", 1), ("last_active", -1)]
                )

            if self.group_analytics_collection is not None:
                # Index for analytics lookups
                self.group_analytics_collection.create_index("group_id", unique=True)

            logger.info("Group memory management database indexes ensured")

        except Exception as e:
            logger.error(f"Error creating group memory management indexes: {e}")


# Global instance with MongoDB connection
try:
    db, client = get_database()
    group_memory_manager = GroupMemoryManager(db=db)
except Exception as e:
    logger.warning(f"Failed to initialize group memory manager with MongoDB: {e}")
    group_memory_manager = GroupMemoryManager()
