"""
Group Conversation Manager - Advanced Group Chat Intelligence System
Handles multi-user conversations with shared memory and intelligent context management
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid

from src.database.connection import DatabaseConnection
from src.utils.config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GroupParticipant:
    """Represents a participant in a group conversation"""

    user_id: int
    username: str
    first_name: str
    last_name: Optional[str] = None
    role: str = "member"  # member, admin, moderator
    join_date: datetime = None
    last_active: datetime = None
    message_count: int = 0
    is_active: bool = True
    preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.join_date is None:
            self.join_date = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()
        if self.preferences is None:
            self.preferences = {}


@dataclass
class GroupMessage:
    """Enhanced message structure for group conversations"""

    message_id: str
    user_id: int
    username: str
    content: str
    timestamp: datetime
    message_type: str = "text"  # text, image, document, command, system
    reply_to: Optional[str] = None
    mentions: List[int] = None
    hashtags: List[str] = None
    sentiment: Optional[str] = None
    topic: Optional[str] = None
    importance_score: float = 0.5
    context_relevance: float = 0.5
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.mentions is None:
            self.mentions = []
        if self.hashtags is None:
            self.hashtags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GroupConversationState:
    """Represents the current state of a group conversation"""

    group_id: int
    group_name: str
    group_type: str  # private, public, supergroup, channel
    participants: Dict[int, GroupParticipant]
    messages: deque
    current_topic: Optional[str] = None
    active_threads: List[str] = None
    group_settings: Dict[str, Any] = None
    conversation_summary: str = ""
    last_activity: datetime = None
    created_at: datetime = None
    model_preferences: Dict[str, Any] = None
    auto_responses: bool = True
    smart_notifications: bool = True

    def __post_init__(self):
        if self.active_threads is None:
            self.active_threads = []
        if self.group_settings is None:
            self.group_settings = {
                "max_context_messages": 50,
                "auto_summarize": True,
                "smart_threading": True,
                "mention_notifications": True,
                "topic_tracking": True,
            }
        if self.last_activity is None:
            self.last_activity = datetime.now()
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.model_preferences is None:
            self.model_preferences = {
                "default_model": "gemini",
                "context_window": "adaptive",
                "response_style": "collaborative",
            }


class GroupConversationManager:
    """Advanced Group Conversation Manager with Intelligence Features"""

    def __init__(self):
        self.config = get_config()
        self.db_connection = DatabaseConnection()
        self.group_states: Dict[int, GroupConversationState] = {}
        self.topic_cache: Dict[int, List[str]] = defaultdict(list)
        self.mention_tracker: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.conversation_threads: Dict[str, List[str]] = defaultdict(list)

        # Intelligence features
        self.sentiment_patterns = {
            "positive": ["great", "awesome", "love", "perfect", "excellent", "amazing"],
            "negative": [
                "bad",
                "terrible",
                "hate",
                "awful",
                "disappointing",
                "frustrated",
            ],
            "question": ["?", "how", "what", "when", "where", "why", "which"],
            "urgent": ["urgent", "asap", "emergency", "important", "critical", "help"],
        }

        self.topic_keywords = {
            "technical": [
                "code",
                "bug",
                "feature",
                "development",
                "programming",
                "api",
            ],
            "planning": [
                "meeting",
                "schedule",
                "deadline",
                "plan",
                "timeline",
                "roadmap",
            ],
            "general": [
                "chat",
                "discussion",
                "talk",
                "conversation",
                "idea",
                "thought",
            ],
            "support": [
                "help",
                "issue",
                "problem",
                "question",
                "assistance",
                "trouble",
            ],
        }

    async def initialize_group(
        self, group_id: int, group_name: str, group_type: str = "group"
    ) -> GroupConversationState:
        """Initialize a new group conversation state"""
        try:
            if group_id not in self.group_states:
                self.group_states[group_id] = GroupConversationState(
                    group_id=group_id,
                    group_name=group_name,
                    group_type=group_type,
                    participants={},
                    messages=deque(maxlen=1000),  # Keep last 1000 messages in memory
                )

                # Create database entry for group
                await self._create_group_record(group_id, group_name, group_type)

                logger.info(
                    f"Initialized group conversation for {group_name} ({group_id})"
                )

            return self.group_states[group_id]

        except Exception as e:
            logger.error(f"Error initializing group {group_id}: {str(e)}")
            raise

    async def add_participant(
        self,
        group_id: int,
        user_id: int,
        username: str,
        first_name: str,
        last_name: str = None,
        role: str = "member",
    ) -> bool:
        """Add a participant to the group conversation"""
        try:
            if group_id not in self.group_states:
                logger.warning(f"Group {group_id} not initialized")
                return False

            participant = GroupParticipant(
                user_id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                role=role,
            )

            self.group_states[group_id].participants[user_id] = participant

            # Save to database
            await self._save_participant(group_id, participant)

            logger.info(f"Added participant {username} to group {group_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error adding participant {user_id} to group {group_id}: {str(e)}"
            )
            return False

    async def process_group_message(
        self,
        group_id: int,
        user_id: int,
        username: str,
        content: str,
        message_type: str = "text",
        reply_to: str = None,
    ) -> GroupMessage:
        """Process a new message in the group conversation"""
        try:
            if group_id not in self.group_states:
                logger.warning(f"Group {group_id} not found")
                return None

            group_state = self.group_states[group_id]

            # Create message object
            message = GroupMessage(
                message_id=str(uuid.uuid4()),
                user_id=user_id,
                username=username,
                content=content,
                timestamp=datetime.now(),
                message_type=message_type,
                reply_to=reply_to,
            )

            # Extract intelligence features
            await self._analyze_message(message, group_state)

            # Update participant activity
            if user_id in group_state.participants:
                participant = group_state.participants[user_id]
                participant.last_active = datetime.now()
                participant.message_count += 1

            # Add to conversation
            group_state.messages.append(message)
            group_state.last_activity = datetime.now()

            # Update conversation context
            await self._update_conversation_context(group_id, message)

            # Save to database
            await self._save_message(group_id, message)

            logger.info(f"Processed message from {username} in group {group_id}")
            return message

        except Exception as e:
            logger.error(f"Error processing message in group {group_id}: {str(e)}")
            return None

    async def _analyze_message(
        self, message: GroupMessage, group_state: GroupConversationState
    ):
        """Analyze message for intelligence features"""
        content_lower = message.content.lower()

        # Sentiment analysis
        for sentiment, keywords in self.sentiment_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                message.sentiment = sentiment
                break

        # Topic detection
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                message.topic = topic
                break

        # Extract mentions
        words = message.content.split()
        for word in words:
            if word.startswith("@"):
                username = word[1:]
                for user_id, participant in group_state.participants.items():
                    if participant.username == username:
                        message.mentions.append(user_id)
                        self.mention_tracker[message.user_id][user_id] += 1

        # Extract hashtags
        message.hashtags = [word[1:] for word in words if word.startswith("#")]

        # Calculate importance score
        importance_factors = [
            len(message.mentions) * 0.2,  # Mentions increase importance
            len(message.hashtags) * 0.1,  # Hashtags add structure
            (
                1.0 if message.sentiment == "urgent" else 0.0
            ),  # Urgent messages are important
            0.3 if message.reply_to else 0.0,  # Replies show engagement
            min(
                len(message.content) / 100, 0.3
            ),  # Longer messages might be more important
        ]
        message.importance_score = min(sum(importance_factors), 1.0)

    async def get_group_context(
        self, group_id: int, max_messages: int = 20
    ) -> Dict[str, Any]:
        """Get intelligent context for the group conversation"""
        try:
            if group_id not in self.group_states:
                return {"error": "Group not found"}

            group_state = self.group_states[group_id]

            # Get recent messages with high importance or relevance
            recent_messages = list(group_state.messages)[-max_messages:]

            # Sort by importance and recency
            important_messages = sorted(
                recent_messages,
                key=lambda m: (m.importance_score, m.timestamp),
                reverse=True,
            )[: max_messages // 2]

            # Get chronologically recent messages
            recent_chrono = recent_messages[-max_messages // 2 :]

            # Combine and deduplicate
            context_messages = []
            seen_ids = set()

            for msg in important_messages + recent_chrono:
                if msg.message_id not in seen_ids:
                    context_messages.append(
                        {
                            "user": msg.username,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "type": msg.message_type,
                            "sentiment": msg.sentiment,
                            "topic": msg.topic,
                            "importance": msg.importance_score,
                        }
                    )
                    seen_ids.add(msg.message_id)

            # Sort by timestamp for final context
            context_messages.sort(key=lambda m: m["timestamp"])

            # Generate conversation summary
            summary = await self._generate_conversation_summary(group_state)

            # Get active participants
            active_participants = [
                {
                    "username": p.username,
                    "role": p.role,
                    "message_count": p.message_count,
                    "last_active": p.last_active.isoformat(),
                }
                for p in group_state.participants.values()
                if p.is_active and p.last_active > datetime.now() - timedelta(hours=24)
            ]

            return {
                "group_id": group_id,
                "group_name": group_state.group_name,
                "messages": context_messages,
                "summary": summary,
                "current_topic": group_state.current_topic,
                "active_participants": active_participants,
                "participant_count": len(group_state.participants),
                "total_messages": len(group_state.messages),
                "last_activity": group_state.last_activity.isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting group context for {group_id}: {str(e)}")
            return {"error": str(e)}

    async def _generate_conversation_summary(
        self, group_state: GroupConversationState
    ) -> str:
        """Generate an intelligent summary of recent conversation"""
        try:
            if not group_state.messages:
                return "No recent conversation activity."

            # Get recent messages for summary
            recent_messages = list(group_state.messages)[-20:]

            # Analyze topics and sentiments
            topics = defaultdict(int)
            sentiments = defaultdict(int)
            participants = set()

            for msg in recent_messages:
                if msg.topic:
                    topics[msg.topic] += 1
                if msg.sentiment:
                    sentiments[msg.sentiment] += 1
                participants.add(msg.username)

            # Build summary
            summary_parts = []

            if topics:
                main_topic = max(topics, key=topics.get)
                summary_parts.append(f"Main discussion topic: {main_topic}")

            if sentiments:
                main_sentiment = max(sentiments, key=sentiments.get)
                summary_parts.append(f"Overall sentiment: {main_sentiment}")

            summary_parts.append(
                f"Active participants: {', '.join(list(participants)[:5])}"
            )

            if len(participants) > 5:
                summary_parts.append(f"and {len(participants) - 5} others")

            return ". ".join(summary_parts) + "."

        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return "Unable to generate conversation summary."

    async def _update_conversation_context(self, group_id: int, message: GroupMessage):
        """Update conversation context based on new message"""
        try:
            group_state = self.group_states[group_id]

            # Update current topic if message has high importance
            if message.importance_score > 0.7 and message.topic:
                group_state.current_topic = message.topic

            # Track conversation threads
            if message.reply_to:
                thread_id = f"{group_id}_{message.reply_to}"
                self.conversation_threads[thread_id].append(message.message_id)

                # Update active threads
                if thread_id not in group_state.active_threads:
                    group_state.active_threads.append(thread_id)

            # Clean up old threads (older than 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            group_state.active_threads = [
                thread
                for thread in group_state.active_threads
                if any(msg.timestamp > cutoff_time for msg in group_state.messages)
            ]

        except Exception as e:
            logger.error(f"Error updating conversation context: {str(e)}")

    async def get_participant_insights(
        self, group_id: int, user_id: int
    ) -> Dict[str, Any]:
        """Get insights about a specific participant"""
        try:
            if group_id not in self.group_states:
                return {"error": "Group not found"}

            group_state = self.group_states[group_id]

            if user_id not in group_state.participants:
                return {"error": "Participant not found"}

            participant = group_state.participants[user_id]

            # Analyze participant's messages
            user_messages = [
                msg for msg in group_state.messages if msg.user_id == user_id
            ]

            topics = defaultdict(int)
            sentiments = defaultdict(int)
            mentions_given = self.mention_tracker[user_id]
            mentions_received = sum(
                1 for msg in group_state.messages if user_id in msg.mentions
            )

            for msg in user_messages:
                if msg.topic:
                    topics[msg.topic] += 1
                if msg.sentiment:
                    sentiments[msg.sentiment] += 1

            return {
                "user_id": user_id,
                "username": participant.username,
                "role": participant.role,
                "join_date": participant.join_date.isoformat(),
                "last_active": participant.last_active.isoformat(),
                "message_count": participant.message_count,
                "favorite_topics": dict(topics),
                "sentiment_analysis": dict(sentiments),
                "mentions_given": dict(mentions_given),
                "mentions_received": mentions_received,
                "engagement_score": min(participant.message_count / 10, 1.0),
            }

        except Exception as e:
            logger.error(f"Error getting participant insights: {str(e)}")
            return {"error": str(e)}

    async def _create_group_record(
        self, group_id: int, group_name: str, group_type: str
    ):
        """Create database record for group"""
        try:
            query = """
                INSERT INTO group_conversations (group_id, group_name, group_type, created_at, settings)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                group_name = VALUES(group_name),
                group_type = VALUES(group_type)
            """

            settings = json.dumps(
                {
                    "max_context_messages": 50,
                    "auto_summarize": True,
                    "smart_threading": True,
                }
            )

            await self.db_connection.execute_query(
                query, (group_id, group_name, group_type, datetime.now(), settings)
            )

        except Exception as e:
            logger.error(f"Error creating group record: {str(e)}")

    async def _save_participant(self, group_id: int, participant: GroupParticipant):
        """Save participant to database"""
        try:
            query = """
                INSERT INTO group_participants (group_id, user_id, username, first_name, last_name, role, join_date, preferences)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                username = VALUES(username),
                first_name = VALUES(first_name),
                last_name = VALUES(last_name),
                role = VALUES(role)
            """

            await self.db_connection.execute_query(
                query,
                (
                    group_id,
                    participant.user_id,
                    participant.username,
                    participant.first_name,
                    participant.last_name,
                    participant.role,
                    participant.join_date,
                    json.dumps(participant.preferences),
                ),
            )

        except Exception as e:
            logger.error(f"Error saving participant: {str(e)}")

    async def _save_message(self, group_id: int, message: GroupMessage):
        """Save message to database"""
        try:
            query = """
                INSERT INTO group_messages (message_id, group_id, user_id, username, content, 
                                          timestamp, message_type, reply_to, mentions, hashtags,
                                          sentiment, topic, importance_score, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            await self.db_connection.execute_query(
                query,
                (
                    message.message_id,
                    group_id,
                    message.user_id,
                    message.username,
                    message.content,
                    message.timestamp,
                    message.message_type,
                    message.reply_to,
                    json.dumps(message.mentions),
                    json.dumps(message.hashtags),
                    message.sentiment,
                    message.topic,
                    message.importance_score,
                    json.dumps(message.metadata),
                ),
            )

        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")

    async def get_group_statistics(self, group_id: int) -> Dict[str, Any]:
        """Get comprehensive group statistics"""
        try:
            if group_id not in self.group_states:
                return {"error": "Group not found"}

            group_state = self.group_states[group_id]

            # Calculate statistics
            total_messages = len(group_state.messages)
            active_participants = len(
                [p for p in group_state.participants.values() if p.is_active]
            )

            # Message activity by hour
            message_hours = defaultdict(int)
            for msg in group_state.messages:
                hour = msg.timestamp.hour
                message_hours[hour] += 1

            # Top participants
            participant_stats = [
                {
                    "username": p.username,
                    "message_count": p.message_count,
                    "role": p.role,
                }
                for p in group_state.participants.values()
            ]
            participant_stats.sort(key=lambda x: x["message_count"], reverse=True)

            return {
                "group_id": group_id,
                "group_name": group_state.group_name,
                "total_messages": total_messages,
                "total_participants": len(group_state.participants),
                "active_participants": active_participants,
                "created_at": group_state.created_at.isoformat(),
                "last_activity": group_state.last_activity.isoformat(),
                "top_participants": participant_stats[:10],
                "hourly_activity": dict(message_hours),
                "current_topic": group_state.current_topic,
                "active_threads": len(group_state.active_threads),
            }

        except Exception as e:
            logger.error(f"Error getting group statistics: {str(e)}")
            return {"error": str(e)}


# Global instance
group_conversation_manager = GroupConversationManager()
