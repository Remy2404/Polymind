"""
👥 Advanced Group Chat Intelligence System
Provides sophisticated group collaboration, shared memory, and team coordination features
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class GroupMember:
    """Represents a group member with their participation metrics"""

    user_id: int
    username: Optional[str] = None
    join_date: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    contribution_score: float = 0.0
    expertise_areas: List[str] = field(default_factory=list)
    role: str = "member"  # member, moderator, admin
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupSession:
    """Represents an active group collaboration session"""

    session_id: str
    group_id: str
    topic: Optional[str] = None
    initiator_id: int = 0
    participants: Set[int] = field(default_factory=set)
    start_time: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    session_type: str = "general"  # general, brainstorm, decision, learning
    shared_context: Dict[str, Any] = field(default_factory=dict)
    collaborative_notes: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GroupKnowledge:
    """Represents shared knowledge within a group"""

    knowledge_id: str
    content: str
    category: str  # fact, process, decision, resource
    contributors: List[int] = field(default_factory=list)
    creation_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    importance_score: float = 0.5
    tags: List[str] = field(default_factory=list)
    related_sessions: List[str] = field(default_factory=list)


class GroupIntelligenceSystem:
    """Advanced group chat intelligence with collaboration features"""

    def __init__(self, memory_manager=None):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = memory_manager

        # Group management
        self.groups: Dict[str, Dict[str, Any]] = {}
        self.group_members: Dict[str, Dict[int, GroupMember]] = {}
        self.active_sessions: Dict[str, GroupSession] = {}

        # Knowledge base
        self.group_knowledge: Dict[str, List[GroupKnowledge]] = {}
        self.knowledge_index: Dict[str, Set[str]] = {}  # Tag-based indexing

        # Collaboration tracking
        self.interaction_patterns: Dict[str, Dict[str, Any]] = {}
        self.group_analytics: Dict[str, Dict[str, Any]] = {}

        # Real-time coordination
        self.active_discussions: Dict[str, Dict[str, Any]] = {}
        self.pending_decisions: Dict[str, List[Dict[str, Any]]] = {}

    async def initialize_group(
        self, group_id: str, group_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize a new group with intelligence features"""
        try:
            self.groups[group_id] = {
                "group_id": group_id,
                "name": group_info.get("name", f"Group {group_id}"),
                "description": group_info.get("description", ""),
                "created_date": datetime.now(),
                "settings": {
                    "auto_summarize": True,
                    "smart_notifications": True,
                    "collaboration_mode": "active",
                    "knowledge_retention": "high",
                    "privacy_level": "group",
                },
                "statistics": {
                    "total_messages": 0,
                    "active_members": 0,
                    "knowledge_items": 0,
                    "sessions_count": 0,
                },
            }

            self.group_members[group_id] = {}
            self.group_knowledge[group_id] = []
            self.interaction_patterns[group_id] = {
                "communication_flow": {},
                "topic_evolution": [],
                "collaboration_strength": 0.0,
            }

            # Initialize with memory manager
            if self.memory_manager:
                await self.memory_manager.load_memory(group_id, is_group=True)

            self.logger.info(f"Initialized group intelligence for {group_id}")
            return self.groups[group_id]

        except Exception as e:
            self.logger.error(f"Error initializing group {group_id}: {e}")
            raise

    async def add_group_member(
        self, group_id: str, user_id: int, member_info: Dict[str, Any]
    ) -> GroupMember:
        """Add a new member to the group with intelligence tracking"""
        try:
            if group_id not in self.group_members:
                self.group_members[group_id] = {}

            member = GroupMember(
                user_id=user_id,
                username=member_info.get("username"),
                role=member_info.get("role", "member"),
                preferences=member_info.get("preferences", {}),
            )

            self.group_members[group_id][user_id] = member

            # Update group statistics
            if group_id in self.groups:
                self.groups[group_id]["statistics"]["active_members"] = len(
                    self.group_members[group_id]
                )

            self.logger.info(f"Added member {user_id} to group {group_id}")
            return member

        except Exception as e:
            self.logger.error(f"Error adding member {user_id} to group {group_id}: {e}")
            raise

    async def start_collaboration_session(
        self, group_id: str, initiator_id: int, session_config: Dict[str, Any]
    ) -> GroupSession:
        """Start a new collaboration session"""
        try:
            session_id = f"session_{group_id}_{int(time.time())}"

            session = GroupSession(
                session_id=session_id,
                group_id=group_id,
                topic=session_config.get("topic"),
                initiator_id=initiator_id,
                participants={initiator_id},
                session_type=session_config.get("type", "general"),
            )

            self.active_sessions[session_id] = session

            # Initialize session in active discussions
            self.active_discussions[group_id] = {
                "current_session": session_id,
                "participants": {initiator_id},
                "topic": session.topic,
                "start_time": session.start_time,
                "message_flow": [],
                "consensus_level": 0.0,
            }

            # Update group statistics
            if group_id in self.groups:
                self.groups[group_id]["statistics"]["sessions_count"] += 1

            self.logger.info(
                f"Started collaboration session {session_id} in group {group_id}"
            )
            return session

        except Exception as e:
            self.logger.error(f"Error starting session in group {group_id}: {e}")
            raise

    async def process_group_message(
        self, group_id: str, user_id: int, message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a group message with intelligence analysis"""
        try:
            analysis = {
                "sentiment": 0.0,
                "topics": [],
                "action_items": [],
                "knowledge_updates": [],
                "collaboration_signals": [],
                "suggestions": [],
            }

            content = message.get("content", "")
            message_type = message.get("type", "text")

            # Update member activity
            await self._update_member_activity(group_id, user_id, message)

            # Analyze message content
            analysis = await self._analyze_message_content(
                group_id, content, message_type
            )

            # Extract knowledge items
            knowledge_items = await self._extract_knowledge_items(
                group_id, user_id, content
            )
            analysis["knowledge_updates"] = knowledge_items

            # Detect collaboration patterns
            collaboration_signals = await self._detect_collaboration_signals(
                group_id, user_id, content
            )
            analysis["collaboration_signals"] = collaboration_signals

            # Update interaction patterns
            await self._update_interaction_patterns(group_id, user_id, analysis)

            # Generate smart suggestions
            suggestions = await self._generate_smart_suggestions(
                group_id, user_id, content, analysis
            )
            analysis["suggestions"] = suggestions

            # Update active discussion if any
            if group_id in self.active_discussions:
                await self._update_active_discussion(
                    group_id, user_id, message, analysis
                )

            return analysis

        except Exception as e:
            self.logger.error(f"Error processing group message in {group_id}: {e}")
            return {"error": str(e)}

    async def get_group_intelligence_summary(self, group_id: str) -> Dict[str, Any]:
        """Get comprehensive intelligence summary for a group"""
        try:
            summary = {
                "group_info": self.groups.get(group_id, {}),
                "member_insights": await self._get_member_insights(group_id),
                "collaboration_health": await self._assess_collaboration_health(
                    group_id
                ),
                "knowledge_summary": await self._get_knowledge_summary(group_id),
                "recent_activity": await self._get_recent_activity_summary(group_id),
                "recommendations": await self._generate_group_recommendations(group_id),
            }

            return summary

        except Exception as e:
            self.logger.error(
                f"Error generating intelligence summary for group {group_id}: {e}"
            )
            return {"error": str(e)}

    async def create_shared_knowledge_item(
        self, group_id: str, contributor_id: int, knowledge_data: Dict[str, Any]
    ) -> GroupKnowledge:
        """Create a new shared knowledge item"""
        try:
            knowledge_id = f"knowledge_{group_id}_{int(time.time())}"

            knowledge = GroupKnowledge(
                knowledge_id=knowledge_id,
                content=knowledge_data.get("content", ""),
                category=knowledge_data.get("category", "general"),
                contributors=[contributor_id],
                importance_score=knowledge_data.get("importance", 0.5),
                tags=knowledge_data.get("tags", []),
            )

            if group_id not in self.group_knowledge:
                self.group_knowledge[group_id] = []

            self.group_knowledge[group_id].append(knowledge)

            # Update knowledge index
            for tag in knowledge.tags:
                if tag not in self.knowledge_index:
                    self.knowledge_index[tag] = set()
                self.knowledge_index[tag].add(knowledge_id)

            # Update group statistics
            if group_id in self.groups:
                self.groups[group_id]["statistics"]["knowledge_items"] += 1

            self.logger.info(
                f"Created knowledge item {knowledge_id} in group {group_id}"
            )
            return knowledge

        except Exception as e:
            self.logger.error(f"Error creating knowledge item in group {group_id}: {e}")
            raise

    async def search_group_knowledge(
        self, group_id: str, query: str, category: Optional[str] = None
    ) -> List[GroupKnowledge]:
        """Search group knowledge base"""
        try:
            if group_id not in self.group_knowledge:
                return []

            results = []
            query_lower = query.lower()

            for knowledge in self.group_knowledge[group_id]:
                # Category filter
                if category and knowledge.category != category:
                    continue

                # Content search
                relevance_score = 0.0

                # Direct content match
                if query_lower in knowledge.content.lower():
                    relevance_score += 0.8

                # Tag match
                for tag in knowledge.tags:
                    if query_lower in tag.lower():
                        relevance_score += 0.6

                # Category match
                if query_lower in knowledge.category.lower():
                    relevance_score += 0.4

                if relevance_score > 0:
                    results.append((knowledge, relevance_score))

            # Sort by relevance and importance
            results.sort(key=lambda x: (x[1], x[0].importance_score), reverse=True)

            return [item[0] for item in results]

        except Exception as e:
            self.logger.error(f"Error searching knowledge in group {group_id}: {e}")
            return []

    async def get_collaboration_insights(
        self, group_id: str, timeframe_days: int = 7
    ) -> Dict[str, Any]:
        """Get detailed collaboration insights"""
        try:
            insights = {
                "participation_metrics": {},
                "communication_patterns": {},
                "knowledge_sharing": {},
                "decision_making": {},
                "recommendations": [],
            }

            # Participation metrics
            if group_id in self.group_members:
                total_members = len(self.group_members[group_id])
                active_members = sum(
                    1
                    for member in self.group_members[group_id].values()
                    if (datetime.now() - member.last_activity).days <= timeframe_days
                )

                insights["participation_metrics"] = {
                    "total_members": total_members,
                    "active_members": active_members,
                    "participation_rate": (
                        active_members / total_members if total_members > 0 else 0
                    ),
                    "top_contributors": await self._get_top_contributors(
                        group_id, timeframe_days
                    ),
                }

            # Communication patterns
            if group_id in self.interaction_patterns:
                patterns = self.interaction_patterns[group_id]
                insights["communication_patterns"] = {
                    "collaboration_strength": patterns.get(
                        "collaboration_strength", 0.0
                    ),
                    "topic_diversity": len(patterns.get("topic_evolution", [])),
                    "interaction_frequency": await self._calculate_interaction_frequency(
                        group_id, timeframe_days
                    ),
                }

            # Knowledge sharing
            knowledge_items = self.group_knowledge.get(group_id, [])
            recent_knowledge = [
                k
                for k in knowledge_items
                if (datetime.now() - k.creation_date).days <= timeframe_days
            ]

            insights["knowledge_sharing"] = {
                "total_knowledge_items": len(knowledge_items),
                "recent_additions": len(recent_knowledge),
                "knowledge_categories": self._get_knowledge_categories(group_id),
                "knowledge_quality_score": await self._assess_knowledge_quality(
                    group_id
                ),
            }

            # Generate recommendations
            insights[
                "recommendations"
            ] = await self._generate_collaboration_recommendations(group_id, insights)

            return insights

        except Exception as e:
            self.logger.error(
                f"Error generating collaboration insights for group {group_id}: {e}"
            )
            return {"error": str(e)}

    # Private helper methods

    async def _update_member_activity(
        self, group_id: str, user_id: int, message: Dict[str, Any]
    ) -> None:
        """Update member activity metrics"""
        try:
            if (
                group_id in self.group_members
                and user_id in self.group_members[group_id]
            ):
                member = self.group_members[group_id][user_id]
                member.last_activity = datetime.now()
                member.message_count += 1

                # Update contribution score based on message quality
                content_length = len(message.get("content", ""))
                quality_bonus = 0.1 if content_length > 50 else 0.05
                member.contribution_score += quality_bonus

        except Exception as e:
            self.logger.error(f"Error updating member activity: {e}")

    async def _analyze_message_content(
        self, group_id: str, content: str, message_type: str
    ) -> Dict[str, Any]:
        """Analyze message content for intelligence insights"""
        analysis = {
            "sentiment": 0.0,
            "topics": [],
            "action_items": [],
            "questions": [],
            "decisions": [],
        }

        content_lower = content.lower()

        # Simple sentiment analysis
        positive_words = [
            "good",
            "great",
            "excellent",
            "awesome",
            "perfect",
            "love",
            "like",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "hate",
            "dislike",
            "problem",
            "issue",
        ]

        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)

        if positive_count + negative_count > 0:
            analysis["sentiment"] = (positive_count - negative_count) / (
                positive_count + negative_count
            )

        # Extract topics (simple keyword extraction)
        topic_keywords = [
            "project",
            "task",
            "idea",
            "plan",
            "goal",
            "issue",
            "problem",
            "solution",
        ]
        for keyword in topic_keywords:
            if keyword in content_lower:
                analysis["topics"].append(keyword)

        # Detect action items
        action_indicators = ["todo", "need to", "should", "must", "action", "task"]
        for indicator in action_indicators:
            if indicator in content_lower:
                # Extract the text around the action indicator
                start_idx = content_lower.find(indicator)
                end_idx = min(start_idx + 100, len(content))
                action_text = content[start_idx:end_idx]
                analysis["action_items"].append(action_text.strip())

        # Detect questions
        if "?" in content:
            questions = [q.strip() + "?" for q in content.split("?") if q.strip()]
            analysis["questions"] = questions[:-1]  # Remove empty last element

        # Detect decisions
        decision_indicators = ["decided", "agreed", "concluded", "final", "chosen"]
        for indicator in decision_indicators:
            if indicator in content_lower:
                analysis["decisions"].append(content[:200])  # First 200 chars
                break

        return analysis

    async def _extract_knowledge_items(
        self, group_id: str, user_id: int, content: str
    ) -> List[Dict[str, Any]]:
        """Extract potential knowledge items from content"""
        knowledge_items = []

        # Look for knowledge indicators
        knowledge_indicators = [
            "remember",
            "important",
            "note",
            "fact",
            "definition",
            "process",
            "procedure",
            "method",
            "tip",
            "lesson",
        ]

        content_lower = content.lower()

        for indicator in knowledge_indicators:
            if indicator in content_lower and len(content) > 30:
                # Determine category based on context
                category = "general"
                if any(
                    word in content_lower
                    for word in ["process", "procedure", "method", "how"]
                ):
                    category = "process"
                elif any(
                    word in content_lower
                    for word in ["fact", "definition", "what", "is"]
                ):
                    category = "fact"
                elif any(
                    word in content_lower for word in ["decided", "agreed", "concluded"]
                ):
                    category = "decision"

                knowledge_items.append(
                    {
                        "content": content,
                        "category": category,
                        "contributor": user_id,
                        "importance": (
                            0.7 if indicator in ["important", "remember"] else 0.5
                        ),
                        "auto_extracted": True,
                    }
                )
                break  # One knowledge item per message

        return knowledge_items

    async def _detect_collaboration_signals(
        self, group_id: str, user_id: int, content: str
    ) -> List[Dict[str, Any]]:
        """Detect collaboration signals in content"""
        signals = []
        content_lower = content.lower()

        # Collaboration indicators
        collaboration_patterns = {
            "request_help": ["help", "assist", "support", "can you", "could you"],
            "offer_help": ["i can", "let me", "i'll help", "i will"],
            "agreement": ["agree", "yes", "correct", "exactly", "right"],
            "disagreement": ["disagree", "no", "wrong", "incorrect", "but"],
            "question": ["?", "how", "what", "when", "where", "why"],
            "suggestion": ["suggest", "recommend", "propose", "maybe", "perhaps"],
            "decision": ["decide", "choose", "final", "concluded", "agreed"],
        }

        for signal_type, indicators in collaboration_patterns.items():
            for indicator in indicators:
                if indicator in content_lower:
                    signals.append(
                        {
                            "type": signal_type,
                            "confidence": 0.8,
                            "content_excerpt": content[:100],
                        }
                    )
                    break

        return signals

    async def _update_interaction_patterns(
        self, group_id: str, user_id: int, analysis: Dict[str, Any]
    ) -> None:
        """Update group interaction patterns"""
        try:
            if group_id not in self.interaction_patterns:
                self.interaction_patterns[group_id] = {
                    "communication_flow": {},
                    "topic_evolution": [],
                    "collaboration_strength": 0.0,
                }

            patterns = self.interaction_patterns[group_id]

            # Update communication flow
            if user_id not in patterns["communication_flow"]:
                patterns["communication_flow"][user_id] = {
                    "message_count": 0,
                    "avg_sentiment": 0.0,
                    "collaboration_score": 0.0,
                }

            user_flow = patterns["communication_flow"][user_id]
            user_flow["message_count"] += 1

            # Update sentiment average
            current_sentiment = analysis.get("sentiment", 0.0)
            user_flow["avg_sentiment"] = (
                user_flow["avg_sentiment"] * (user_flow["message_count"] - 1)
                + current_sentiment
            ) / user_flow["message_count"]

            # Update collaboration score based on signals
            collaboration_signals = analysis.get("collaboration_signals", [])
            collaboration_boost = len(collaboration_signals) * 0.1
            user_flow["collaboration_score"] = min(
                user_flow["collaboration_score"] + collaboration_boost, 1.0
            )

            # Update topic evolution
            topics = analysis.get("topics", [])
            for topic in topics:
                if topic not in patterns["topic_evolution"]:
                    patterns["topic_evolution"].append(topic)

            # Keep only recent topics (last 20)
            patterns["topic_evolution"] = patterns["topic_evolution"][-20:]

            # Calculate overall collaboration strength
            all_scores = [
                flow["collaboration_score"]
                for flow in patterns["communication_flow"].values()
            ]
            patterns["collaboration_strength"] = (
                sum(all_scores) / len(all_scores) if all_scores else 0.0
            )

        except Exception as e:
            self.logger.error(f"Error updating interaction patterns: {e}")

    async def _generate_smart_suggestions(
        self, group_id: str, user_id: int, content: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate smart suggestions based on analysis"""
        suggestions = []

        # Action item suggestions
        if analysis.get("action_items"):
            suggestions.append(
                {
                    "type": "action_tracking",
                    "title": "Track Action Items",
                    "description": "I detected action items. Would you like me to track them?",
                    "priority": "medium",
                }
            )

        # Knowledge capture suggestions
        if any(
            item.get("auto_extracted") for item in analysis.get("knowledge_updates", [])
        ):
            suggestions.append(
                {
                    "type": "knowledge_capture",
                    "title": "Save as Group Knowledge",
                    "description": "This seems like important information. Save to group knowledge base?",
                    "priority": "low",
                }
            )

        # Collaboration suggestions
        collaboration_signals = analysis.get("collaboration_signals", [])
        if any(
            signal.get("type") == "request_help" for signal in collaboration_signals
        ):
            suggestions.append(
                {
                    "type": "help_coordination",
                    "title": "Coordinate Help",
                    "description": "Someone needs help. I can notify relevant team members.",
                    "priority": "high",
                }
            )

        # Decision tracking
        if any(signal.get("type") == "decision" for signal in collaboration_signals):
            suggestions.append(
                {
                    "type": "decision_tracking",
                    "title": "Record Decision",
                    "description": "Record this decision in the group's decision log?",
                    "priority": "medium",
                }
            )

        return suggestions

    async def _update_active_discussion(
        self,
        group_id: str,
        user_id: int,
        message: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> None:
        """Update active discussion tracking"""
        try:
            if group_id in self.active_discussions:
                discussion = self.active_discussions[group_id]
                discussion["participants"].add(user_id)
                discussion["message_flow"].append(
                    {
                        "user_id": user_id,
                        "timestamp": datetime.now(),
                        "sentiment": analysis.get("sentiment", 0.0),
                        "topics": analysis.get("topics", []),
                    }
                )

                # Update consensus level based on agreement signals
                collaboration_signals = analysis.get("collaboration_signals", [])
                agreements = sum(
                    1
                    for signal in collaboration_signals
                    if signal.get("type") == "agreement"
                )
                disagreements = sum(
                    1
                    for signal in collaboration_signals
                    if signal.get("type") == "disagreement"
                )

                if agreements + disagreements > 0:
                    consensus_boost = (agreements - disagreements) * 0.1
                    discussion["consensus_level"] = max(
                        0, min(1, discussion["consensus_level"] + consensus_boost)
                    )

        except Exception as e:
            self.logger.error(f"Error updating active discussion: {e}")

    async def _get_member_insights(self, group_id: str) -> Dict[str, Any]:
        """Get insights about group members"""
        insights = {
            "total_members": 0,
            "active_members": 0,
            "top_contributors": [],
            "expertise_distribution": {},
            "engagement_levels": {},
        }

        if group_id not in self.group_members:
            return insights

        members = self.group_members[group_id]
        insights["total_members"] = len(members)

        # Calculate active members (active in last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        active_members = [m for m in members.values() if m.last_activity > cutoff_date]
        insights["active_members"] = len(active_members)

        # Top contributors
        sorted_members = sorted(
            members.values(), key=lambda m: m.contribution_score, reverse=True
        )
        insights["top_contributors"] = [
            {
                "user_id": m.user_id,
                "username": m.username,
                "contribution_score": m.contribution_score,
                "message_count": m.message_count,
            }
            for m in sorted_members[:5]
        ]

        return insights

    async def _assess_collaboration_health(self, group_id: str) -> Dict[str, Any]:
        """Assess the health of group collaboration"""
        health = {
            "overall_score": 0.0,
            "participation": 0.0,
            "communication": 0.0,
            "knowledge_sharing": 0.0,
            "decision_making": 0.0,
            "status": "unknown",
        }

        try:
            # Participation score
            if group_id in self.group_members:
                total_members = len(self.group_members[group_id])
                active_members = sum(
                    1
                    for member in self.group_members[group_id].values()
                    if (datetime.now() - member.last_activity).days <= 7
                )
                health["participation"] = (
                    active_members / total_members if total_members > 0 else 0
                )

            # Communication score
            if group_id in self.interaction_patterns:
                health["communication"] = self.interaction_patterns[group_id].get(
                    "collaboration_strength", 0.0
                )

            # Knowledge sharing score - count recent knowledge items
            recent_knowledge = len(
                [
                    k
                    for k in self.group_knowledge.get(group_id, [])
                    if (datetime.now() - k.creation_date).days <= 7
                ]
            )
            health["knowledge_sharing"] = min(
                recent_knowledge / 5, 1.0
            )  # Normalize to 5 items per week

            # Overall score
            scores = [
                health["participation"],
                health["communication"],
                health["knowledge_sharing"],
            ]
            health["overall_score"] = sum(scores) / len(scores)

            # Status
            if health["overall_score"] >= 0.8:
                health["status"] = "excellent"
            elif health["overall_score"] >= 0.6:
                health["status"] = "good"
            elif health["overall_score"] >= 0.4:
                health["status"] = "moderate"
            else:
                health["status"] = "needs_attention"

        except Exception as e:
            self.logger.error(f"Error assessing collaboration health: {e}")

        return health

    async def _get_knowledge_summary(self, group_id: str) -> Dict[str, Any]:
        """Get summary of group knowledge"""
        summary = {
            "total_items": 0,
            "categories": {},
            "recent_additions": 0,
            "top_contributors": [],
            "trending_topics": [],
        }

        knowledge_items = self.group_knowledge.get(group_id, [])
        summary["total_items"] = len(knowledge_items)

        # Category distribution
        for item in knowledge_items:
            category = item.category
            summary["categories"][category] = summary["categories"].get(category, 0) + 1

        # Recent additions (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        recent_items = [k for k in knowledge_items if k.creation_date > cutoff_date]
        summary["recent_additions"] = len(recent_items)

        # Top contributors
        contributor_counts = defaultdict(int)
        for item in knowledge_items:
            for contributor in item.contributors:
                contributor_counts[contributor] += 1

        summary["top_contributors"] = [
            {"user_id": user_id, "contributions": count}
            for user_id, count in sorted(
                contributor_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]

        return summary

    async def _get_recent_activity_summary(
        self, group_id: str, days: int = 7
    ) -> Dict[str, Any]:
        """Get summary of recent group activity"""
        summary = {
            "messages_count": 0,
            "active_users": 0,
            "sessions_started": 0,
            "knowledge_added": 0,
            "decisions_made": 0,
        }

        # This would typically query the message history
        # For now, we'll use the tracked data

        cutoff_date = datetime.now() - timedelta(days=days)

        # Count active users
        if group_id in self.group_members:
            summary["active_users"] = sum(
                1
                for member in self.group_members[group_id].values()
                if member.last_activity > cutoff_date
            )

        # Count recent knowledge additions
        knowledge_items = self.group_knowledge.get(group_id, [])
        summary["knowledge_added"] = sum(
            1 for item in knowledge_items if item.creation_date > cutoff_date
        )

        return summary

    async def _generate_group_recommendations(
        self, group_id: str
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for improving group collaboration"""
        recommendations = []

        # Get current health assessment
        health = await self._assess_collaboration_health(group_id)

        # Participation recommendations
        if health["participation"] < 0.5:
            recommendations.append(
                {
                    "type": "participation",
                    "priority": "high",
                    "title": "Increase Member Participation",
                    "description": "Consider scheduling regular check-ins or creating engaging discussion topics.",
                    "actions": [
                        "Start a group session with a specific topic",
                        "Ask direct questions to inactive members",
                        "Share interesting updates or resources",
                    ],
                }
            )

        # Communication recommendations
        if health["communication"] < 0.6:
            recommendations.append(
                {
                    "type": "communication",
                    "priority": "medium",
                    "title": "Improve Communication Flow",
                    "description": "Encourage more collaborative discussions and feedback.",
                    "actions": [
                        "Use more collaborative language",
                        "Ask for opinions and feedback",
                        "Acknowledge others' contributions",
                    ],
                }
            )

        # Knowledge sharing recommendations
        if health["knowledge_sharing"] < 0.4:
            recommendations.append(
                {
                    "type": "knowledge",
                    "priority": "medium",
                    "title": "Enhance Knowledge Sharing",
                    "description": "Create and share more valuable information within the group.",
                    "actions": [
                        "Document important decisions",
                        "Share useful resources and tips",
                        "Create how-to guides for common tasks",
                    ],
                }
            )

        return recommendations
