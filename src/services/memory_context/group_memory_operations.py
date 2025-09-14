"""
Group Memory Operations Module
Handles group-specific memory operations and context management
"""
import logging
import time
import re
from typing import Dict, List, Any, Tuple
from collections import defaultdict
logger = logging.getLogger(__name__)
class GroupMemoryOperations:
    """Handles group-specific memory operations and context management"""
    def __init__(self):
        self.group_contexts = {}
        self.shared_knowledge = {}
    async def get_group_participants(
        self, group_id: str, group_memory_cache: Dict
    ) -> List[str]:
        """Get list of participants in a group conversation"""
        if group_id not in group_memory_cache:
            return []
        participants = set()
        for message in group_memory_cache[group_id]:
            if message.get("user_id"):
                participants.add(message["user_id"])
        return list(participants)
    async def get_group_activity_summary(
        self, group_id: str, group_memory_cache: Dict, days: int = 7
    ) -> Dict[str, Any]:
        """Get group activity summary for specified days"""
        if group_id not in group_memory_cache:
            return {}
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        recent_messages = [
            msg
            for msg in group_memory_cache[group_id]
            if msg.get("timestamp", 0) > cutoff_time
        ]
        user_activity = defaultdict(int)
        message_types = defaultdict(int)
        topics = []
        for message in recent_messages:
            if message.get("user_id"):
                user_activity[message["user_id"]] += 1
            message_types[message.get("message_type", "text")] += 1
            content = message.get("content", "")
            if len(content) > 20:
                topics.append(content[:100])
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
        }
    async def update_group_context(self, group_id: str, message: Dict[str, Any]):
        """Update active group conversation context"""
        if group_id not in self.group_contexts:
            self.group_contexts[group_id] = {
                "active_users": set(),
                "current_topic": None,
                "last_activity": time.time(),
                "message_count": 0,
            }
        context = self.group_contexts[group_id]
        if message.get("user_id"):
            context["active_users"].add(message["user_id"])
        context["last_activity"] = time.time()
        context["message_count"] += 1
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
    async def update_shared_knowledge(self, group_id: str, content: str):
        """Update shared knowledge base for the group"""
        if group_id not in self.shared_knowledge:
            self.shared_knowledge[group_id] = []
        if len(content) > 50:
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
            if len(self.shared_knowledge[group_id]) > 100:
                self.shared_knowledge[group_id].sort(
                    key=lambda x: (x["importance"], x["timestamp"]),
                    reverse=True,
                )
                self.shared_knowledge[group_id] = self.shared_knowledge[group_id][:100]
    async def get_shared_knowledge(
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
                    relevant_knowledge.append((-(idx + 1000), similarity))
        return relevant_knowledge
    def get_group_contexts(self) -> Dict[str, Any]:
        """Get all group contexts"""
        return self.group_contexts
    def get_shared_knowledge_for_group(self, group_id: str) -> List[Dict[str, Any]]:
        """Get all shared knowledge for a specific group"""
        return self.shared_knowledge.get(group_id, [])
    def clear_group_data(self, group_id: str):
        """Clear all group-specific data"""
        self.group_contexts.pop(group_id, None)
        self.shared_knowledge.pop(group_id, None)
