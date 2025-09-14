import logging
import asyncio
from typing import Dict, List, Any, Optional
from .memory_manager import MemoryManager
from .model_history_manager import ModelHistoryManager
logger = logging.getLogger(__name__)
class ConversationManager:
    """Enhanced conversation manager with group collaboration and intelligent memory"""
    def __init__(
        self, memory_manager: MemoryManager, model_history_manager: ModelHistoryManager
    ):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = memory_manager
        self.model_history_manager = model_history_manager
        self.group_sessions = {}
        self.group_participants = {}
        self.collaborative_contexts = {}
    async def get_conversation_history(
        self,
        user_id: int,
        max_messages: int = 10,
        model: Optional[str] = None,
        group_id: Optional[str] = None,
        include_group_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history with enhanced group support and context awareness.
        """
        if group_id:
            history = await self._get_group_conversation_history(
                group_id, user_id, max_messages, include_group_context
            )
            if history:
                self.logger.info(
                    f"Retrieved {len(history)} message(s) from group {group_id} for user {user_id}"
                )
                return history
        if model:
            history = await self.model_history_manager.get_history(
                user_id, max_messages=max_messages, model_id=model
            )
            if history:
                enhanced_history = await self._enhance_history_with_context(
                    user_id, history, group_id
                )
                self.logger.info(
                    f"Retrieved {len(enhanced_history)} enhanced message(s) from model-specific history for {model}"
                )
                return enhanced_history
        history = await self.model_history_manager.get_history(
            user_id, max_messages=max_messages
        )
        return await self._enhance_history_with_context(user_id, history, group_id)
    async def save_message_pair(
        self,
        user_id: int,
        user_message: str,
        assistant_message: str,
        model_id: Optional[str] = None,
        group_id: Optional[str] = None,
        message_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a user-assistant message pair with enhanced group and context support."""
        importance = await self._calculate_message_importance(
            user_message, assistant_message
        )
        await self.model_history_manager.save_message_pair(
            user_id, user_message, assistant_message, model_id
        )
        if group_id:
            await self._save_group_message_pair(
                group_id,
                user_id,
                user_message,
                assistant_message,
                importance,
                message_context,
            )
        else:
            conversation_id = f"user_{user_id}"
            await self.memory_manager.add_user_message(
                conversation_id,
                user_message,
                str(user_id),
                importance=importance,
                message_context=message_context or {},
            )
            await self.memory_manager.add_assistant_message(
                conversation_id,
                assistant_message,
                importance=importance,
                related_user_message=user_message,
            )
    async def save_media_interaction(
        self,
        user_id: int,
        media_type: str,
        prompt: str,
        response: str,
        group_id: Optional[str] = None,
        **metadata,
    ) -> None:
        """Save media interaction with enhanced group support and intelligence."""
        conversation_id = f"user_{user_id}"
        content = f"[{media_type.capitalize()} Analysis] {prompt}"
        base_importance = 0.8 if media_type.lower() == "image" else 0.6
        context_boost = (
            0.1
            if any(
                keyword in prompt.lower()
                for keyword in ["important", "remember", "save", "note"]
            )
            else 0
        )
        importance = min(base_importance + context_boost, 1.0)
        enhanced_metadata = {
            "is_media": True,
            "media_type": media_type,
            "prompt": prompt,
            "timestamp": metadata.get("timestamp"),
            "group_id": group_id,
            "analysis_quality": await self._assess_response_quality(response),
            **metadata,
        }
        if group_id:
            await self.memory_manager.add_user_message(
                conversation_id,
                content,
                str(user_id),
                message_type=media_type,
                importance=importance,
                is_group=True,
                group_id=group_id,
                **enhanced_metadata,
            )
            await self.memory_manager.add_assistant_message(
                conversation_id,
                response,
                message_type=f"{media_type}_analysis",
                importance=importance + 0.05,
                is_group=True,
                group_id=group_id,
                media_description=True,
                related_media_type=media_type,
            )
            await self._update_group_collaborative_context(
                group_id, user_id, media_type, prompt, response
            )
        else:
            await self.memory_manager.add_user_message(
                conversation_id,
                content,
                str(user_id),
                message_type=media_type,
                importance=importance,
                **enhanced_metadata,
            )
            if media_type.lower() == "image":
                await self.memory_manager.add_assistant_message(
                    conversation_id,
                    response,
                    message_type="image_analysis",
                    importance=importance + 0.05,
                    media_description=True,
                    related_media_type="image",
                )
            else:
                await self.memory_manager.add_assistant_message(
                    conversation_id, response, importance=importance
                )
    async def start_group_session(
        self, group_id: str, initiator_user_id: int, session_topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a collaborative group session with shared context."""
        session_id = f"group_{group_id}_{int(asyncio.get_event_loop().time())}"
        session_data = {
            "session_id": session_id,
            "group_id": group_id,
            "initiator": initiator_user_id,
            "topic": session_topic,
            "participants": {initiator_user_id},
            "start_time": asyncio.get_event_loop().time(),
            "active": True,
            "shared_context": {},
            "collaborative_notes": [],
            "decision_points": [],
        }
        self.group_sessions[session_id] = session_data
        await self.memory_manager.load_memory(group_id, is_group=True)
        self.logger.info(f"Started group session {session_id} for group {group_id}")
        return session_data
    async def join_group_session(self, session_id: str, user_id: int) -> bool:
        """Add a user to an active group session."""
        if session_id not in self.group_sessions:
            return False
        session = self.group_sessions[session_id]
        if not session["active"]:
            return False
        session["participants"].add(user_id)
        group_id = session["group_id"]
        if group_id not in self.group_participants:
            self.group_participants[group_id] = set()
        self.group_participants[group_id].add(user_id)
        self.logger.info(f"User {user_id} joined group session {session_id}")
        return True
    async def get_group_context(self, group_id: str, user_id: int) -> Dict[str, Any]:
        """Get comprehensive group context for a user."""
        context = {
            "group_summary": await self.memory_manager.get_conversation_summary(
                "", is_group=True, group_id=group_id
            ),
            "recent_activity": await self.memory_manager.get_short_term_memory(
                "", limit=10, is_group=True, group_id=group_id
            ),
            "participants": await self.memory_manager.get_group_participants(group_id),
            "shared_knowledge": [],
            "active_topics": [],
            "user_contributions": [],
        }
        if group_id in self.memory_manager.group_memory_cache:
            user_messages = [
                msg
                for msg in self.memory_manager.group_memory_cache[group_id]
                if msg.get("user_id") == str(user_id)
            ]
            context["user_contributions"] = user_messages[-5:]
        if group_id in self.collaborative_contexts:
            context["collaborative_notes"] = self.collaborative_contexts[group_id].get(
                "notes", []
            )
            context["active_topics"] = self.collaborative_contexts[group_id].get(
                "topics", []
            )
        return context
    async def add_collaborative_note(
        self, group_id: str, user_id: int, note: str, note_type: str = "general"
    ) -> None:
        """Add a collaborative note that's shared across the group."""
        if group_id not in self.collaborative_contexts:
            self.collaborative_contexts[group_id] = {
                "notes": [],
                "topics": [],
                "decisions": [],
                "action_items": [],
            }
        note_data = {
            "content": note,
            "type": note_type,
            "author": user_id,
            "timestamp": asyncio.get_event_loop().time(),
            "id": f"note_{len(self.collaborative_contexts[group_id]['notes'])}",
        }
        self.collaborative_contexts[group_id]["notes"].append(note_data)
        await self.memory_manager.add_assistant_message(
            "",
            f"[Collaborative Note by User {user_id}] {note}",
            importance=0.9,
            is_group=True,
            group_id=group_id,
            message_type="collaborative_note",
        )
    async def get_intelligent_context(
        self, user_id: int, current_query: str, group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get intelligent context based on current query and conversation state."""
        context = {
            "relevant_memory": [],
            "suggested_actions": [],
            "related_topics": [],
            "group_insights": None,
        }
        if group_id:
            relevant_memory = await self.memory_manager.get_relevant_memory(
                "", current_query, limit=5, is_group=True, group_id=group_id
            )
        else:
            conversation_id = f"user_{user_id}"
            relevant_memory = await self.memory_manager.get_relevant_memory(
                conversation_id, current_query, limit=5
            )
        context["relevant_memory"] = relevant_memory
        context["suggested_actions"] = await self._generate_smart_suggestions(
            current_query, relevant_memory, group_id
        )
        if group_id:
            context["group_insights"] = await self._generate_group_insights(
                group_id, current_query
            )
        return context
    async def add_quoted_message_context(
        self,
        user_id: int,
        quoted_text: str,
        user_message: str,
        assistant_message: str,
        model_id: Optional[str] = None,
        group_id: Optional[str] = None,
        message_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a message pair with quoted context for reply functionality."""
        enhanced_context = message_context or {}
        enhanced_context.update(
            {
                "quoted_text": quoted_text,
                "message_type": "reply",
                "has_context": True,
            }
        )
        enhanced_user_message = (
            f'[Replying to: "{quoted_text[:100]}{"..." if len(quoted_text) > 100 else ""}"]\n\n'
            f"{user_message}"
        )
        importance = await self._calculate_message_importance(
            enhanced_user_message, assistant_message
        )
        importance = min(1.0, importance + 0.1)
        await self.model_history_manager.save_message_pair(
            user_id, enhanced_user_message, assistant_message, model_id
        )
        if group_id:
            await self._save_group_message_pair(
                group_id,
                user_id,
                enhanced_user_message,
                assistant_message,
                importance,
                enhanced_context,
            )
        else:
            conversation_id = f"user_{user_id}"
            await self.memory_manager.add_user_message(
                conversation_id,
                enhanced_user_message,
                str(user_id),
                importance=importance,
                message_context=enhanced_context,
            )
            await self.memory_manager.add_assistant_message(
                conversation_id,
                assistant_message,
                importance=importance,
                related_user_message=enhanced_user_message,
            )
    async def _get_group_conversation_history(
        self, group_id: str, user_id: int, max_messages: int, include_context: bool
    ) -> List[Dict[str, Any]]:
        """Get group conversation history with context."""
        recent_messages = await self.memory_manager.get_short_term_memory(
            "", limit=max_messages, is_group=True, group_id=group_id
        )
        if include_context:
            relevant_context = await self.memory_manager.get_relevant_memory(
                "",
                f"user_{user_id}_recent_activity",
                limit=3,
                is_group=True,
                group_id=group_id,
            )
            all_messages = recent_messages + relevant_context
            all_messages.sort(key=lambda x: x.get("timestamp", 0))
            return all_messages[-max_messages:]
        return recent_messages
    async def _enhance_history_with_context(
        self, user_id: int, history: List[Dict[str, Any]], group_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Enhance conversation history with intelligent context."""
        if not history:
            return history
        enhanced_history = []
        for message in history:
            enhanced_message = message.copy()
            if group_id:
                enhanced_message["group_context"] = True
                enhanced_message["group_id"] = group_id
            content = message.get("content", "")
            if len(content) > 20:
                enhanced_message["content_summary"] = (
                    content[:100] + "..." if len(content) > 100 else content
                )
            enhanced_history.append(enhanced_message)
        return enhanced_history
    async def _calculate_message_importance(
        self, user_message: str, assistant_message: str
    ) -> float:
        """Calculate message importance based on content analysis."""
        base_importance = 0.5
        important_keywords = [
            "remember",
            "important",
            "save",
            "note",
            "task",
            "deadline",
            "meeting",
            "project",
            "decision",
            "action",
            "follow up",
        ]
        combined_text = (user_message + " " + assistant_message).lower()
        keyword_matches = sum(
            1 for keyword in important_keywords if keyword in combined_text
        )
        importance_boost = min(keyword_matches * 0.1, 0.3)
        length_factor = min(len(combined_text) / 500, 0.2)
        qa_pattern_boost = (
            0.1 if ("?" in user_message and len(assistant_message) > 50) else 0
        )
        final_importance = min(
            base_importance + importance_boost + length_factor + qa_pattern_boost, 1.0
        )
        return final_importance
    async def _save_group_message_pair(
        self,
        group_id: str,
        user_id: int,
        user_message: str,
        assistant_message: str,
        importance: float,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Save message pair to group memory with collaborative features."""
        conversation_id = f"group_{group_id}"
        group_metadata = {
            "group_participants": await self.memory_manager.get_group_participants(
                group_id
            ),
            "message_context": context or {},
            "collaborative": True,
        }
        await self.memory_manager.add_user_message(
            conversation_id,
            user_message,
            str(user_id),
            importance=importance,
            is_group=True,
            group_id=group_id,
            **group_metadata,
        )
        await self.memory_manager.add_assistant_message(
            conversation_id,
            assistant_message,
            importance=importance,
            is_group=True,
            group_id=group_id,
            related_user_message=user_message,
        )
        await self._update_group_collaborative_context(
            group_id, user_id, "text", user_message, assistant_message
        )
    async def _update_group_collaborative_context(
        self,
        group_id: str,
        user_id: int,
        interaction_type: str,
        user_input: str,
        assistant_response: str,
    ) -> None:
        """Update collaborative context for the group."""
        if group_id not in self.collaborative_contexts:
            self.collaborative_contexts[group_id] = {
                "notes": [],
                "topics": [],
                "decisions": [],
                "action_items": [],
            }
        context = self.collaborative_contexts[group_id]
        if any(
            keyword in user_input.lower()
            for keyword in ["todo", "task", "action", "need to"]
        ):
            action_item = {
                "content": user_input,
                "assigned_by": user_id,
                "timestamp": asyncio.get_event_loop().time(),
                "status": "pending",
            }
            context["action_items"].append(action_item)
        if any(
            keyword in assistant_response.lower()
            for keyword in ["decided", "conclusion", "agree", "final"]
        ):
            decision = {
                "content": assistant_response[:200],
                "participants": [user_id],
                "timestamp": asyncio.get_event_loop().time(),
            }
            context["decisions"].append(decision)
        topics = []
        for text in [user_input, assistant_response]:
            words = text.lower().split()
            for i, word in enumerate(words):
                if word in ["about", "regarding", "concerning"] and i + 1 < len(words):
                    topics.append(words[i + 1])
        for topic in topics:
            if topic not in context["topics"]:
                context["topics"].append(topic)
        context["topics"] = context["topics"][-10:]
    async def _assess_response_quality(self, response: str) -> float:
        """Assess the quality of an AI response for importance scoring."""
        if not response:
            return 0.0
        quality_factors = {
            "length": min(len(response) / 200, 1.0),
            "structure": (
                0.2
                if any(marker in response for marker in ["1.", "2.", "-", "*"])
                else 0
            ),
            "code": 0.3 if "```" in response else 0,
            "explanation": (
                0.2
                if any(
                    word in response.lower()
                    for word in ["because", "therefore", "explains", "analysis"]
                )
                else 0
            ),
        }
        return min(sum(quality_factors.values()) / len(quality_factors), 1.0)
    async def _generate_smart_suggestions(
        self, query: str, relevant_memory: List[Dict[str, Any]], group_id: Optional[str]
    ) -> List[str]:
        """Generate smart action suggestions based on context."""
        suggestions = []
        query_lower = query.lower()
        if any(
            keyword in query_lower for keyword in ["help", "how", "what", "explain"]
        ):
            suggestions.append("Ask for more detailed explanation")
            suggestions.append("Request examples or use cases")
        if any(
            keyword in query_lower for keyword in ["code", "programming", "function"]
        ):
            suggestions.append("Ask for code examples")
            suggestions.append("Request best practices")
        if group_id and any(
            keyword in query_lower for keyword in ["team", "group", "collaborate"]
        ):
            suggestions.append("Share with group members")
            suggestions.append("Create collaborative note")
            suggestions.append("Assign action items")
        if relevant_memory:
            suggestions.append("Review related previous conversations")
            if len(relevant_memory) > 2:
                suggestions.append("Get conversation summary")
        return suggestions[:5]
    async def _generate_group_insights(
        self, group_id: str, current_query: str
    ) -> Dict[str, Any]:
        """Generate insights about group collaboration patterns."""
        try:
            activity_summary = await self.memory_manager.get_group_activity_summary(
                group_id
            )
            insights = {
                "participation_level": (
                    "high"
                    if activity_summary.get("total_messages", 0) > 50
                    else "moderate"
                ),
                "collaboration_health": (
                    "good" if activity_summary.get("active_users", 0) > 1 else "low"
                ),
                "recent_trends": [],
                "recommendations": [],
            }
            if activity_summary.get("active_users", 0) == 1:
                insights["recommendations"].append(
                    "Consider inviting more team members to join the discussion"
                )
            if activity_summary.get("total_messages", 0) > 100:
                insights["recommendations"].append(
                    "Consider creating a summary of recent discussions"
                )
            return insights
        except Exception as e:
            self.logger.error(f"Error generating group insights: {e}")
            return {"error": "Unable to generate insights"}
    async def reset_conversation(
        self, user_id: int, group_id: Optional[str] = None
    ) -> None:
        """Clear conversation history with group support."""
        if group_id:
            await self.memory_manager.clear_conversation(
                "", is_group=True, group_id=group_id
            )
            self.collaborative_contexts.pop(group_id, None)
            self.group_participants.pop(group_id, None)
        else:
            await self.model_history_manager.clear_history(user_id)
            conversation_id = f"user_{user_id}"
            await self.memory_manager.clear_conversation(conversation_id)
    async def get_short_term_memory(
        self, user_id: int, limit: int = 5, group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent messages with group support."""
        if group_id:
            return await self.memory_manager.get_short_term_memory(
                "", limit=limit, is_group=True, group_id=group_id
            )
        else:
            conversation_id = f"user_{user_id}"
            return await self.memory_manager.get_short_term_memory(
                conversation_id, limit
            )
    async def get_long_term_memory(
        self, user_id: int, query: str, limit: int = 3, group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant messages from long-term memory with group support."""
        if group_id:
            return await self.memory_manager.get_relevant_memory(
                "",
                query,
                limit,
                is_group=True,
                group_id=group_id,
                include_group_knowledge=True,
            )
        else:
            conversation_id = f"user_{user_id}"
            return await self.memory_manager.get_relevant_memory(
                conversation_id, query, limit
            )
