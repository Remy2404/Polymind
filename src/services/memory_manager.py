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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
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
    """Manages conversation memory with a tiered approach for different time horizons"""

    def __init__(self, db=None, storage_path="./data/memory"):
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.storage_path = storage_path

        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)

        # Initialize memory structures
        self.short_term_memory = {}  # Most recent interactions (last 10)
        self.medium_term_memory = {}  # Important context (last 50)
        self.long_term_memory = {}  # Persistent user preferences and critical info

        # Conversation memory limits
        self.short_term_limit = 10
        self.medium_term_limit = 50
        self.conversation_expiry = timedelta(
            hours=24
        )  # When to move to long-term storage
        self.token_limit = 8192  # Default token limit

        # Setup message importance classification patterns
        self._init_importance_patterns()

    def _init_importance_patterns(self):
        """Initialize patterns for detecting message importance"""
        self.high_importance_patterns = [
            # User preferences and personal info
            r"\bmy name is\b|\bi am called\b|\bplease call me\b|\bi'm (\w+)\b|\bI am (\w+)\b",
            r"\bi prefer\b|\bi like\b|\bi want\b|\bi need\b",
            r"\bremember that\b|\bdon\'t forget\b|\bmake sure\b",
            # Professional or technical information
            r"\btechnical specs\b|\bspecifications\b|\brequirements\b",
            r"\bAPI key\b|\bpassword\b|\bcredentials\b|\btoken\b",
            # Location or time context
            r"\bi\'m in\b|\bi am in\b|\bmy location\b|\bi live in\b",
            r"\btime zone\b|\bschedule\b|\bdeadline\b|\bdue date\b",
            # Document context
            r"\bdocument ID\b|\bfile reference\b|\breport number\b",
            r"\bversion\b|\brelease\b|\bupdate\b|\bpatch\b",
            # User intentions
            r"\bmy goal is\b|\bi\'m trying to\b|\bi want to achieve\b",
            r"\bproject\b|\btask\b|\bassignment\b|\bmission\b",
            # Specific questions about the model's memory
            r"\bdo you (?:know|remember) (?:my|me)\b",
        ]

        # Add specific pattern for capturing names
        self.name_pattern = re.compile(
            r"\bmy name is\s+(\w+)|\bi am\s+(\w+)|\bcall me\s+(\w+)", re.IGNORECASE
        )

    async def add_user_message(
        self,
        conversation_id: str,
        message: str,
        user_id: str,
        message_type: str = "text",
        **metadata,
    ) -> None:
        """Add a user message to memory with importance classification"""
        timestamp = metadata.get("timestamp", datetime.now().timestamp())

        # Evaluate message importance
        importance = self._evaluate_message_importance(message)

        # Create message object
        message_obj = {
            "sender": "user",
            "content": message,
            "timestamp": timestamp,
            "type": message_type,
            "importance": importance,
            "metadata": metadata,
        }

        # Initialize conversation if needed
        if conversation_id not in self.short_term_memory:
            self.short_term_memory[conversation_id] = []

        if conversation_id not in self.medium_term_memory:
            self.medium_term_memory[conversation_id] = []

        # Add to short-term memory
        self.short_term_memory[conversation_id].append(message_obj)

        # Also add to medium-term memory if important
        if importance >= 0.6:
            self.medium_term_memory[conversation_id].append(message_obj)

        # Add key information to long-term memory
        if importance >= 0.8:
            if user_id not in self.long_term_memory:
                self.long_term_memory[user_id] = {
                    "preferences": {},
                    "facts": [],
                    "contexts": {},
                }

            # Extract information for long-term memory
            self._extract_long_term_info(user_id, message, message_obj)

        # Trim memory if needed
        self._trim_memory(conversation_id)

        # Persist memory periodically (could be optimized with a periodic task)
        await self._save_memory(conversation_id, user_id)

    async def add_assistant_message(
        self,
        conversation_id: str,
        message: str,
        message_type: str = "text",
        **metadata,
    ) -> None:
        """Add an assistant message to memory"""
        timestamp = metadata.get("timestamp", datetime.now().timestamp())

        # Create message object
        message_obj = {
            "sender": "assistant",
            "content": message,
            "timestamp": timestamp,
            "type": message_type,
            "metadata": metadata,
        }

        # Initialize conversation if needed
        if conversation_id not in self.short_term_memory:
            self.short_term_memory[conversation_id] = []

        # Add to short-term memory
        self.short_term_memory[conversation_id].append(message_obj)

        # Trim memory if needed
        self._trim_memory(conversation_id)

        # Persist memory periodically
        await self._save_memory(
            conversation_id, ""
        )  # No user ID needed for assistant messages

    # This is the missing method that's causing the error
    def get_formatted_history(
        self, conversation_id: str, max_messages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history formatted for AI model consumption.

        Args:
            conversation_id: The ID of the conversation to retrieve
            max_messages: Maximum number of messages to include

        Returns:
            List of message dictionaries formatted for model context
        """
        formatted_history = []

        # Get messages from short-term memory
        messages = self.short_term_memory.get(conversation_id, [])

        # Only take the most recent messages up to max_messages
        recent_messages = (
            messages[-max_messages:] if len(messages) > max_messages else messages
        )

        # Format the messages for the AI model
        for msg in recent_messages:
            sender = msg.get("sender", "")
            content = msg.get("content", "")

            # Map 'sender' to standard role names expected by AI models
            role = "user" if sender == "user" else "assistant"

            formatted_history.append({"role": role, "content": content})

        return formatted_history

    def get_messages(self, conversation_id: str) -> List[Message]:
        """Get all messages for a conversation as Message objects"""
        messages = []

        # Get raw message data from memory
        raw_messages = self.short_term_memory.get(conversation_id, [])

        # Convert to Message objects
        for msg in raw_messages:
            role = "user" if msg.get("sender") == "user" else "assistant"
            message = Message(
                role=role,
                content=msg.get("content", ""),
                timestamp=msg.get("timestamp", time.time()),
                metadata=msg.get("metadata", {}),
            )
            messages.append(message)

        return messages

    async def _maybe_manage_context_window(self, conversation_id: str) -> None:
        """Check if context window is getting too large and trim if needed"""
        if conversation_id not in self.short_term_memory:
            return

        # Estimate token count (very rough approximation)
        total_tokens = sum(
            len(msg.get("content", "").split()) * 1.3
            for msg in self.short_term_memory[conversation_id]
        )

        # If approaching token limit, trim older messages
        if total_tokens > self.token_limit * 0.8:
            # Sort by importance first, then timestamp
            messages = self.short_term_memory[conversation_id]
            messages.sort(
                key=lambda x: (x.get("importance", 0), x.get("timestamp", 0)),
                reverse=True,
            )

            # Keep most important messages within token limit
            kept_messages = []
            token_count = 0

            for msg in messages:
                msg_tokens = len(msg.get("content", "").split()) * 1.3
                if token_count + msg_tokens < self.token_limit * 0.7:
                    kept_messages.append(msg)
                    token_count += msg_tokens

            # Sort back by timestamp
            kept_messages.sort(key=lambda x: x.get("timestamp", 0))
            self.short_term_memory[conversation_id] = kept_messages

            # Log the context management action
            self.logger.info(
                f"Managed context window for {conversation_id}: Reduced from {total_tokens:.0f} to {token_count:.0f} tokens"
            )

    async def add_bot_message(
        self,
        conversation_id: str,
        message: str,
        user_id: str,
        message_type: str = "text",
        **metadata,
    ) -> None:
        """Add a bot message to memory"""
        timestamp = metadata.get("timestamp", datetime.now().timestamp())

        # Create message object
        message_obj = {
            "sender": "bot",
            "content": message,
            "timestamp": timestamp,
            "type": message_type,
            "metadata": metadata,
        }

        # Initialize conversation if needed
        if conversation_id not in self.short_term_memory:
            self.short_term_memory[conversation_id] = []

        if conversation_id not in self.medium_term_memory:
            self.medium_term_memory[conversation_id] = []

        # Add to short-term memory
        self.short_term_memory[conversation_id].append(message_obj)

        # Trim memory if needed
        self._trim_memory(conversation_id)

        # Persist memory periodically
        await self._save_memory(conversation_id, user_id)

    def _evaluate_message_importance(self, message: str) -> float:
        """Evaluate the importance of a message for memory retention (0.0 to 1.0)"""
        # Basic importance score
        importance = 0.5

        # Check for high importance patterns
        for pattern in self.high_importance_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                importance = min(importance + 0.2, 1.0)

        # Length-based importance (longer messages might contain more info)
        if len(message) > 200:
            importance = min(importance + 0.1, 1.0)

        # Question importance
        if "?" in message:
            importance = min(importance + 0.1, 1.0)

        # Check for code blocks or structured data
        if "```" in message or re.search(r"\[.*?\]\(.*?\)", message):
            importance = min(importance + 0.2, 1.0)

        return importance

    def _extract_long_term_info(
        self, user_id: str, message: str, message_obj: Dict
    ) -> None:
        """Extract information for long-term memory"""
        # Extract user preferences
        preference_patterns = [
            (r"I prefer (.*?)(?:\.|\n|$)", "preference"),
            (r"I like (.*?)(?:\.|\n|$)", "preference"),
            (r"I want (.*?)(?:\.|\n|$)", "preference"),
            (r"I need (.*?)(?:\.|\n|$)", "need"),
            (r"My favorite (.*?) is (.*?)(?:\.|\n|$)", "favorite"),
            (r"I don\'t like (.*?)(?:\.|\n|$)", "dislike"),
        ]

        # Extract name information specifically - this is high priority
        name_match = self.name_pattern.search(message)
        if name_match:
            # Get the first non-None group (the actual name)
            name = next(
                (group for group in name_match.groups() if group is not None), None
            )
            if (
                name and len(name) > 1
            ):  # Ensure it's a reasonable name (more than one character)
                self.long_term_memory[user_id]["preferences"]["name"] = name
                # Also add this as a high-importance fact
                name_fact = {
                    "content": f"The user's name is {name}",
                    "timestamp": message_obj.get(
                        "timestamp", datetime.now().timestamp()
                    ),
                    "type": "personal_info",
                    "importance": 1.0,  # Maximum importance
                }
                # Add at the beginning of facts
                self.long_term_memory[user_id]["facts"].insert(0, name_fact)
                self.logger.info(f"Extracted user name '{name}' for user {user_id}")

        # Check for direct questions about name, set a flag to remember these
        if re.search(r"\bdo you (?:know|remember) my name\b", message, re.IGNORECASE):
            self.long_term_memory[user_id]["context_flags"] = self.long_term_memory[
                user_id
            ].get("context_flags", {})
            self.long_term_memory[user_id]["context_flags"]["asked_about_name"] = True

        for pattern, pref_type in preference_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                if pref_type == "favorite" and len(match.groups()) > 1:
                    category = match.group(1).strip()
                    value = match.group(2).strip()
                    self.long_term_memory[user_id]["preferences"][
                        f"favorite_{category}"
                    ] = value
                else:
                    preference = match.group(1).strip()
                    self.long_term_memory[user_id]["preferences"][
                        pref_type
                    ] = preference

        # Extract factual information
        if message_obj.get("importance", 0) > 0.7:
            # Only store high importance messages as facts
            truncated_msg = message[:200] + "..." if len(message) > 200 else message
            fact = {
                "content": truncated_msg,
                "timestamp": message_obj.get("timestamp", datetime.now().timestamp()),
                "type": message_obj.get("type", "text"),
                "importance": message_obj.get("importance", 0.7),
            }
            self.long_term_memory[user_id]["facts"].append(fact)

            # Limit facts to most recent/important 50
            if len(self.long_term_memory[user_id]["facts"]) > 50:
                self.long_term_memory[user_id]["facts"].sort(
                    key=lambda x: (x.get("importance", 0), x.get("timestamp", 0)),
                    reverse=True,
                )
                self.long_term_memory[user_id]["facts"] = self.long_term_memory[
                    user_id
                ]["facts"][:50]

    def _trim_memory(self, conversation_id: str) -> None:
        """Trim memory to stay within limits"""
        # Trim short-term memory
        if conversation_id in self.short_term_memory:
            if len(self.short_term_memory[conversation_id]) > self.short_term_limit:
                # Keep the most recent messages
                self.short_term_memory[conversation_id] = self.short_term_memory[
                    conversation_id
                ][-self.short_term_limit :]

        # Trim medium-term memory
        if conversation_id in self.medium_term_memory:
            if len(self.medium_term_memory[conversation_id]) > self.medium_term_limit:
                # Sort by importance and timestamp, keep most relevant
                messages = self.medium_term_memory[conversation_id]
                messages.sort(
                    key=lambda x: (x.get("importance", 0), x.get("timestamp", 0)),
                    reverse=True,
                )
                self.medium_term_memory[conversation_id] = messages[
                    : self.medium_term_limit
                ]

    async def get_conversation_context(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 10,
        include_long_term: bool = True,
    ) -> List[Dict]:
        """Get the conversation context with tiered memory integration, highlighting recent questions."""
        # Load memory if not already loaded
        await self._load_memory(conversation_id, user_id)

        # Combine contexts with priority to short-term
        context = []
        recent_user_questions = []  # Track recent questions

        # Add short-term memory (most recent conversations)
        short_term = self.short_term_memory.get(conversation_id, [])
        recent_short_term = short_term[-limit:]
        context.extend(recent_short_term)

        # Identify recent user questions from short-term memory
        for msg in reversed(recent_short_term):  # Check most recent first
            if msg.get("sender") == "user" and "?" in msg.get("content", ""):
                recent_user_questions.append(msg.get("content"))
                if (
                    len(recent_user_questions) >= 2
                ):  # Limit to last 2 questions for context summary
                    break
        recent_user_questions.reverse()  # Put them back in chronological order

        # Add relevant medium-term memory not in short-term
        medium_term = self.medium_term_memory.get(conversation_id, [])
        # Only add medium-term messages not already in the context
        short_term_timestamps = {msg.get("timestamp") for msg in context}
        medium_term_added_count = 0
        for msg in reversed(medium_term):  # Prioritize more recent medium-term
            if (
                msg.get("timestamp") not in short_term_timestamps
                and msg.get("importance", 0) >= 0.7
                and medium_term_added_count
                < (limit // 2)  # Limit medium-term additions
            ):
                context.append(msg)
                medium_term_added_count += 1

                # Limit how many medium-term messages we add overall
                if len(context) >= limit * 1.5:
                    break

        # Add long-term memory context if requested
        if include_long_term and user_id in self.long_term_memory:
            long_term = self.long_term_memory[user_id]

            # First check if there are any critical personal information to include (like name)
            personal_info = []
            user_name = long_term.get("preferences", {}).get("name")

            if user_name:
                personal_info.append(f"User's name is {user_name}")

            # Add any collected personal info at the top with high importance
            if personal_info:
                personal_info_text = "Important user information: " + "; ".join(
                    personal_info
                )
                context.insert(
                    0,
                    {
                        "sender": "system",
                        "content": personal_info_text,
                        "type": "personal_info",
                        "importance": 1.0,
                        "timestamp": time.time() - 10,  # Ensure it sorts early
                    },
                )

            # Add user preferences as context
            if long_term.get("preferences"):
                preferences = long_term["preferences"]
                if preferences:  # Only add if we have actual preferences
                    preference_text = "User preferences: " + ", ".join(
                        [
                            f"{k}: {v}" for k, v in preferences.items() if k != "name"
                        ]  # Exclude name, handled separately
                    )
                    if preference_text != "User preferences: ":  # Only add if not empty
                        context.insert(
                            (
                                0 if not personal_info else 1
                            ),  # Insert after personal info if it exists
                            {
                                "sender": "system",
                                "content": preference_text,
                                "type": "memory",
                                "importance": 0.9,
                                "timestamp": time.time() - 9,  # Ensure it sorts early
                            },
                        )

            # Check for specific context flags
            if long_term.get("context_flags", {}).get("asked_about_name"):
                # The user has previously asked if we know their name, prioritize this information
                if user_name:
                    context.insert(
                        0,
                        {
                            "sender": "system",
                            "content": f"IMPORTANT: User asked if you know their name. Their name is {user_name}.",
                            "type": "memory",
                            "importance": 1.0,
                            "timestamp": time.time() - 11,  # Ensure it sorts earliest
                        },
                    )
                    # Reset this flag after it's been addressed once
                    long_term["context_flags"]["asked_about_name"] = False
                    await self._save_memory("", user_id)

            # Add relevant facts from long-term memory
            if long_term.get("facts"):
                # Take the 5 most important facts, sorted by importance then recency
                important_facts = sorted(
                    long_term["facts"],
                    key=lambda x: (x.get("importance", 0), x.get("timestamp", 0)),
                    reverse=True,
                )[:5]

                fact_insert_index = 1  # Default insert index
                if personal_info:
                    fact_insert_index += 1
                if (
                    long_term.get("preferences")
                    and preference_text != "User preferences: "
                ):
                    fact_insert_index += 1

                for fact in important_facts:
                    # Skip personal info facts if we've already added personal info
                    if fact.get("type") == "personal_info" and personal_info:
                        continue

                    context.insert(
                        fact_insert_index,
                        {
                            "sender": "system",
                            "content": f"User previously mentioned: {fact['content']}",
                            "type": "memory",
                            "importance": fact.get("importance", 0.8),
                            "timestamp": fact.get(
                                "timestamp", time.time() - 8
                            ),  # Ensure it sorts early
                        },
                    )

        # Add a system message summarizing recent user questions if any were found
        if recent_user_questions:
            questions_summary = "\n".join([f'- "{q}"' for q in recent_user_questions])
            context.insert(
                0,  # Add near the beginning of the context
                {
                    "sender": "system",
                    "content": f"Context: The user's previous questions in this conversation include:\n{questions_summary}",
                    "type": "context_summary",
                    "importance": 0.95,  # High importance
                    "timestamp": time.time() - 5,  # Ensure it sorts relatively early
                },
            )

        # Sort context primarily by timestamp, but keep system messages near the top if needed
        # We added timestamps to system messages to help control sorting
        context.sort(key=lambda x: x.get("timestamp", 0))

        # Ensure the context doesn't exceed a reasonable limit (e.g., limit * 2)
        final_context = context[-(limit * 2) :]

        # Reformat for model consumption (role/content)
        model_context = []
        for msg in final_context:
            role = "user" if msg.get("sender") == "user" else "assistant"
            if msg.get("sender") == "system":
                role = "system"  # Or potentially map to 'user' or 'assistant' depending on model needs

            # Ensure content exists and is a string
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)  # Convert non-string content

            # Skip empty messages
            if not content.strip():
                continue

            model_context.append({"role": role, "content": content})

        # Gemini specific formatting: alternate user/assistant roles, merge system prompts
        formatted_model_context = []
        system_prompts = []
        last_role = None

        for msg in model_context:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_prompts.append(content)
                continue

            # Merge consecutive messages from the same role if needed (optional, Gemini handles alternation well)
            # if formatted_model_context and role == last_role:
            #    formatted_model_context[-1]["content"] += "\n" + content
            #    continue

            # Add message
            formatted_model_context.append({"role": role, "content": content})
            last_role = role

        # Prepend system prompts if any
        if system_prompts:
            # Gemini prefers system instructions at the start or potentially as part of the first user message
            # Let's add it as a separate system message if the model supports it,
            # otherwise prepend to the first user message or handle according to model docs.
            # For now, let's assume a separate system message is okay or handled by the caller.
            # We will return it separately or integrate based on the calling function's needs.
            # This function's primary goal is context assembly.
            # The calling function (`generate_response`) should handle final formatting for Gemini.

            # Returning the raw context list including system messages.
            # The calling code will format it for the specific model.
            pass  # System messages are already included in model_context

        # Return the assembled context, ready for final formatting by the caller
        return model_context  # Return the list of dicts with role/content

    async def search_memory(
        self, user_id: str, query: str, limit: int = 5
    ) -> List[Dict]:
        """Search memory for relevant messages using keyword matching"""
        results = []
        query_terms = set(query.lower().split())

        # Find conversations for this user
        user_conversations = set()
        for conv_id, messages in self.medium_term_memory.items():
            for message in messages:
                message_user_id = message.get("metadata", {}).get("user_id")
                if message_user_id == user_id:
                    user_conversations.add(conv_id)
                    break

        # Search through medium-term memory for this user
        for conv_id in user_conversations:
            for message in self.medium_term_memory.get(conv_id, []):
                content = message.get("content", "").lower()
                score = 0

                # Calculate simple relevance score
                for term in query_terms:
                    if term in content:
                        score += 1

                # Adjust for importance
                score = score * (1 + message.get("importance", 0))

                if score > 0:
                    result = dict(message)
                    result["relevance_score"] = score
                    result["conversation_id"] = conv_id
                    results.append(result)

        # Sort by relevance and limit results
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return results[:limit]

    async def get_document_context(self, user_id: str, limit: int = 3) -> List[Dict]:
        """Get recent document interactions for a user"""
        document_contexts = []

        # Find conversations for this user
        user_conversations = set()
        for conv_id, messages in self.medium_term_memory.items():
            for message in messages:
                message_user_id = message.get("metadata", {}).get("user_id")
                if message_user_id == user_id:
                    user_conversations.add(conv_id)
                    break

        # Collect document-related messages
        for conv_id in user_conversations:
            for message in self.medium_term_memory.get(conv_id, []):
                if message.get("type") == "document" or "document" in message.get(
                    "metadata", {}
                ).get("type", ""):

                    # Create document context entry
                    doc_context = {
                        "content": message.get("content", ""),
                        "document_id": message.get("metadata", {}).get(
                            "document_id", "unknown"
                        ),
                        "timestamp": message.get("timestamp", 0),
                        "conversation_id": conv_id,
                    }
                    document_contexts.append(doc_context)

        # Sort by timestamp (most recent first) and limit
        document_contexts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        return document_contexts[:limit]

    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a specific conversation from memory"""
        try:
            if conversation_id in self.short_term_memory:
                del self.short_term_memory[conversation_id]

            if conversation_id in self.medium_term_memory:
                del self.medium_term_memory[conversation_id]

            # Remove conversation file if exists
            file_path = os.path.join(
                self.storage_path, f"conversation_{conversation_id}.json"
            )
            if os.path.exists(file_path):
                os.remove(file_path)

            return True
        except Exception as e:
            self.logger.error(
                f"Error clearing conversation {conversation_id}: {str(e)}"
            )
            return False

    async def clear_user_data(self, user_id: str) -> bool:
        """Clear all data for a specific user"""
        try:
            # Clear from long-term memory
            if user_id in self.long_term_memory:
                del self.long_term_memory[user_id]

            # Find and clear conversations for this user
            user_conversations = set()
            for conv_id, messages in list(self.medium_term_memory.items()):
                for message in messages:
                    message_user_id = message.get("metadata", {}).get("user_id")
                    if message_user_id == user_id:
                        user_conversations.add(conv_id)
                        break

            # Clear found conversations
            for conv_id in user_conversations:
                await self.clear_conversation(conv_id)

            # Remove user file if exists
            file_path = os.path.join(self.storage_path, f"user_{user_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)

            return True
        except Exception as e:
            self.logger.error(f"Error clearing user data {user_id}: {str(e)}")
            return False

    async def _save_memory(self, conversation_id: str, user_id: str) -> None:
        """Save memory to persistent storage"""
        try:
            # Save conversation memory
            if conversation_id:
                conversation_data = {
                    "short_term": self.short_term_memory.get(conversation_id, []),
                    "medium_term": self.medium_term_memory.get(conversation_id, []),
                    "last_updated": datetime.now().isoformat(),
                }

                file_path = os.path.join(
                    self.storage_path, f"conversation_{conversation_id}.json"
                )
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json.dumps(conversation_data, indent=2))

            # Save user long-term memory
            if user_id and user_id in self.long_term_memory:
                file_path = os.path.join(self.storage_path, f"user_{user_id}.json")
                async with aiofiles.open(file_path, "w") as f:
                    await f.write(json.dumps(self.long_term_memory[user_id], indent=2))

            # Save to database if available
            if self.db is not None:
                try:
                    if conversation_id:
                        conversation_data = {
                            "conversation_id": conversation_id,
                            "short_term": self.short_term_memory.get(
                                conversation_id, []
                            ),
                            "medium_term": self.medium_term_memory.get(
                                conversation_id, []
                            ),
                            "last_updated": datetime.now(),
                        }

                        await asyncio.to_thread(
                            self.db.conversations.update_one,
                            {"conversation_id": conversation_id},
                            {"$set": conversation_data},
                            upsert=True,
                        )

                    if user_id and user_id in self.long_term_memory:
                        user_data = {
                            "user_id": user_id,
                            "long_term_memory": self.long_term_memory[user_id],
                            "last_updated": datetime.now(),
                        }

                        await asyncio.to_thread(
                            self.db.user_memory.update_one,
                            {"user_id": user_id},
                            {"$set": user_data},
                            upsert=True,
                        )
                except Exception as db_error:
                    self.logger.error(f"Failed to save to database: {str(db_error)}")

        except Exception as e:
            self.logger.error(f"Error saving memory: {str(e)}")

    async def _load_memory(self, conversation_id: str, user_id: str) -> None:
        """Load memory from persistent storage if not already loaded"""
        try:
            # Load conversation memory if not already loaded
            if conversation_id and (
                conversation_id not in self.short_term_memory
                or conversation_id not in self.medium_term_memory
            ):
                # Try loading from file
                file_path = os.path.join(
                    self.storage_path, f"conversation_{conversation_id}.json"
                )
                if os.path.exists(file_path):
                    async with aiofiles.open(file_path, "r") as f:
                        conversation_data = json.loads(await f.read())

                        self.short_term_memory[conversation_id] = conversation_data.get(
                            "short_term", []
                        )
                        self.medium_term_memory[conversation_id] = (
                            conversation_data.get("medium_term", [])
                        )

                        self.logger.info(
                            f"Loaded conversation {conversation_id} from file storage"
                        )

                # If not found in file, try database
                elif self.db is not None:
                    conversation_doc = await asyncio.to_thread(
                        self.db.conversations.find_one,
                        {"conversation_id": conversation_id},
                    )

                    if conversation_doc:
                        self.short_term_memory[conversation_id] = conversation_doc.get(
                            "short_term", []
                        )
                        self.medium_term_memory[conversation_id] = conversation_doc.get(
                            "medium_term", []
                        )

                        self.logger.info(
                            f"Loaded conversation {conversation_id} from database"
                        )

            # Load user long-term memory if not already loaded
            if user_id and user_id not in self.long_term_memory:
                # Try loading from file
                file_path = os.path.join(self.storage_path, f"user_{user_id}.json")
                if os.path.exists(file_path):
                    async with aiofiles.open(file_path, "r") as f:
                        self.long_term_memory[user_id] = json.loads(await f.read())

                        self.logger.info(
                            f"Loaded user {user_id} memory from file storage"
                        )

                # If not found in file, try database
                elif self.db is not None:
                    user_doc = await asyncio.to_thread(
                        self.db.user_memory.find_one, {"user_id": user_id}
                    )

                    if user_doc and "long_term_memory" in user_doc:
                        self.long_term_memory[user_id] = user_doc["long_term_memory"]

                        self.logger.info(f"Loaded user {user_id} memory from database")

                # Initialize if not found anywhere
                if user_id not in self.long_term_memory:
                    self.long_term_memory[user_id] = {
                        "preferences": {},
                        "facts": [],
                        "contexts": {},
                    }

        except Exception as e:
            self.logger.error(f"Error loading memory: {str(e)}")
            # Initialize with empty structure
            if conversation_id:
                self.short_term_memory[conversation_id] = []
                self.medium_term_memory[conversation_id] = []

            if user_id and user_id not in self.long_term_memory:
                self.long_term_memory[user_id] = {
                    "preferences": {},
                    "facts": [],
                    "contexts": {},
                }

    async def update_user_preference(self, user_id: str, key: str, value: Any) -> None:
        """Update a user preference in long-term memory"""
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = {
                "preferences": {},
                "facts": [],
                "contexts": {},
            }

        self.long_term_memory[user_id]["preferences"][key] = value

        # Save the update
        await self._save_memory("", user_id)

    async def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get all user preferences from long-term memory"""
        # Ensure memory is loaded
        await self._load_memory("", user_id)

        if user_id in self.long_term_memory:
            return self.long_term_memory[user_id].get("preferences", {})
        return {}

    async def get_memory_text(self, user_id: str) -> str:
        """
        Generate a human-readable text summary of what the bot remembers about the user.
        This is useful for when users ask what the bot knows/remembers about them.

        Args:
            user_id: The user ID to retrieve memory for

        Returns:
            Formatted text describing user memory
        """
        # Ensure memory is loaded
        await self._load_memory("", user_id)

        if user_id not in self.long_term_memory:
            return "I don't have any specific information saved about you yet."

        memory_text = []
        user_memory = self.long_term_memory[user_id]

        # Add name if it exists (most important)
        if "name" in user_memory.get("preferences", {}):
            name = user_memory["preferences"]["name"]
            memory_text.append(
                f"I remember your name is {name}."
            )  # Changed phrasing slightly
        else:
            memory_text.append("I don't know your name yet.")

        # Add other preferences
        other_prefs = {}
        for k, v in user_memory.get("preferences", {}).items():
            if k != "name":  # Skip name, already added
                other_prefs[k] = v

        if other_prefs:
            memory_text.append(
                "\nRegarding your preferences, I remember:"
            )  # Improved section header
            for pref_type, value in other_prefs.items():
                # Format the preference type for better readability
                formatted_type = pref_type.replace("_", " ")
                memory_text.append(
                    f"- Your {formatted_type} preference is: {value}"
                )  # Clearer phrasing

        # Add important facts (up to 5)
        important_facts = sorted(
            user_memory.get("facts", []),
            key=lambda x: (x.get("importance", 0), x.get("timestamp", 0)),
            reverse=True,
        )[:5]

        if important_facts:
            memory_text.append(
                "\nI also recall these key details from our conversations:"
            )  # Improved section header
            for fact in important_facts:
                # Skip facts that are just about the user's name if already mentioned
                if (
                    fact.get("type") == "personal_info"
                    and "user's name is" in fact.get("content", "")
                    and "name" in user_memory.get("preferences", {})
                ):
                    continue

                content = fact.get("content", "").strip()
                if content:
                    # Remove quotes and normalize text
                    content = content.strip("\"'")
                    # Make it sound more like recalling a fact
                    memory_text.append(f'- You mentioned: "{content}"')

        # Handle case where we don't have much information beyond maybe the name
        if not other_prefs and not important_facts:
            if "name" in user_memory.get("preferences", {}):
                memory_text.append(
                    "\nBeyond your name, we haven't discussed specific preferences or details for me to remember yet."
                )
            else:
                # This case is covered by the initial check, but added for robustness
                memory_text = [
                    "I don't have any specific information saved about you yet."
                ]

        return "\n".join(memory_text)

    async def check_name_memory(
        self, user_id: str, conversation_id: str, message: str
    ) -> Optional[str]:
        """
        Special handler for when users ask if the bot knows their name.

        Args:
            user_id: The user ID
            conversation_id: The conversation ID
            message: The user's message

        Returns:
            Response about the user's name if pattern matches, None otherwise
        """
        # Check if this is a question about the bot knowing the user's name
        name_question_patterns = [
            r"(?:do you|can you|could you)?\s*(?:know|remember|recall)\s+my\s+name",
            r"what(?:'s| is) my name",
            r"who am i",
        ]

        is_name_question = any(
            re.search(pattern, message, re.IGNORECASE)
            for pattern in name_question_patterns
        )

        if is_name_question:
            # Check if we have the user's name
            await self._load_memory(conversation_id, user_id)

            if user_id in self.long_term_memory:
                name = self.long_term_memory[user_id].get("preferences", {}).get("name")

                if name:
                    # Set the flag so the model knows we've addressed this
                    context_flags = self.long_term_memory[user_id].get(
                        "context_flags", {}
                    )
                    self.long_term_memory[user_id]["context_flags"] = context_flags
                    self.long_term_memory[user_id]["context_flags"][
                        "asked_about_name"
                    ] = False
                    await self._save_memory("", user_id)

                    return f"Yes, I remember your name is {name}!"
                else:
                    return "I don't know your name yet. Would you like to tell me?"

        return None

    async def handle_name_introduction(
        self, user_id: str, conversation_id: str, message: str
    ) -> Optional[str]:
        """
        Handle cases where user is introducing their name for the first time.

        Args:
            user_id: The user ID
            conversation_id: The conversation ID
            message: The user's message

        Returns:
            Confirmation response if name was extracted, None otherwise
        """
        # Extract name from introduction patterns
        introduction_patterns = [
            (
                r"(?:my name is|i am|i'm|call me)\s+(\w+)(?:\.|\s|$)",
                1,
            ),  # Group 1 contains the name
            (r"(?:i'm|i am)\s+called\s+(\w+)(?:\.|\s|$)", 1),
            (r"(?:you can|please)\s+call\s+me\s+(\w+)(?:\.|\s|$)", 1),
        ]

        name = None
        for pattern, group in introduction_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match and match.group(group):
                name = match.group(group).strip()
                # Capitalize first letter
                name = name[0].upper() + name[1:] if len(name) > 1 else name.upper()
                break

        if name and len(name) > 1:  # Ensure it's a reasonable name
            # Store the name
            if user_id not in self.long_term_memory:
                self.long_term_memory[user_id] = {
                    "preferences": {},
                    "facts": [],
                    "contexts": {},
                    "context_flags": {},
                }

            self.long_term_memory[user_id]["preferences"]["name"] = name

            # Add as a high-importance fact
            name_fact = {
                "content": f"The user's name is {name}",
                "timestamp": time.time(),
                "type": "personal_info",
                "importance": 1.0,  # Maximum importance
            }

            # Add or replace existing name fact
            facts = self.long_term_memory[user_id].get("facts", [])
            # Remove any existing name facts
            facts = [f for f in facts if "user's name" not in f.get("content", "")]
            # Add the new name fact at the beginning
            facts.insert(0, name_fact)
            self.long_term_memory[user_id]["facts"] = facts

            # Save immediately
            await self._save_memory(conversation_id, user_id)

            return f"Nice to meet you, {name}! I'll remember your name."

        return None
