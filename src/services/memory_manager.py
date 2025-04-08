import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import asyncio
import re
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


import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import aiofiles
import os
import re
from collections import defaultdict


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
            r"\bmy name is\b|\bi am called\b|\bplease call me\b",
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
        ]

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
        """Get the conversation context with tiered memory integration"""
        # Load memory if not already loaded
        await self._load_memory(conversation_id, user_id)

        # Combine contexts with priority to short-term
        context = []

        # Add short-term memory (most recent conversations)
        short_term = self.short_term_memory.get(conversation_id, [])
        context.extend(short_term[-limit:])

        # Add relevant medium-term memory not in short-term
        medium_term = self.medium_term_memory.get(conversation_id, [])
        # Only add medium-term messages not already in the context
        short_term_timestamps = {msg.get("timestamp") for msg in context}
        for msg in medium_term:
            if (
                msg.get("timestamp") not in short_term_timestamps
                and msg.get("importance", 0) >= 0.7
            ):
                context.append(msg)

                # Limit how many medium-term messages we add
                if len(context) >= limit * 1.5:
                    break

        # Add long-term memory context if requested
        if include_long_term and user_id in self.long_term_memory:
            long_term = self.long_term_memory[user_id]

            # Add user preferences as context
            if long_term.get("preferences"):
                preferences = long_term["preferences"]
                preference_text = "User preferences: " + ", ".join(
                    [f"{k}: {v}" for k, v in preferences.items()]
                )
                context.insert(
                    0,
                    {
                        "sender": "system",
                        "content": preference_text,
                        "type": "memory",
                        "importance": 0.9,
                    },
                )

            # Add relevant facts from long-term memory
            if long_term.get("facts"):
                # Take the 3 most recent facts
                recent_facts = sorted(
                    long_term["facts"],
                    key=lambda x: x.get("timestamp", 0),
                    reverse=True,
                )[:3]

                for fact in recent_facts:
                    context.insert(
                        1,
                        {  # Insert after preferences
                            "sender": "system",
                            "content": f"User previously mentioned: {fact['content']}",
                            "type": "memory",
                            "importance": 0.8,
                        },
                    )

        # Sort by timestamp for chronological order
        context.sort(key=lambda x: x.get("timestamp", 0))

        return context

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
