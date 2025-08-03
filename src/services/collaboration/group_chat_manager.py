"""
👥 Advanced Group Chat Integration System
Enables seamless team collaboration with shared memory, intelligent context, and rich UI
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatMember
from telegram.ext import ContextTypes
from telegram.constants import ChatType
import time

logger = logging.getLogger(__name__)


class GroupChatManager:
    """Advanced group chat management with collaborative features"""

    def __init__(self, conversation_manager, enhanced_ui_components):
        self.conversation_manager = conversation_manager
        self.ui_components = enhanced_ui_components
        self.logger = logging.getLogger(__name__)

        # Group management
        self.active_groups = {}  # group_id -> group_info
        self.group_settings = {}  # group_id -> settings
        self.group_roles = {}  # group_id -> {user_id: role}
        self.shared_workspaces = {}  # workspace_id -> workspace_info

        # Collaboration features
        self.active_discussions = {}  # discussion_id -> discussion_info
        self.group_tasks = {}  # group_id -> [tasks]
        self.knowledge_base = {}  # group_id -> knowledge_items

        # Real-time collaboration
        self.typing_indicators = {}  # group_id -> {user_id: timestamp}
        self.activity_streams = {}  # group_id -> activity_log

    async def initialize_group(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> Dict[str, Any]:
        """Initialize a group for enhanced collaboration"""
        chat = update.effective_chat
        user = update.effective_user

        if not chat or chat.type not in [ChatType.GROUP, ChatType.SUPERGROUP]:
            return {
                "success": False,
                "error": "This feature is only available in groups",
            }

        group_id = str(chat.id)
        user_id = user.id if user else None

        # Check if user is admin
        try:
            member = await context.bot.get_chat_member(chat.id, user_id)
            if member.status not in [ChatMember.ADMINISTRATOR, ChatMember.OWNER]:
                return {
                    "success": False,
                    "error": "Only group administrators can initialize enhanced features",
                }
        except Exception as e:
            self.logger.error(f"Error checking admin status: {e}")
            return {"success": False, "error": "Unable to verify permissions"}

        # Initialize group settings
        group_info = {
            "group_id": group_id,
            "group_name": chat.title or "Unknown Group",
            "initialized_by": user_id,
            "initialization_time": datetime.now().isoformat(),
            "member_count": await self._get_member_count(chat, context),
            "features_enabled": {
                "shared_memory": True,
                "collaborative_notes": True,
                "task_management": True,
                "smart_summaries": True,
                "activity_tracking": True,
            },
            "ai_personality": "collaborative_assistant",
        }

        self.active_groups[group_id] = group_info

        # Initialize default settings
        self.group_settings[group_id] = {
            "auto_summarize": True,
            "smart_notifications": True,
            "context_awareness": True,
            "collaboration_mode": "enhanced",
            "privacy_level": "group_only",
            "ai_proactivity": "moderate",
        }

        # Start group session
        session_info = await self.conversation_manager.start_group_session(
            group_id, user_id, "Group Initialization"
        )

        # Create welcome message with enhanced UI
        welcome_message = self._create_group_welcome_message(group_info)
        welcome_keyboard = self._create_group_setup_keyboard(group_id)

        await update.message.reply_text(
            welcome_message, reply_markup=welcome_keyboard, parse_mode="Markdown"
        )

        # Log initialization
        await self._log_group_activity(
            group_id,
            user_id,
            "group_initialized",
            {"session_id": session_info.get("session_id")},
        )

        return {"success": True, "group_info": group_info, "session": session_info}

    async def handle_group_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> Dict[str, Any]:
        """Handle messages in enhanced group chats with intelligent features"""
        chat = update.effective_chat
        user = update.effective_user
        message = update.message

        if not chat or not user or not message:
            return {"success": False, "error": "Invalid message data"}

        group_id = str(chat.id)
        user_id = user.id
        message_text = message.text or ""

        # Check if group is initialized
        if group_id not in self.active_groups:
            return {
                "success": False,
                "error": "Group not initialized for enhanced features",
            }

        # Process message with AI intelligence
        processing_result = await self._process_group_message_intelligently(
            group_id, user_id, message_text, message, context
        )

        # Handle special commands and mentions
        if message_text.startswith("/") or self._is_bot_mentioned(
            message_text, context
        ):
            response_result = await self._handle_group_ai_interaction(
                group_id, user_id, message_text, context
            )
            processing_result.update(response_result)

        # Update group activity
        await self._update_group_activity(group_id, user_id, message_text)

        # Check for collaboration triggers
        await self._check_collaboration_triggers(
            group_id, user_id, message_text, context
        )

        return processing_result

    async def _process_group_message_intelligently(
        self, group_id: str, user_id: int, message_text: str, message, context
    ) -> Dict[str, Any]:
        """Process group messages with enhanced intelligence"""

        # Extract message context
        message_context = {
            "reply_to": (
                message.reply_to_message.text if message.reply_to_message else None
            ),
            "media_type": self._detect_media_type(message),
            "timestamp": (
                message.date.isoformat() if message.date else datetime.now().isoformat()
            ),
            "message_id": message.message_id,
            "user_mention": self._extract_mentions(message_text),
            "hashtags": self._extract_hashtags(message_text),
            "urls": self._extract_urls(message_text),
        }

        # Analyze message intent
        intent_analysis = await self._analyze_message_intent(
            message_text, message_context
        )

        # Save to group memory with enhanced metadata
        await self.conversation_manager.save_message_pair(
            user_id=user_id,
            user_message=message_text,
            assistant_message="",  # Will be filled if AI responds
            group_id=group_id,
            message_context={
                **message_context,
                "intent": intent_analysis,
                "group_context": await self._get_current_group_context(group_id),
            },
        )

        # Update shared knowledge base
        await self._update_group_knowledge_base(group_id, message_text, intent_analysis)

        return {
            "success": True,
            "intent": intent_analysis,
            "context": message_context,
            "requires_response": intent_analysis.get("needs_ai_response", False),
        }

    async def _handle_group_ai_interaction(
        self, group_id: str, user_id: int, message_text: str, context
    ) -> Dict[str, Any]:
        """Handle AI interactions in group context"""

        # Get intelligent group context
        group_context = await self.conversation_manager.get_intelligent_context(
            user_id, message_text, group_id
        )

        # Generate contextual response
        if message_text.startswith("/"):
            response = await self._handle_group_command(
                group_id, user_id, message_text, context
            )
        else:
            response = await self._generate_contextual_group_response(
                group_id, user_id, message_text, group_context, context
            )

        return response

    async def _handle_group_command(
        self, group_id: str, user_id: int, command: str, context
    ) -> Dict[str, Any]:
        """Handle special group commands"""

        command_parts = command.lower().split()
        base_command = command_parts[0]

        if base_command == "/groupsummary":
            return await self._generate_group_summary(group_id, context)

        elif base_command == "/sharedknowledge":
            return await self._show_shared_knowledge(group_id, context)

        elif base_command == "/collaboratenote":
            note_content = (
                " ".join(command_parts[1:]) if len(command_parts) > 1 else None
            )
            return await self._create_collaborative_note(
                group_id, user_id, note_content, context
            )

        elif base_command == "/grouptasks":
            return await self._show_group_tasks(group_id, context)

        elif base_command == "/startdiscussion":
            topic = (
                " ".join(command_parts[1:])
                if len(command_parts) > 1
                else "General Discussion"
            )
            return await self._start_group_discussion(group_id, user_id, topic, context)

        elif base_command == "/groupstats":
            return await self._show_group_statistics(group_id, context)

        elif base_command == "/smartsearch":
            query = " ".join(command_parts[1:]) if len(command_parts) > 1 else ""
            return await self._perform_group_smart_search(group_id, query, context)

        else:
            return {"success": False, "error": f"Unknown command: {base_command}"}

    async def _generate_group_summary(self, group_id: str, context) -> Dict[str, Any]:
        """Generate intelligent group conversation summary"""

        try:
            # Get group activity summary
            activity_summary = await self.conversation_manager.memory_manager.get_group_activity_summary(
                group_id
            )

            # Get recent important messages
            recent_memory = await self.conversation_manager.get_short_term_memory(
                user_id=0, limit=20, group_id=group_id
            )

            # Create rich summary
            summary_text = "📊 **Group Conversation Summary**\n\n"

            # Activity metrics
            summary_text += "📈 **Activity Metrics:**\n"
            summary_text += (
                f"• Total messages: {activity_summary.get('total_messages', 0)}\n"
            )
            summary_text += (
                f"• Active participants: {activity_summary.get('active_users', 0)}\n"
            )
            summary_text += f"• Most active: User {activity_summary.get('most_active_user', 'N/A')}\n\n"

            # Key topics
            if self.knowledge_base.get(group_id):
                summary_text += "🎯 **Key Topics Discussed:**\n"
                for topic in list(self.knowledge_base[group_id].keys())[:5]:
                    summary_text += f"• {topic}\n"
                summary_text += "\n"

            # Recent highlights
            important_messages = [
                msg for msg in recent_memory if msg.get("importance", 0) > 0.7
            ]
            if important_messages:
                summary_text += "⭐ **Recent Highlights:**\n"
                for msg in important_messages[:3]:
                    content = msg.get("content", "")[:100]
                    summary_text += f"• {content}...\n"
                summary_text += "\n"

            # Collaboration insights
            if group_id in self.conversation_manager.collaborative_contexts:
                collab_context = self.conversation_manager.collaborative_contexts[
                    group_id
                ]
                summary_text += "🤝 **Collaboration Status:**\n"
                summary_text += (
                    f"• Active notes: {len(collab_context.get('notes', []))}\n"
                )
                summary_text += (
                    f"• Action items: {len(collab_context.get('action_items', []))}\n"
                )
                summary_text += (
                    f"• Decisions made: {len(collab_context.get('decisions', []))}\n"
                )

            # Create summary keyboard
            summary_keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "📝 Export Summary",
                            callback_data=f"export_summary_{group_id}",
                        ),
                        InlineKeyboardButton(
                            "📊 Detailed Stats",
                            callback_data=f"detailed_stats_{group_id}",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "🔍 Smart Search", callback_data=f"smart_search_{group_id}"
                        ),
                        InlineKeyboardButton(
                            "🤝 Collaboration Hub",
                            callback_data=f"collab_hub_{group_id}",
                        ),
                    ],
                ]
            )

            # Send summary
            await context.bot.send_message(
                chat_id=group_id,
                text=summary_text,
                reply_markup=summary_keyboard,
                parse_mode="Markdown",
            )

            return {"success": True, "summary_sent": True}

        except Exception as e:
            self.logger.error(f"Error generating group summary: {e}")
            return {"success": False, "error": str(e)}

    async def _create_collaborative_note(
        self, group_id: str, user_id: int, note_content: Optional[str], context
    ) -> Dict[str, Any]:
        """Create or prompt for collaborative note"""

        if not note_content:
            # Prompt user to provide note content
            prompt_keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "📝 Add Note",
                            callback_data=f"add_note_{group_id}_{user_id}",
                        )
                    ],
                    [
                        InlineKeyboardButton(
                            "📋 Meeting Notes",
                            callback_data=f"note_meeting_{group_id}_{user_id}",
                        ),
                        InlineKeyboardButton(
                            "💡 Idea", callback_data=f"note_idea_{group_id}_{user_id}"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "✅ Action Item",
                            callback_data=f"note_action_{group_id}_{user_id}",
                        ),
                        InlineKeyboardButton(
                            "🎯 Decision",
                            callback_data=f"note_decision_{group_id}_{user_id}",
                        ),
                    ],
                ]
            )

            await context.bot.send_message(
                chat_id=group_id,
                text="📝 **Create Collaborative Note**\n\nChoose note type or reply with your note content:",
                reply_markup=prompt_keyboard,
                parse_mode="Markdown",
            )

            return {"success": True, "awaiting_input": True}

        # Create the note
        await self.conversation_manager.add_collaborative_note(
            group_id, user_id, note_content, "general"
        )

        # Notify group
        note_message = "📝 **Collaborative Note Added**\n\n"
        note_message += f"👤 By: User {user_id}\n"
        note_message += f"📄 Content: {note_content}\n"
        note_message += f"🕐 Time: {datetime.now().strftime('%H:%M')}"

        note_keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "👍 Helpful", callback_data=f"note_react_helpful_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "💬 Discuss", callback_data=f"note_discuss_{group_id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "📚 View All Notes", callback_data=f"view_notes_{group_id}"
                    )
                ],
            ]
        )

        await context.bot.send_message(
            chat_id=group_id,
            text=note_message,
            reply_markup=note_keyboard,
            parse_mode="Markdown",
        )

        return {"success": True, "note_created": True}

    async def _start_group_discussion(
        self, group_id: str, user_id: int, topic: str, context
    ) -> Dict[str, Any]:
        """Start a structured group discussion"""

        discussion_id = f"disc_{group_id}_{int(time.time())}"

        discussion_info = {
            "discussion_id": discussion_id,
            "group_id": group_id,
            "topic": topic,
            "started_by": user_id,
            "start_time": datetime.now().isoformat(),
            "participants": {user_id},
            "messages": [],
            "status": "active",
            "structured_mode": True,
        }

        self.active_discussions[discussion_id] = discussion_info

        # Create discussion announcement
        discussion_message = "🗣️ **New Discussion Started**\n\n"
        discussion_message += f"📋 **Topic:** {topic}\n"
        discussion_message += f"👤 **Started by:** User {user_id}\n"
        discussion_message += f"🕐 **Time:** {datetime.now().strftime('%H:%M')}\n\n"
        discussion_message += "Join the discussion by responding below!"

        discussion_keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "🙋‍♂️ Join Discussion",
                        callback_data=f"join_disc_{discussion_id}",
                    ),
                    InlineKeyboardButton(
                        "📊 Discussion Stats",
                        callback_data=f"disc_stats_{discussion_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "📝 Add Point", callback_data=f"disc_point_{discussion_id}"
                    ),
                    InlineKeyboardButton(
                        "🏁 End Discussion", callback_data=f"end_disc_{discussion_id}"
                    ),
                ],
            ]
        )

        await context.bot.send_message(
            chat_id=group_id,
            text=discussion_message,
            reply_markup=discussion_keyboard,
            parse_mode="Markdown",
        )

        return {
            "success": True,
            "discussion_started": True,
            "discussion_id": discussion_id,
        }

    async def _perform_group_smart_search(
        self, group_id: str, query: str, context
    ) -> Dict[str, Any]:
        """Perform intelligent search across group conversations"""

        if not query:
            # Show search interface
            search_keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🔍 Search Messages",
                            callback_data=f"search_messages_{group_id}",
                        ),
                        InlineKeyboardButton(
                            "📝 Search Notes", callback_data=f"search_notes_{group_id}"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "👥 Search by User", callback_data=f"search_user_{group_id}"
                        ),
                        InlineKeyboardButton(
                            "📅 Search by Date", callback_data=f"search_date_{group_id}"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "🎯 Search Topics",
                            callback_data=f"search_topics_{group_id}",
                        ),
                        InlineKeyboardButton(
                            "📊 Search Stats", callback_data=f"search_stats_{group_id}"
                        ),
                    ],
                ]
            )

            await context.bot.send_message(
                chat_id=group_id,
                text="🔍 **Smart Group Search**\n\nWhat would you like to search for?",
                reply_markup=search_keyboard,
                parse_mode="Markdown",
            )

            return {"success": True, "search_interface_shown": True}

        # Perform the search
        try:
            search_results = (
                await self.conversation_manager.memory_manager.get_relevant_memory(
                    "",
                    query,
                    limit=10,
                    is_group=True,
                    group_id=group_id,
                    include_group_knowledge=True,
                )
            )

            if not search_results:
                await context.bot.send_message(
                    chat_id=group_id,
                    text=f"🔍 No results found for: `{query}`",
                    parse_mode="Markdown",
                )
                return {"success": True, "results_found": False}

            # Format search results
            results_text = f"🔍 **Search Results for:** `{query}`\n\n"

            for i, result in enumerate(search_results[:5], 1):
                content = result.get("content", "")
                timestamp = result.get("timestamp", 0)
                user_id = result.get("user_id", "N/A")
                importance = result.get("importance", 0)

                # Format timestamp
                if timestamp:
                    try:
                        dt = datetime.fromtimestamp(timestamp)
                        time_str = dt.strftime("%m/%d %H:%M")
                    except:
                        time_str = "Unknown"
                else:
                    time_str = "Unknown"

                # Importance indicator
                importance_indicator = "⭐" if importance > 0.7 else "📄"

                results_text += f"{importance_indicator} **Result {i}**\n"
                results_text += f"👤 User {user_id} • 🕐 {time_str}\n"
                results_text += (
                    f"💬 {content[:150]}{'...' if len(content) > 150 else ''}\n\n"
                )

            # Create results keyboard
            results_keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "📊 More Results",
                            callback_data=f"more_results_{group_id}_{query}",
                        ),
                        InlineKeyboardButton(
                            "💾 Save Search",
                            callback_data=f"save_search_{group_id}_{query}",
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "🔍 New Search", callback_data=f"new_search_{group_id}"
                        )
                    ],
                ]
            )

            await context.bot.send_message(
                chat_id=group_id,
                text=results_text,
                reply_markup=results_keyboard,
                parse_mode="Markdown",
            )

            return {
                "success": True,
                "results_found": True,
                "result_count": len(search_results),
            }

        except Exception as e:
            self.logger.error(f"Error performing group search: {e}")
            return {"success": False, "error": str(e)}

    def _create_group_welcome_message(self, group_info: Dict[str, Any]) -> str:
        """Create a rich welcome message for newly initialized groups"""

        message = "🎉 **Welcome to Enhanced Group AI!**\n\n"
        message += f"👥 **Group:** {group_info['group_name']}\n"
        message += f"👤 **Initialized by:** User {group_info['initialized_by']}\n"
        message += f"📊 **Members:** {group_info['member_count']}\n\n"

        message += "🚀 **New Features Activated:**\n"
        message += "• 🧠 **Shared Group Memory** - I remember our conversations\n"
        message += (
            "• 📝 **Collaborative Notes** - Create shared notes and action items\n"
        )
        message += "• 🔍 **Smart Search** - Find information across all conversations\n"
        message += "• 📊 **Group Analytics** - Track activity and insights\n"
        message += "• 🤝 **Team Coordination** - Enhanced collaboration tools\n\n"

        message += "💡 **Quick Commands:**\n"
        message += "`/groupsummary` - Get conversation summary\n"
        message += "`/collaboratenote` - Create shared note\n"
        message += "`/smartsearch <query>` - Search conversations\n"
        message += "`/grouptasks` - Manage team tasks\n\n"

        message += "Let's make teamwork smarter! 🚀"

        return message

    def _create_group_setup_keyboard(self, group_id: str) -> InlineKeyboardMarkup:
        """Create setup keyboard for new groups"""

        return InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "⚙️ Group Settings", callback_data=f"group_settings_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "👥 Member Roles", callback_data=f"member_roles_{group_id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🧠 AI Personality", callback_data=f"ai_personality_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "📊 Features Setup", callback_data=f"features_setup_{group_id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🎓 Tutorial", callback_data=f"group_tutorial_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "✅ Setup Complete", callback_data=f"setup_complete_{group_id}"
                    ),
                ],
            ]
        )

    # Helper methods
    async def _get_member_count(self, chat, context) -> int:
        """Get group member count"""
        try:
            return await context.bot.get_chat_member_count(chat.id)
        except:
            return 0

    def _is_bot_mentioned(self, message_text: str, context) -> bool:
        """Check if bot is mentioned in message"""
        bot_username = context.bot.username
        return f"@{bot_username}" in message_text if bot_username else False

    def _detect_media_type(self, message) -> str:
        """Detect media type in message"""
        if message.photo:
            return "photo"
        elif message.video:
            return "video"
        elif message.document:
            return "document"
        elif message.voice:
            return "voice"
        elif message.sticker:
            return "sticker"
        else:
            return "text"

    def _extract_mentions(self, message_text: str) -> List[str]:
        """Extract user mentions from message"""
        import re

        return re.findall(r"@\w+", message_text)

    def _extract_hashtags(self, message_text: str) -> List[str]:
        """Extract hashtags from message"""
        import re

        return re.findall(r"#\w+", message_text)

    def _extract_urls(self, message_text: str) -> List[str]:
        """Extract URLs from message"""
        import re

        url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        return url_pattern.findall(message_text)

    async def _analyze_message_intent(
        self, message_text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze message intent and determine if AI response is needed"""

        intent = {
            "type": "general",
            "confidence": 0.5,
            "needs_ai_response": False,
            "priority": "normal",
            "keywords": [],
            "sentiment": "neutral",
        }

        message_lower = message_text.lower()

        # Check for questions
        if any(
            q in message_lower
            for q in ["?", "how", "what", "why", "when", "where", "who"]
        ):
            intent["type"] = "question"
            intent["needs_ai_response"] = True
            intent["priority"] = "high"

        # Check for help requests
        elif any(h in message_lower for h in ["help", "assist", "support", "guidance"]):
            intent["type"] = "help_request"
            intent["needs_ai_response"] = True
            intent["priority"] = "high"

        # Check for task-related content
        elif any(
            t in message_lower for t in ["task", "todo", "deadline", "action", "assign"]
        ):
            intent["type"] = "task_related"
            intent["needs_ai_response"] = False
            intent["priority"] = "medium"

        # Check for decision-making
        elif any(
            d in message_lower
            for d in ["decide", "choice", "option", "vote", "opinion"]
        ):
            intent["type"] = "decision_making"
            intent["needs_ai_response"] = True
            intent["priority"] = "high"

        # Check for information sharing
        elif any(
            i in message_lower for i in ["fyi", "info", "update", "news", "share"]
        ):
            intent["type"] = "information_sharing"
            intent["needs_ai_response"] = False
            intent["priority"] = "low"

        return intent

    async def _get_current_group_context(self, group_id: str) -> Dict[str, Any]:
        """Get current context for the group"""

        context = {
            "active_participants": len(self.typing_indicators.get(group_id, {})),
            "recent_activity": len(self.activity_streams.get(group_id, [])),
            "ongoing_discussions": len(
                [
                    d
                    for d in self.active_discussions.values()
                    if d["group_id"] == group_id and d["status"] == "active"
                ]
            ),
            "knowledge_items": len(self.knowledge_base.get(group_id, {})),
            "group_settings": self.group_settings.get(group_id, {}),
        }

        return context

    async def _update_group_knowledge_base(
        self, group_id: str, message_text: str, intent: Dict[str, Any]
    ) -> None:
        """Update group knowledge base with relevant information"""

        if group_id not in self.knowledge_base:
            self.knowledge_base[group_id] = {}

        # Extract potential knowledge items based on intent and content
        if (
            intent["type"] in ["information_sharing", "decision_making"]
            and len(message_text) > 50
        ):
            # Simple keyword extraction (can be enhanced with NLP)
            keywords = []
            important_words = [
                "project",
                "deadline",
                "meeting",
                "decision",
                "important",
                "remember",
            ]

            for word in important_words:
                if word in message_text.lower():
                    keywords.append(word)

            if keywords:
                knowledge_item = {
                    "content": message_text[:200],
                    "keywords": keywords,
                    "timestamp": datetime.now().isoformat(),
                    "importance": intent.get("priority", "normal"),
                    "type": intent["type"],
                }

                # Use first keyword as key
                key = keywords[0] if keywords else "general"
                if key not in self.knowledge_base[group_id]:
                    self.knowledge_base[group_id][key] = []

                self.knowledge_base[group_id][key].append(knowledge_item)

                # Keep only recent items (last 20 per category)
                self.knowledge_base[group_id][key] = self.knowledge_base[group_id][key][
                    -20:
                ]

    async def _log_group_activity(
        self, group_id: str, user_id: int, activity_type: str, data: Dict[str, Any]
    ) -> None:
        """Log group activity for analytics"""

        if group_id not in self.activity_streams:
            self.activity_streams[group_id] = []

        activity_log = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "activity_type": activity_type,
            "data": data,
        }

        self.activity_streams[group_id].append(activity_log)

        # Keep only recent activities (last 100)
        self.activity_streams[group_id] = self.activity_streams[group_id][-100:]

    async def _update_group_activity(
        self, group_id: str, user_id: int, message_text: str
    ) -> None:
        """Update real-time group activity"""

        await self._log_group_activity(
            group_id,
            user_id,
            "message_sent",
            {
                "message_length": len(message_text),
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def _check_collaboration_triggers(
        self, group_id: str, user_id: int, message_text: str, context
    ) -> None:
        """Check for collaboration triggers and suggest actions"""

        message_lower = message_text.lower()

        # Check for automatic collaboration suggestions
        suggestions = []

        if any(word in message_lower for word in ["meeting", "schedule", "when"]):
            suggestions.append("📅 Schedule a meeting")

        if any(word in message_lower for word in ["task", "assignment", "todo"]):
            suggestions.append("✅ Create task")

        if any(word in message_lower for word in ["important", "remember", "note"]):
            suggestions.append("📝 Save as note")

        if any(word in message_lower for word in ["decision", "vote", "choose"]):
            suggestions.append("🗳️ Start poll")

        # Send suggestions if any found
        if suggestions and len(suggestions) <= 2:  # Don't overwhelm
            suggestion_keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            suggestion, callback_data=f"collab_suggest_{group_id}_{i}"
                        )
                    ]
                    for i, suggestion in enumerate(suggestions)
                ]
            )

            await context.bot.send_message(
                chat_id=group_id,
                text="💡 **Smart Suggestion:**",
                reply_markup=suggestion_keyboard,
                parse_mode="Markdown",
            )

    async def handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle callback queries for group features"""

        query = update.callback_query
        if not query or not query.data:
            return

        data_parts = query.data.split("_")
        if len(data_parts) < 2:
            return

        action = data_parts[0]

        try:
            if action == "group":
                await self._handle_group_callback(query, data_parts, context)
            elif action == "collab":
                await self._handle_collaboration_callback(query, data_parts, context)
            elif action == "search":
                await self._handle_search_callback(query, data_parts, context)
            elif action == "disc":
                await self._handle_discussion_callback(query, data_parts, context)

            await query.answer()

        except Exception as e:
            self.logger.error(f"Error handling group callback: {e}")
            await query.answer("❌ An error occurred. Please try again.")

    async def _handle_group_callback(self, query, data_parts, context) -> None:
        """Handle group-specific callbacks"""
        # Implementation for group callbacks
        pass

    async def _handle_collaboration_callback(self, query, data_parts, context) -> None:
        """Handle collaboration callbacks"""
        # Implementation for collaboration callbacks
        pass

    async def _handle_search_callback(self, query, data_parts, context) -> None:
        """Handle search callbacks"""
        # Implementation for search callbacks
        pass

    async def _handle_discussion_callback(self, query, data_parts, context) -> None:
        """Handle discussion callbacks"""
        # Implementation for discussion callbacks
        pass
