import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class GroupUIManager:
    """
    Manages UI components and formatting for group chat features.
    Provides rich, interactive elements for better user experience.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # UI Constants
        self.EMOJIS = {
            "group": "👥",
            "thread": "🧵",
            "topic": "🔖",
            "memory": "🧠",
            "analytics": "📊",
            "settings": "⚙️",
            "participant": "👤",
            "message": "💬",
            "active": "🟢",
            "inactive": "🔴",
            "warning": "⚠️",
            "success": "✅",
            "error": "❌",
            "info": "ℹ️",
        }

        self.logger.info("GroupUIManager initialized")

    async def enhance_group_response(
        self, update: Update, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance responses with UI components based on context.

        Args:
            update: Telegram update object
            metadata: Context metadata from group processing

        Returns:
            Dictionary of UI enhancements
        """
        try:
            enhancements = {
                "inline_keyboard": None,
                "formatting": "markdown",
                "status_indicators": [],
                "context_preview": None,
            }

            # Add context indicators
            if metadata.get("thread_id"):
                enhancements["status_indicators"].append(
                    f"{self.EMOJIS['thread']} Thread conversation"
                )

            if metadata.get("shared_topics"):
                topics = ", ".join(metadata["shared_topics"][:3])
                enhancements["status_indicators"].append(
                    f"{self.EMOJIS['topic']} Topics: {topics}"
                )

            # Add quick action buttons for group management
            if self._should_show_management_buttons(update, metadata):
                enhancements["inline_keyboard"] = (
                    await self._create_quick_actions_keyboard(metadata.get("group_id"))
                )

            return enhancements

        except Exception as e:
            self.logger.error(f"Error enhancing group response: {e}")
            return {}

    def _should_show_management_buttons(
        self, update: Update, metadata: Dict[str, Any]
    ) -> bool:
        """Determine if management buttons should be shown."""
        # Show buttons for group admins or when explicitly requested
        chat = update.effective_chat
        user = update.effective_user

        if not chat or not user:
            return False

        # TODO: Check if user is admin (requires bot admin privileges)
        # For now, show buttons in groups with more than 3 participants
        return metadata.get("participant_count", 0) > 3

    async def _create_quick_actions_keyboard(
        self, group_id: Optional[int]
    ) -> InlineKeyboardMarkup:
        """Create quick action buttons for group management."""
        buttons = [
            [
                InlineKeyboardButton(
                    f"{self.EMOJIS['analytics']} Stats",
                    callback_data=f"group_stats_{group_id}",
                ),
                InlineKeyboardButton(
                    f"{self.EMOJIS['thread']} Threads",
                    callback_data=f"group_threads_{group_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    f"{self.EMOJIS['memory']} Context",
                    callback_data=f"group_context_{group_id}",
                ),
                InlineKeyboardButton(
                    f"{self.EMOJIS['settings']} Settings",
                    callback_data=f"group_settings_{group_id}",
                ),
            ],
        ]

        return InlineKeyboardMarkup(buttons)

    async def format_group_analytics(self, analytics: Dict[str, Any]) -> str:
        """Format group analytics into a rich display."""
        if not analytics:
            return f"{self.EMOJIS['error']} No analytics available for this group."

        # Header
        message_parts = [
            f"{self.EMOJIS['analytics']} **Group Analytics**",
            f"{self.EMOJIS['group']} **{analytics.get('group_name', 'Unknown Group')}**",
            "",
        ]

        # Participant Statistics
        message_parts.extend(
            [
                "👥 **Participants:**",
                f"  • Total: {analytics.get('total_participants', 0)}",
                f"  • Active: {analytics.get('active_participants', 0)}",
                f"  • Total Messages: {analytics.get('total_messages', 0)}",
                "",
            ]
        )

        # Active Topics
        if analytics.get("active_topics"):
            topics = analytics["active_topics"]
            message_parts.extend(
                [
                    f"{self.EMOJIS['topic']} **Active Topics:**",
                    f"  {', '.join(topics[:5])}",
                    "",
                ]
            )

        # Thread Information
        if analytics.get("active_threads", 0) > 0:
            message_parts.extend(
                [
                    f"{self.EMOJIS['thread']} **Conversation Threads:**",
                    f"  • Active Threads: {analytics.get('active_threads', 0)}",
                    "",
                ]
            )

        # Memory Usage
        if analytics.get("shared_memory_items", 0) > 0:
            message_parts.extend(
                [
                    f"{self.EMOJIS['memory']} **Shared Memory:**",
                    f"  • Stored Items: {analytics.get('shared_memory_items', 0)}",
                    "",
                ]
            )

        # Timeline
        created_at = analytics.get("created_at", "")
        last_activity = analytics.get("last_activity", "")

        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                message_parts.append(
                    f"📅 **Created:** {created_date.strftime('%Y-%m-%d %H:%M')}"
                )
            except:
                pass

        if last_activity:
            try:
                activity_date = datetime.fromisoformat(
                    last_activity.replace("Z", "+00:00")
                )
                message_parts.append(
                    f"🕒 **Last Activity:** {activity_date.strftime('%Y-%m-%d %H:%M')}"
                )
            except:
                pass

        return "\n".join(message_parts)

    async def format_thread_list(self, threads: Dict[str, Any]) -> str:
        """Format active threads into a readable list."""
        if not threads:
            return (
                f"{self.EMOJIS['info']} No active conversation threads in this group."
            )

        message_parts = [f"{self.EMOJIS['thread']} **Active Conversation Threads**", ""]

        active_threads = [t for t in threads.values() if t.get("is_active", True)]

        if not active_threads:
            message_parts.append(
                f"{self.EMOJIS['info']} All threads are currently inactive."
            )
            return "\n".join(message_parts)

        for i, thread in enumerate(active_threads[:10], 1):  # Limit to 10 threads
            topic = thread.get("topic", "General discussion")
            participant_count = len(thread.get("participants", []))
            message_count = thread.get("message_count", 0)

            # Calculate thread age
            last_message = thread.get("last_message_at", "")
            age_str = ""
            if last_message:
                try:
                    last_msg_time = datetime.fromisoformat(
                        last_message.replace("Z", "+00:00")
                    )
                    age = datetime.now() - last_msg_time
                    if age.days > 0:
                        age_str = f" ({age.days}d ago)"
                    elif age.seconds > 3600:
                        age_str = f" ({age.seconds // 3600}h ago)"
                    else:
                        age_str = f" ({age.seconds // 60}m ago)"
                except:
                    pass

            thread_info = (
                f"**{i}.** {topic}\n"
                f"   👥 {participant_count} participants • "
                f"💬 {message_count} messages{age_str}"
            )

            message_parts.append(thread_info)

        if len(threads) > 10:
            message_parts.append(f"\n*... and {len(threads) - 10} more threads*")

        return "\n".join(message_parts)

    async def format_context_summary(self, context_info: Dict[str, Any]) -> str:
        """Format conversation context summary."""
        if not context_info:
            return f"{self.EMOJIS['error']} No context information available."

        message_parts = [
            f"{self.EMOJIS['memory']} **Conversation Context Summary**",
            f"{self.EMOJIS['group']} **{context_info.get('group_name', 'Unknown Group')}**",
            "",
        ]

        # Recent Topics
        if context_info.get("active_topics"):
            topics = context_info["active_topics"][:8]
            message_parts.extend(
                [
                    f"{self.EMOJIS['topic']} **Recent Topics:**",
                    f"  {', '.join(topics)}",
                    "",
                ]
            )

        # Shared Memory Highlights
        if context_info.get("shared_memory"):
            memory_items = list(context_info["shared_memory"].items())[:5]
            message_parts.extend([f"{self.EMOJIS['memory']} **Shared Knowledge:**"])

            for key, value in memory_items:
                value_preview = str(value)[:60]
                if len(str(value)) > 60:
                    value_preview += "..."
                message_parts.append(f"  • **{key}:** {value_preview}")

            message_parts.append("")

        # Thread Summary
        if context_info.get("threads"):
            active_threads = [
                t for t in context_info["threads"].values() if t.get("last_message_at")
            ]

            if active_threads:
                message_parts.extend(
                    [
                        f"{self.EMOJIS['thread']} **Active Threads:** {len(active_threads)}",
                        "",
                    ]
                )

        # Participant Activity
        if context_info.get("participants"):
            total_participants = len(context_info["participants"])
            message_parts.extend(
                [
                    f"{self.EMOJIS['participant']} **Participants:** {total_participants}",
                    "",
                ]
            )

        # Timeline
        created_at = context_info.get("created_at", "")
        updated_at = context_info.get("updated_at", "")

        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                message_parts.append(
                    f"📅 **Context Since:** {created_date.strftime('%Y-%m-%d')}"
                )
            except:
                pass

        if updated_at:
            try:
                updated_date = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                time_diff = datetime.now() - updated_date
                if time_diff.days > 0:
                    last_update = f"{time_diff.days} days ago"
                elif time_diff.seconds > 3600:
                    last_update = f"{time_diff.seconds // 3600} hours ago"
                else:
                    last_update = f"{time_diff.seconds // 60} minutes ago"

                message_parts.append(f"🕒 **Last Updated:** {last_update}")
            except:
                pass

        return "\n".join(message_parts)

    async def create_settings_menu(self, group_id: int) -> str:
        """Create an interactive settings menu for the group."""
        message_parts = [
            f"{self.EMOJIS['settings']} **Group Chat Settings**",
            f"Group ID: `{group_id}`",
            "",
            "**Available Settings:**",
            "",
            f"{self.EMOJIS['memory']} **Shared Memory**",
            "  • Enable/disable group conversation memory",
            "  • Set context retention period",
            "",
            f"{self.EMOJIS['thread']} **Thread Management**",
            "  • Enable/disable conversation threading",
            "  • Set thread timeout duration",
            "",
            f"{self.EMOJIS['participant']} **Participation Settings**",
            "  • Set AI participation level",
            "  • Configure response triggers",
            "",
            "**Quick Commands:**",
            "  • `/groupstats` - View group analytics",
            "  • `/groupcontext` - Show conversation context",
            "  • `/groupthreads` - List active threads",
            "  • `/cleanthreads` - Clean inactive threads",
            "",
            f"{self.EMOJIS['info']} Use the buttons below to modify settings.",
        ]

        return "\n".join(message_parts)

    async def create_participant_summary(self, participants: Dict[int, Any]) -> str:
        """Create a summary of group participants."""
        if not participants:
            return f"{self.EMOJIS['info']} No participant information available."

        message_parts = [
            f"{self.EMOJIS['participant']} **Group Participants**",
            f"Total: {len(participants)}",
            "",
        ]

        # Sort by message count
        sorted_participants = sorted(
            participants.items(),
            key=lambda x: x[1].get("message_count", 0),
            reverse=True,
        )

        for i, (user_id, participant) in enumerate(sorted_participants[:10], 1):
            name = participant.get("full_name", f"User {user_id}")
            username = participant.get("username", "")
            message_count = participant.get("message_count", 0)
            role = participant.get("role", "member")

            # Activity indicator
            last_active = participant.get("last_active", "")
            activity_indicator = self.EMOJIS["active"]

            if last_active:
                try:
                    last_active_time = datetime.fromisoformat(
                        last_active.replace("Z", "+00:00")
                    )
                    if datetime.now() - last_active_time > timedelta(days=7):
                        activity_indicator = self.EMOJIS["inactive"]
                except:
                    pass

            # Format participant info
            participant_info = f"{activity_indicator} **{name}**"
            if username:
                participant_info += f" (@{username})"

            participant_info += f"\n   💬 {message_count} messages • {role}"

            message_parts.append(participant_info)

        if len(participants) > 10:
            message_parts.append(
                f"\n*... and {len(participants) - 10} more participants*"
            )

        return "\n".join(message_parts)

    def format_status_indicator(self, status_type: str, message: str) -> str:
        """Format a status indicator message."""
        emoji = self.EMOJIS.get(status_type, self.EMOJIS["info"])
        return f"{emoji} {message}"

    def create_progress_bar(self, current: int, total: int, length: int = 10) -> str:
        """Create a text-based progress bar."""
        if total == 0:
            return "░" * length

        filled = int(length * current / total)
        bar = "█" * filled + "░" * (length - filled)
        percentage = int(100 * current / total)

        return f"{bar} {percentage}%"

    async def format_error_message(
        self, error: str, context: Optional[str] = None
    ) -> str:
        """Format an error message for display."""
        message_parts = [f"{self.EMOJIS['error']} **Error**", f"```{error}```"]

        if context:
            message_parts.extend(["", f"**Context:** {context}"])

        return "\n".join(message_parts)

    async def format_success_message(
        self, message: str, details: Optional[str] = None
    ) -> str:
        """Format a success message for display."""
        message_parts = [f"{self.EMOJIS['success']} {message}"]

        if details:
            message_parts.extend(["", details])

        return "\n".join(message_parts)
