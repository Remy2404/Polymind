"""
Group Chat Integration Module for Telegram Bot
Seamlessly integrates advanced group features into the existing bot architecture.
"""

import logging
from typing import Dict, Any, Optional
from telegram import Update, Chat
from telegram.ext import ContextTypes

from src.services.group_chat.group_manager import GroupManager
from src.services.group_chat.ui_components import GroupUIManager

logger = logging.getLogger(__name__)


class GroupChatIntegration:
    """
    Integration layer for group chat functionality.
    Extends the existing bot with advanced group features.
    """

    def __init__(self, user_data_manager, conversation_manager):
        self.user_data_manager = user_data_manager
        self.conversation_manager = conversation_manager
        self.logger = logging.getLogger(__name__)

        # Initialize group components
        self.group_manager = GroupManager(user_data_manager, conversation_manager)
        self.ui_manager = GroupUIManager()

        self.logger.info("GroupChatIntegration initialized")

    async def process_group_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, message_text: str
    ) -> tuple[str, Dict[str, Any]]:
        """
        Process group messages with enhanced context and UI.

        Args:
            update: Telegram update object
            context: Telegram context
            message_text: The message text to process

        Returns:
            Tuple of (enhanced_message, metadata)
        """
        try:
            chat = update.effective_chat

            # Check if this is a group chat
            if not chat or chat.type not in ["group", "supergroup"]:
                return message_text, {}

            # Process through group manager
            enhanced_message, metadata = await self.group_manager.handle_group_message(
                update, context, message_text
            )

            # Add UI enhancements
            ui_enhancements = await self.ui_manager.enhance_group_response(
                update, metadata
            )

            # Combine metadata
            combined_metadata = {
                **metadata,
                "ui_components": ui_enhancements,
                "group_features_enabled": True,
            }

            return enhanced_message, combined_metadata

        except Exception as e:
            self.logger.error(f"Error processing group message: {e}")
            return message_text, {}

    async def handle_group_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        command: str,
        args: list,
    ) -> Optional[str]:
        """
        Handle group-specific commands.

        Args:
            update: Telegram update object
            context: Telegram context
            command: Command name
            args: Command arguments

        Returns:
            Response message or None
        """
        try:
            chat = update.effective_chat

            if not chat or chat.type not in ["group", "supergroup"]:
                return None

            group_id = chat.id

            if command == "groupstats":
                # Show group analytics
                analytics = await self.group_manager.get_group_analytics(group_id)
                return await self.ui_manager.format_group_analytics(analytics)

            elif command == "groupsettings":
                # Show group settings menu
                return await self.ui_manager.create_settings_menu(group_id)

            elif command == "groupcontext":
                # Show conversation context options
                if group_id in self.group_manager.group_contexts:
                    context_info = await self.group_manager.export_group_context(
                        group_id
                    )
                    return await self.ui_manager.format_context_summary(context_info)
                else:
                    return "No conversation context available for this group yet."

            elif command == "groupthreads":
                # Show active threads
                if group_id in self.group_manager.group_contexts:
                    group_context = self.group_manager.group_contexts[group_id]
                    return await self.ui_manager.format_thread_list(
                        group_context.threads
                    )
                else:
                    return "No active threads in this group."

            elif command == "cleanthreads":
                # Clean up inactive threads
                await self.group_manager.cleanup_inactive_threads(group_id)
                return "✅ Cleaned up inactive conversation threads."

            return None

        except Exception as e:
            self.logger.error(f"Error handling group command '{command}': {e}")
            return f"❌ Error processing command: {str(e)}"

    def is_group_chat(self, chat: Chat) -> bool:
        """Check if the chat is a group chat."""
        return chat and chat.type in ["group", "supergroup"]

    async def get_group_context_for_ai(self, group_id: int) -> Optional[str]:
        """Get formatted group context for AI processing."""
        try:
            if group_id in self.group_manager.group_contexts:
                group_context = self.group_manager.group_contexts[group_id]

                # Format context for AI
                context_parts = [
                    f"Group: {group_context.group_name}",
                    f"Participants: {len(group_context.participants)}",
                ]

                if group_context.active_topics:
                    topics = ", ".join(group_context.active_topics[:5])
                    context_parts.append(f"Active topics: {topics}")

                if group_context.recent_messages:
                    context_parts.append("Recent conversation:")
                    for msg in group_context.recent_messages[-3:]:
                        user_name = msg.get("user_name", "User")
                        content = msg.get("content", "")[:100]
                        context_parts.append(f"  {user_name}: {content}")

                return "\n".join(context_parts)

            return None

        except Exception as e:
            self.logger.error(f"Error getting group context for AI: {e}")
            return None
