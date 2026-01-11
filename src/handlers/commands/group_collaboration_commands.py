"""
Group Collaboration Command Handlers
Provides team chat integration with shared memory and collaborative features
"""

import logging
from typing import Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from datetime import datetime

logger = logging.getLogger(__name__)


class GroupCollaborationCommands:
    """Handles group collaboration features and team chat integration"""

    def __init__(self, conversation_manager, user_data_manager, telegram_logger):
        self.conversation_manager = conversation_manager
        self.user_data_manager = user_data_manager
        self.telegram_logger = telegram_logger
        self.logger = logging.getLogger(__name__)

    async def start_group_session_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Start a collaborative group session"""
        if not update.effective_chat or update.effective_chat.type == "private":
            await update.message.reply_text(
                "❌ This command can only be used in group chats!"
            )
            return
        group_id = str(update.effective_chat.id)
        user_id = update.effective_user.id
        topic = None
        if context.args:
            topic = " ".join(context.args)
        session_data = await self.conversation_manager.start_group_session(
            group_id, user_id, topic
        )
        keyboard = [
            [
                InlineKeyboardButton(
                    "📝 Add Note", callback_data=f"group_add_note_{group_id}"
                ),
                InlineKeyboardButton(
                    "👥 View Participants",
                    callback_data=f"group_participants_{group_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    "📊 Group Summary", callback_data=f"group_summary_{group_id}"
                ),
                InlineKeyboardButton(
                    "🧠 Shared Memory", callback_data=f"group_memory_{group_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    "⚙️ Session Settings", callback_data=f"group_settings_{group_id}"
                ),
                InlineKeyboardButton(
                    "🔚 End Session", callback_data=f"group_end_{group_id}"
                ),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        session_text = "🚀 **Group Collaboration Session Started!**\n\n"
        session_text += f"**Session ID:** `{session_data['session_id']}`\n"
        session_text += f"**Initiator:** {update.effective_user.first_name}\n"
        if topic:
            session_text += f"**Topic:** {topic}\n"
        session_text += "\n💡 **Features Available:**\n"
        session_text += "• 🧠 Shared memory across all participants\n"
        session_text += "• 📝 Collaborative notes and action items\n"
        session_text += "• 🔍 Intelligent context sharing\n"
        session_text += "• 📊 Real-time group insights\n\n"
        session_text += "All group members can now collaborate with shared AI context!"
        await update.message.reply_text(
            session_text, reply_markup=reply_markup, parse_mode="Markdown"
        )
        self.telegram_logger.log_message(
            user_id,
            f"Started group session in {group_id} with topic: {topic or 'None'}",
        )

    async def group_summary_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Generate and display group conversation summary"""
        group_id = (
            str(update.effective_chat.id)
            if update.effective_chat.type != "private"
            else None
        )
        user_id = update.effective_user.id
        if not group_id:
            await update.message.reply_text(
                "❌ This command can only be used in group chats!"
            )
            return
        progress_msg = await update.message.reply_text(
            "🔍 Analyzing group conversation..."
        )
        try:
            summary = (
                await self.conversation_manager.memory_manager.get_conversation_summary(
                    "", is_group=True, group_id=group_id
                )
            )
            activity_summary = await self.conversation_manager.memory_manager.get_group_activity_summary(
                group_id, days=7
            )
            group_context = await self.conversation_manager.get_group_context(
                group_id, user_id
            )
            summary_text = "📊 **Group Conversation Summary**\n\n"
            summary_text += "📈 **Activity (Last 7 Days):**\n"
            summary_text += f"• Messages: {activity_summary.get('total_messages', 0)}\n"
            summary_text += (
                f"• Active participants: {activity_summary.get('active_users', 0)}\n"
            )
            if activity_summary.get("most_active_user"):
                summary_text += (
                    f"• Most active: User {activity_summary['most_active_user']}\n"
                )
            if activity_summary.get("message_types"):
                summary_text += "\n📋 **Content Types:**\n"
                for msg_type, count in activity_summary["message_types"].items():
                    type_emoji = {
                        "text": "💬",
                        "image": "🖼️",
                        "document": "📄",
                        "voice": "🎤",
                    }.get(msg_type, "📄")
                    summary_text += f"• {type_emoji} {msg_type.title()}: {count}\n"
            summary_text += f"\n🧠 **Conversation Overview:**\n{summary}\n"
            if group_context.get("collaborative_notes"):
                summary_text += "\n📝 **Recent Collaborative Notes:**\n"
                for note in group_context["collaborative_notes"][-3:]:
                    summary_text += f"• {note.get('content', '')[:100]}...\n"
            if group_context.get("active_topics"):
                summary_text += "\n🎯 **Active Topics:**\n"
                for topic in group_context["active_topics"][-5:]:
                    summary_text += f"• {topic}\n"
            keyboard = [
                [
                    InlineKeyboardButton(
                        "📤 Export Summary", callback_data=f"export_summary_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "🔍 Detailed Analysis",
                        callback_data=f"detailed_analysis_{group_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "📝 Add Note", callback_data=f"group_add_note_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "🔙 Back to Chat", callback_data="dismiss_message"
                    ),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await progress_msg.edit_text(
                summary_text, reply_markup=reply_markup, parse_mode="Markdown"
            )
        except Exception as e:
            self.logger.error(f"Error generating group summary: {e}")
            await progress_msg.edit_text(
                "❌ Failed to generate group summary. Please try again later."
            )

    async def group_memory_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Display shared group memory and knowledge"""
        group_id = (
            str(update.effective_chat.id)
            if update.effective_chat.type != "private"
            else None
        )
        if not group_id:
            await update.message.reply_text(
                "❌ This command can only be used in group chats!"
            )
            return
        query = " ".join(context.args) if context.args else "recent group discussions"
        progress_msg = await update.message.reply_text(
            f"🧠 Searching group memory for: '{query}'..."
        )
        try:
            relevant_memories = (
                await self.conversation_manager.memory_manager.get_relevant_memory(
                    "",
                    query,
                    limit=8,
                    is_group=True,
                    group_id=group_id,
                    include_group_knowledge=True,
                )
            )
            if not relevant_memories:
                await progress_msg.edit_text(
                    f"🔍 No relevant memories found for '{query}' in group history.\n\n"
                    "Try a different search term or start a conversation to build group memory!"
                )
                return
            memory_text = "🧠 **Group Shared Memory**\n\n"
            memory_text += f"🔍 **Search:** {query}\n"
            memory_text += f"📊 **Found {len(relevant_memories)} relevant items**\n\n"
            for i, memory in enumerate(relevant_memories[:5], 1):
                content = memory.get("content", "")
                timestamp = memory.get("timestamp", 0)
                user_id_mem = memory.get("user_id", "Assistant")
                if timestamp:
                    time_str = datetime.fromtimestamp(timestamp).strftime("%m/%d %H:%M")
                else:
                    time_str = "Unknown"
                display_content = (
                    content[:150] + "..." if len(content) > 150 else content
                )
                memory_text += f"**{i}.** [{time_str}] User {user_id_mem}\n"
                memory_text += f"    {display_content}\n\n"
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔍 Search Different Term",
                        callback_data=f"memory_search_{group_id}",
                    ),
                    InlineKeyboardButton(
                        "📊 Memory Stats", callback_data=f"memory_stats_{group_id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "💾 Export Memories",
                        callback_data=f"export_memories_{group_id}",
                    ),
                    InlineKeyboardButton(
                        "🗑️ Clear Memories", callback_data=f"clear_memories_{group_id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🔙 Back to Chat", callback_data="dismiss_message"
                    )
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await progress_msg.edit_text(
                memory_text, reply_markup=reply_markup, parse_mode="Markdown"
            )
        except Exception as e:
            self.logger.error(f"Error accessing group memory: {e}")
            await progress_msg.edit_text(
                "❌ Failed to access group memory. Please try again later."
            )

    async def add_group_note_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Add a collaborative note to group memory"""
        group_id = (
            str(update.effective_chat.id)
            if update.effective_chat.type != "private"
            else None
        )
        user_id = update.effective_user.id
        if not group_id:
            await update.message.reply_text(
                "❌ This command can only be used in group chats!"
            )
            return
        if not context.args:
            await update.message.reply_text(
                "📝 **Add Group Note**\n\n"
                "Usage: `/addnote <your note here>`\n\n"
                "Examples:\n"
                "• `/addnote Meeting scheduled for Friday 2PM`\n"
                "• `/addnote Remember to review the project proposal`\n"
                "• `/addnote Action item: Update documentation`\n\n"
                "Notes are shared with all group members and saved to group memory!"
            )
            return
        note_content = " ".join(context.args)
        try:
            await self.conversation_manager.add_collaborative_note(
                group_id, user_id, note_content, "user_note"
            )
            confirmation_text = "✅ **Note Added to Group Memory!**\n\n"
            confirmation_text += f"📝 **Note:** {note_content}\n"
            confirmation_text += (
                f"👤 **Added by:** {update.effective_user.first_name}\n"
            )
            confirmation_text += f"🕒 **Time:** {datetime.now().strftime('%H:%M')}\n\n"
            confirmation_text += "This note is now available to all group members and will be considered in future AI responses!"
            keyboard = [
                [
                    InlineKeyboardButton(
                        "📝 Add Another Note",
                        callback_data=f"add_another_note_{group_id}",
                    ),
                    InlineKeyboardButton(
                        "📋 View All Notes", callback_data=f"view_notes_{group_id}"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🔙 Back to Chat", callback_data="dismiss_message"
                    )
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                confirmation_text, reply_markup=reply_markup, parse_mode="Markdown"
            )
            self.telegram_logger.log_message(
                user_id,
                f"Added collaborative note to group {group_id}: {note_content[:50]}...",
            )
        except Exception as e:
            self.logger.error(f"Error adding group note: {e}")
            await update.message.reply_text(
                "❌ Failed to add note to group memory. Please try again later."
            )

    async def group_participants_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show group participants and their contribution stats"""
        group_id = (
            str(update.effective_chat.id)
            if update.effective_chat.type != "private"
            else None
        )
        if not group_id:
            await update.message.reply_text(
                "❌ This command can only be used in group chats!"
            )
            return
        try:
            participants = (
                await self.conversation_manager.memory_manager.get_group_participants(
                    group_id
                )
            )
            activity_summary = await self.conversation_manager.memory_manager.get_group_activity_summary(
                group_id, days=30
            )
            participants_text = "👥 **Group Participants & Stats**\n\n"
            participants_text += f"📊 **Total Participants:** {len(participants)}\n"
            participants_text += f"📈 **Active in Last 30 Days:** {activity_summary.get('active_users', 0)}\n\n"
            user_activity = activity_summary.get("user_activity", {})
            participants_text += "**📋 Contribution Stats:**\n"
            sorted_participants = sorted(
                participants, key=lambda x: user_activity.get(x, 0), reverse=True
            )
            for i, participant_id in enumerate(sorted_participants[:10], 1):
                message_count = user_activity.get(participant_id, 0)
                if message_count > 50:
                    activity_emoji = "🔥"
                elif message_count > 20:
                    activity_emoji = "⭐"
                elif message_count > 5:
                    activity_emoji = "✅"
                else:
                    activity_emoji = "💤"
                participants_text += f"{i}. {activity_emoji} User {participant_id}: {message_count} messages\n"
            if len(participants) > 10:
                participants_text += (
                    f"\n... and {len(participants) - 10} more participants\n"
                )
            participants_text += "\n🤝 **Collaboration Insights:**\n"
            if activity_summary.get("total_messages", 0) > 100:
                participants_text += (
                    "• High engagement - active collaboration detected\n"
                )
            elif activity_summary.get("total_messages", 0) > 20:
                participants_text += "• Moderate engagement - good team interaction\n"
            else:
                participants_text += (
                    "• Getting started - encourage more participation\n"
                )
            if len(participants) > 1:
                participants_text += (
                    "• Multi-participant discussions enhance AI context\n"
                )
            keyboard = [
                [
                    InlineKeyboardButton(
                        "📊 Detailed Stats", callback_data=f"detailed_stats_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "🎯 Engagement Tips",
                        callback_data=f"engagement_tips_{group_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "🔙 Back to Chat", callback_data="dismiss_message"
                    )
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                participants_text, reply_markup=reply_markup, parse_mode="Markdown"
            )
        except Exception as e:
            self.logger.error(f"Error getting group participants: {e}")
            await update.message.reply_text(
                "❌ Failed to get group participants. Please try again later."
            )

    async def group_search_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Search group conversation history"""
        group_id = (
            str(update.effective_chat.id)
            if update.effective_chat.type != "private"
            else None
        )
        user_id = update.effective_user.id
        if not group_id:
            await update.message.reply_text(
                "❌ This command can only be used in group chats!"
            )
            return
        if not context.args:
            await update.message.reply_text(
                "🔍 **Search Group History**\n\n"
                "Usage: `/groupsearch <search terms>`\n\n"
                "Examples:\n"
                "• `/groupsearch project deadline`\n"
                "• `/groupsearch meeting notes`\n"
                "• `/groupsearch python code`\n"
                "• `/groupsearch user123 suggestions`\n\n"
                "I'll search through all group conversations and shared memory!"
            )
            return
        search_query = " ".join(context.args)
        progress_msg = await update.message.reply_text(
            f"🔍 Searching group history for: '{search_query}'..."
        )
        try:
            intelligent_context = (
                await self.conversation_manager.get_intelligent_context(
                    user_id, search_query, group_id
                )
            )
            relevant_memories = intelligent_context.get("relevant_memory", [])
            suggestions = intelligent_context.get("suggested_actions", [])
            if not relevant_memories:
                await progress_msg.edit_text(
                    f"🔍 No results found for '{search_query}'\n\n"
                    "Try different keywords or check if the topic was discussed in this group."
                )
                return
            results_text = "🔍 **Group Search Results**\n\n"
            results_text += f"**Query:** {search_query}\n"
            results_text += (
                f"**Found:** {len(relevant_memories)} relevant conversations\n\n"
            )
            for i, memory in enumerate(relevant_memories[:6], 1):
                content = memory.get("content", "")
                timestamp = memory.get("timestamp", 0)
                user_id_mem = memory.get("user_id", "Assistant")
                if timestamp:
                    time_str = datetime.fromtimestamp(timestamp).strftime("%m/%d %H:%M")
                else:
                    time_str = "Recent"
                display_content = (
                    content[:200] + "..." if len(content) > 200 else content
                )
                results_text += f"**{i}.** [{time_str}] User {user_id_mem}\n"
                results_text += f"    {display_content}\n\n"
            if suggestions:
                results_text += "💡 **Suggested Actions:**\n"
                for suggestion in suggestions[:3]:
                    results_text += f"• {suggestion}\n"
            keyboard = [
                [
                    InlineKeyboardButton(
                        "🔍 Refine Search", callback_data=f"refine_search_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "📊 Search Analytics",
                        callback_data=f"search_analytics_{group_id}",
                    ),
                ],
                [
                    InlineKeyboardButton(
                        "💾 Save Results", callback_data=f"save_search_{group_id}"
                    ),
                    InlineKeyboardButton(
                        "🔙 Back to Chat", callback_data="dismiss_message"
                    ),
                ],
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await progress_msg.edit_text(
                results_text, reply_markup=reply_markup, parse_mode="Markdown"
            )
        except Exception as e:
            self.logger.error(f"Error searching group history: {e}")
            await progress_msg.edit_text(
                "❌ Failed to search group history. Please try again later."
            )

    async def handle_group_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle group-related callback queries"""
        query = update.callback_query
        await query.answer()
        callback_data = query.data
        try:
            if callback_data.startswith("group_add_note_"):
                group_id = callback_data.split("_")[-1]
                await self._prompt_for_note(query, group_id)
            elif callback_data.startswith("group_summary_"):
                group_id = callback_data.split("_")[-1]
                await self._show_group_summary_menu(query, group_id)
            elif callback_data.startswith("group_memory_"):
                group_id = callback_data.split("_")[-1]
                await self._show_memory_interface(query, group_id)
            elif callback_data.startswith("group_participants_"):
                group_id = callback_data.split("_")[-1]
                await self._show_participants_details(query, group_id)
            elif callback_data == "dismiss_message":
                await query.message.delete()
        except Exception as e:
            self.logger.error(f"Error handling group callback: {e}")
            await query.message.reply_text(
                "❌ Error processing request. Please try again."
            )

    async def _prompt_for_note(self, query, group_id: str):
        """Prompt user to add a note"""
        prompt_text = (
            "📝 **Add Collaborative Note**\n\n"
            "Please reply to this message with your note, or use:\n"
            "`/addnote Your note content here`\n\n"
            "Your note will be shared with all group members!"
        )
        keyboard = [
            [InlineKeyboardButton("❌ Cancel", callback_data="dismiss_message")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            prompt_text, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def _show_group_summary_menu(self, query, group_id: str):
        """Show group summary options"""
        menu_text = (
            "📊 **Group Summary Options**\n\nChoose what type of summary you'd like:"
        )
        keyboard = [
            [
                InlineKeyboardButton(
                    "📈 Activity Summary", callback_data=f"activity_summary_{group_id}"
                ),
                InlineKeyboardButton(
                    "💬 Conversation Summary", callback_data=f"conv_summary_{group_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    "📝 Notes Summary", callback_data=f"notes_summary_{group_id}"
                ),
                InlineKeyboardButton(
                    "🎯 Topic Analysis", callback_data=f"topic_analysis_{group_id}"
                ),
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="dismiss_message")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            menu_text, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def _show_memory_interface(self, query, group_id: str):
        """Show memory interface options"""
        interface_text = (
            "🧠 **Group Memory Interface**\n\nExplore shared group knowledge:"
        )
        keyboard = [
            [
                InlineKeyboardButton(
                    "🔍 Search Memory", callback_data=f"memory_search_{group_id}"
                ),
                InlineKeyboardButton(
                    "📊 Memory Stats", callback_data=f"memory_stats_{group_id}"
                ),
            ],
            [
                InlineKeyboardButton(
                    "📚 Recent Memories", callback_data=f"recent_memories_{group_id}"
                ),
                InlineKeyboardButton(
                    "⭐ Important Items", callback_data=f"important_memories_{group_id}"
                ),
            ],
            [InlineKeyboardButton("🔙 Back", callback_data="dismiss_message")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            interface_text, reply_markup=reply_markup, parse_mode="Markdown"
        )

    async def _show_participants_details(self, query, group_id: str):
        """Show detailed participant information"""
        try:
            participants = (
                await self.conversation_manager.memory_manager.get_group_participants(
                    group_id
                )
            )
            details_text = "👥 **Group Participants Details**\n\n"
            details_text += f"**Total Members:** {len(participants)}\n\n"
            for participant_id in participants[:5]:
                group_context = await self.conversation_manager.get_group_context(
                    group_id, int(participant_id)
                )
                contributions = group_context.get("user_contributions", [])
                details_text += f"**User {participant_id}:**\n"
                details_text += f"• Recent contributions: {len(contributions)}\n"
                if contributions:
                    last_contribution = contributions[-1]
                    timestamp = last_contribution.get("timestamp", 0)
                    if timestamp:
                        time_str = datetime.fromtimestamp(timestamp).strftime(
                            "%m/%d %H:%M"
                        )
                        details_text += f"• Last active: {time_str}\n"
                details_text += "\n"
            keyboard = [
                [InlineKeyboardButton("🔙 Back", callback_data="dismiss_message")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                details_text, reply_markup=reply_markup, parse_mode="Markdown"
            )
        except Exception as e:
            self.logger.error(f"Error showing participant details: {e}")
            await query.edit_message_text("❌ Error loading participant details.")


async def is_group_admin(bot, group_id: str, user_id: int) -> bool:
    """Check if user is admin in the group"""
    try:
        chat_member = await bot.get_chat_member(group_id, user_id)
        return chat_member.status in ["administrator", "creator"]
    except Exception:
        return False


async def get_group_info(bot, group_id: str) -> Dict[str, Any]:
    """Get group information"""
    try:
        chat = await bot.get_chat(group_id)
        return {
            "title": chat.title,
            "type": chat.type,
            "member_count": await bot.get_chat_member_count(group_id),
            "description": chat.description,
        }
    except Exception:
        return {}
