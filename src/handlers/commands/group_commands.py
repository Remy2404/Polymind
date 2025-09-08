"""
Group Commands Handler
Handles all group-specific commands for the Polymind Telegram bot.
"""

import sys
import os
import logging

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class GroupCommands:
    """
    Handles group-specific commands for the Polymind Telegram bot.
    Follows single responsibility principle by focusing solely on group operations.
    """

    def __init__(self):
        """
        Initialize GroupCommands.
        """
        self.logger = logging.getLogger(__name__)

    async def group_settings_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle /groupsettings command for group configuration.

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            if not update:
                raise ValueError("The 'update' parameter is required but not provided.")

            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get settings menu
                settings_menu = await group_integration.ui_manager.create_settings_menu(
                    chat.id
                )

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(settings_menu)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_settings_command: {e}")
                await update.message.reply_text(
                    "❌ Settings feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_settings_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving group settings. Please try again."
            )

    async def group_context_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle /groupcontext command to show shared memory.

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get group context
                group_context = (
                    await group_integration.group_manager._get_or_create_group_context(
                        chat, update.effective_user
                    )
                )

                # Format shared memory
                if group_context.shared_memory:
                    context_text = "🧠 Group Shared Memory:\n\n"
                    for key, value in group_context.shared_memory.items():
                        context_text += f"• {key}: {value}\n"
                else:
                    context_text = "🧠 Group Shared Memory is empty\n\nAs the conversation continues, important information will be automatically stored here for future reference."

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(context_text)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_context_command: {e}")
                await update.message.reply_text(
                    "❌ Memory feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_context_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving group context. Please try again."
            )

    async def group_threads_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle /groupthreads command to list active conversation threads.

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Get group context
                group_context = (
                    await group_integration.group_manager._get_or_create_group_context(
                        chat, update.effective_user
                    )
                )

                # Format threads
                if group_context.threads:
                    formatted_threads = (
                        await group_integration.ui_manager.format_thread_list(
                            group_context.threads
                        )
                    )
                else:
                    formatted_threads = "🧵 No active conversation threads\n\nThreads are created automatically when users reply to messages. Start a discussion by replying to a message!"

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(formatted_threads)

            except AttributeError as e:
                self.logger.error(f"Missing attribute in group_threads_command: {e}")
                await update.message.reply_text(
                    "❌ Thread feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in group_threads_command: {e}")
            await update.message.reply_text(
                "❌ Error retrieving conversation threads. Please try again."
            )

    async def clean_threads_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle /cleanthreads command to clean up inactive threads.

        Args:
            update: Telegram update object
            context: Telegram context object
        """
        try:
            chat = update.effective_chat
            if chat.type not in ["group", "supergroup"]:
                await update.message.reply_text(
                    "❌ This command is only available in group chats."
                )
                return

            # Check if user is admin
            user_member = await context.bot.get_chat_member(
                chat.id, update.effective_user.id
            )
            if user_member.status not in ["administrator", "creator"]:
                await update.message.reply_text(
                    "❌ Only group administrators can clean conversation threads."
                )
                return

            # Get group chat integration
            group_integration = context.bot_data.get("group_chat_integration")
            if not group_integration:
                await update.message.reply_text(
                    "❌ Group chat features are not available."
                )
                return

            try:
                # Clean threads
                cleaned_count = (
                    await group_integration.group_manager.cleanup_inactive_threads(
                        chat.id
                    )
                )

                # Send without Markdown formatting to avoid escaping issues
                await update.message.reply_text(
                    f"🧹 Thread Cleanup Complete\n\nRemoved {cleaned_count} inactive conversation threads."
                )

            except AttributeError as e:
                self.logger.error(f"Missing attribute in clean_threads_command: {e}")
                await update.message.reply_text(
                    "❌ Thread cleanup feature is unavailable. Please contact the bot administrator."
                )

        except Exception as e:
            self.logger.error(f"Error in clean_threads_command: {e}")
            await update.message.reply_text(
                "❌ Error cleaning conversation threads. Please try again."
            )
