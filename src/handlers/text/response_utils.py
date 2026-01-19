"""
Response formatting and utility functions for text handlers.
"""
import re
import logging
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatAction
from src.utils.log.telegramlog import telegram_logger


class ResponseUtilitiesMixin:
    """Mixin providing response utilities and helper methods."""

    class MockMessage:
        """Mock message for sending to chat without an update object."""
        def __init__(self, bot, chat_id):
            self.bot = bot
            self.chat_id = chat_id

        async def reply_text(self, text, **kwargs):
            return await self.bot.send_message(chat_id=self.chat_id, text=text, **kwargs)

    async def _send_appropriate_chat_action(self, update, context, has_attached_media, media_type):
        """Send appropriate chat action based on message type."""
        action = ChatAction.TYPING
        if has_attached_media:
            action_map = {
                "photo": ChatAction.UPLOAD_PHOTO,
                "video": ChatAction.UPLOAD_VIDEO,
                "audio": ChatAction.RECORD_VOICE,
                "document": ChatAction.UPLOAD_DOCUMENT,
            }
            action = action_map.get(media_type, ChatAction.TYPING)
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=action)

    async def _get_user_preferred_model(self, user_id: int) -> str:
        """Get user's preferred model."""
        from src.services.user_preferences_manager import UserPreferencesManager
        preferences_manager = UserPreferencesManager(self.user_data_manager)
        preferred_model = await preferences_manager.get_user_model_preference(user_id)
        self.logger.info(f"Preferred model for user {user_id}: {preferred_model}")
        return preferred_model

    async def _load_user_context(self, user_id: int, update: Update) -> str:
        """Load user context including name and profile information."""
        try:
            user_context_parts = []
            user = update.effective_user

            if user:
                if user.first_name:
                    user_context_parts.append(f"Name: {user.first_name}")
                if user.last_name:
                    user_context_parts.append(f"Last name: {user.last_name}")
                if user.username:
                    user_context_parts.append(f"Username: @{user.username}")

            try:
                user_profile = await self.memory_manager.get_user_profile(user_id)
                if user_profile:
                    if user_profile.get("name"):
                        user_context_parts.append(f"Preferred name: {user_profile['name']}")
                    if user_profile.get("conversation_count"):
                        user_context_parts.append(f"Previous conversations: {user_profile['conversation_count']}")
                    for key, value in user_profile.items():
                        if key not in ["name", "conversation_count", "created_at", "last_updated"] and value:
                            user_context_parts.append(f"{key.capitalize()}: {value}")

                user_data = await self.user_data_manager.get_user_data(user_id)
                if user_data:
                    user_prefs = user_data.get("preferences", {})
                    if user_prefs.get("name") and f"Preferred name: {user_prefs['name']}" not in user_context_parts:
                        user_context_parts.append(f"Preferred name: {user_prefs['name']}")
            except Exception as e:
                self.logger.debug(f"Could not load user profile from MongoDB: {e}")

            return "; ".join(user_context_parts) if user_context_parts else ""
        except Exception as e:
            self.logger.error(f"Error loading user context: {e}")
            return ""

    def _clean_response_content(self, content: str) -> str:
        """Clean response content by removing thinking tags and tool calls."""
        if not content:
            return content

        # Remove thinking and tool call tags
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        content = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL)
        content = re.sub(r"<[^>]+>.*?</[^>]+>", "", content, flags=re.DOTALL)

        # Clean up whitespace
        content = content.strip()
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
        return content

    async def _send_formatted_response(
        self, update, context, message, response,
        model_indicator, quoted_text, quoted_message_id,
    ):
        """Format and send the AI response."""
        message_chunks = await self.response_formatter.split_long_message(response)
        sent_messages = []
        context.user_data["last_message_indicator"] = model_indicator

        is_reply = self.context_handler.should_use_reply_format(quoted_text, quoted_message_id)

        for i, chunk in enumerate(message_chunks):
            try:
                text_to_send = (
                    self.response_formatter.format_with_model_indicator(chunk, model_indicator, is_reply)
                    if i == 0 else chunk
                )

                if i == 0 and is_reply:
                    last_message = await self.response_formatter.safe_send_message(
                        message, text_to_send, reply_to_message_id=quoted_message_id
                    )
                elif i == 0:
                    last_message = await self.response_formatter.safe_send_message(message, text_to_send)
                else:
                    mock_message = self.MockMessage(context.bot, update.effective_chat.id)
                    last_message = await self.response_formatter.safe_send_message(mock_message, text_to_send)

                if last_message:
                    sent_messages.append(last_message)
            except Exception as e:
                self.logger.error(f"Failed to send message chunk {i}: {str(e)}")
                continue

        if sent_messages:
            if "bot_messages" not in context.user_data:
                context.user_data["bot_messages"] = {}
            context.user_data["bot_messages"][message.message_id] = [msg.message_id for msg in sent_messages]

    async def _handle_mcp_document_output(
        self, context: ContextTypes.DEFAULT_TYPE, user_id: int, message, mcp_response: str,
    ):
        """Handle document outputs from MCP tools."""
        try:
            document_indicators = ["document created", "saved to", "file created", ".docx", ".pdf", ".xlsx", ".pptx"]

            if not any(ind in mcp_response.lower() for ind in document_indicators):
                return

            recent_documents = self.document_sender.find_recent_documents(directory=".", max_age_seconds=300)
            if not recent_documents:
                return

            import os
            document_path = recent_documents[0]
            file_name = os.path.basename(document_path)

            success = await self.document_sender.send_document(
                bot=context.bot, chat_id=message.chat_id,
                file_path=document_path, caption=f"{file_name}",
                reply_to_message_id=message.message_id,
            )

            if success:
                telegram_logger.log_message(f"Document sent: {file_name}", user_id)
        except Exception as e:
            self.logger.error(f"Error handling MCP document output: {str(e)}")
