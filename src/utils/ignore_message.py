"""
Message filtering module for Telegram bot.
This module provides functions to determine which messages should be ignored or processed.
"""

import logging
from .bot_username_helper import BotUsernameHelper

logger = logging.getLogger(__name__)


class MessageFilter:
    """
    Filters for handling Telegram bot message processing rules.
    Provides configurable rules for ignoring or processing messages.
    """

    def __init__(self):
        """Initialize the message filter with default settings."""
        self.logger = logging.getLogger(__name__)

    def should_ignore_update(
        self, update_data: dict, bot_username: str = "", context=None
    ) -> bool:
        """
        Determine if an update should be ignored based on content and chat type.
        In group chats:
        - Only respond when the bot is explicitly mentioned
        - Always ignore images/videos regardless of mention
        In private chats:
        - Respond to all messages (default behavior)
        Args:
            update_data: The update data received from Telegram
            bot_username: The bot's username for mention detection (deprecated, use context)
            context: Telegram context object for dynamic username detection
        Returns:
            True if the update should be ignored, False otherwise
        """
        try:
            if "message" not in update_data:
                return False
            message = update_data["message"]
            chat = message.get("chat", {})
            chat_type = chat.get("type", "")
            is_group = chat_type in ["group", "supergroup"]
            if not is_group:
                return False
            has_image = any(key in message for key in ["photo", "sticker", "animation"])
            has_video = "video" in message
            if has_image or has_video:
                self.logger.info(
                    f"Ignoring media in group chat, update_id: {update_data.get('update_id', 'unknown')}"
                )
                return True
            has_text = "text" in message
            if not has_text:
                return True
            if context:
                message_text = message.get("text", "")
                entities = message.get("entities", [])
                is_mentioned = BotUsernameHelper.is_bot_mentioned(
                    message_text, context, entities=entities
                )
            else:
                if not bot_username:
                    bot_username = "Gemini_AIAssistBot"
                message_text = message.get("text", "")
                mentioned_in_text = f"@{bot_username}" in message_text
                entities = message.get("entities", [])
                mentioned_in_entities = False
                for entity in entities:
                    if entity.get("type") == "mention":
                        start = entity.get("offset", 0)
                        length = entity.get("length", 0)
                        if start + length <= len(message_text):
                            mention_text = message_text[start : start + length]
                            if f"@{bot_username}" == mention_text:
                                mentioned_in_entities = True
                                break
                    elif entity.get("type") == "text_mention":
                        user = entity.get("user", {})
                        if (
                            user.get("is_bot", False)
                            and user.get("username") == bot_username
                        ):
                            mentioned_in_entities = True
                            break
                is_mentioned = mentioned_in_text or mentioned_in_entities
            if is_group and not is_mentioned:
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error in update filter: {str(e)}")
            return False

    def configure_filters(self, config):
        """
        Configure the message filter with custom settings.
        Args:
            config: Dictionary containing filter configuration options
        """
        pass


message_filter = MessageFilter()
