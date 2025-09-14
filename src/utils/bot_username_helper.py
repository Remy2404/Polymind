"""
Bot Username Helper Module
This module provides utilities for dynamic bot username detection
to support developer team testing with different bot names.
Perfect for clone projects where each developer might use different bot instances.
"""
import logging
from typing import Optional
from telegram.ext import ContextTypes
logger = logging.getLogger(__name__)
class BotUsernameHelper:
    """
    Helper class for dynamic bot username detection.
    This class provides consistent bot username handling across the entire codebase,
    making it easy for development teams to test with different bot instances
    without hardcoding usernames.
    """
    @classmethod
    def get_bot_username(
        cls,
        context: Optional[ContextTypes.DEFAULT_TYPE] = None,
        bot=None,
        with_at: bool = True,
    ) -> str:
        """
        Get the bot's username dynamically from context or bot instance.
        Args:
            context: Telegram context object (preferred)
            bot: Bot instance (alternative if context not available)
            with_at: Whether to include '@' prefix (default: True)
        Returns:
            Bot username with or without '@' prefix
        Examples:
            >>> get_bot_username(context)
            >>> get_bot_username(context, with_at=False)
        """
        username = None
        if context and hasattr(context, "bot") and context.bot:
            try:
                username = context.bot.username
                if username:
                    logger.debug(f"Bot username from context: {username}")
            except Exception as e:
                logger.warning(f"Failed to get username from context: {e}")
        if not username and bot:
            try:
                username = bot.username
                if username:
                    logger.debug(f"Bot username from bot instance: {username}")
            except Exception as e:
                logger.warning(f"Failed to get username from bot instance: {e}")
        if not username:
            logger.warning("Bot username could not be determined.")
            username = ""
        if with_at and username and not username.startswith("@"):
            return f"@{username}"
        elif not with_at and username.startswith("@"):
            return username[1:]
        return username
    @classmethod
    def is_bot_mentioned(
        cls,
        text: str,
        context: Optional[ContextTypes.DEFAULT_TYPE] = None,
        bot=None,
        entities: list = None,
    ) -> bool:
        """
        Check if the bot is mentioned in the given text.
        Args:
            text: Message text to check
            context: Telegram context object
            bot: Bot instance (alternative)
            entities: Telegram entities for mention detection
        Returns:
            True if bot is mentioned, False otherwise
        """
        if not text:
            return False
        bot_username = cls.get_bot_username(context, bot, with_at=True)
        mentioned_in_text = bot_username in text
        mentioned_in_entities = False
        if entities:
            bot_username_no_at = cls.get_bot_username(context, bot, with_at=False)
            for entity in entities:
                if entity.get("type") == "mention":
                    start = entity.get("offset", 0)
                    length = entity.get("length", 0)
                    if start + length <= len(text):
                        mention_text = text[start : start + length]
                        if mention_text == bot_username:
                            mentioned_in_entities = True
                            break
                elif entity.get("type") == "text_mention":
                    user = entity.get("user", {})
                    if (
                        user.get("is_bot", False)
                        and user.get("username") == bot_username_no_at
                    ):
                        mentioned_in_entities = True
                        break
        is_mentioned = mentioned_in_text or mentioned_in_entities
        if is_mentioned:
            logger.debug(f"Bot mentioned in text: {bot_username}")
        return is_mentioned
    @classmethod
    def remove_bot_mention(
        cls, text: str, context: Optional[ContextTypes.DEFAULT_TYPE] = None, bot=None
    ) -> str:
        """
        Remove bot mention from text.
        Args:
            text: Original text with potential mention
            context: Telegram context object
            bot: Bot instance (alternative)
        Returns:
            Text with bot mention removed and stripped
        """
        if not text:
            return text
        bot_username = cls.get_bot_username(context, bot, with_at=True)
        cleaned_text = text.replace(bot_username, "").strip()
        logger.debug(f"Removed bot mention '{bot_username}' from text")
        return cleaned_text
    @classmethod
    def get_bot_info_summary(
        cls, context: Optional[ContextTypes.DEFAULT_TYPE] = None, bot=None
    ) -> dict:
        """
        Get a summary of bot information for debugging/logging.
        Args:
            context: Telegram context object
            bot: Bot instance (alternative)
        Returns:
            Dictionary with bot information
        """
        bot_instance = None
        if context and hasattr(context, "bot"):
            bot_instance = context.bot
        elif bot:
            bot_instance = bot
        if not bot_instance:
            return {
                "username": "",
                "source": "not_available",
                "id": None,
                "first_name": None,
                "available": False,
            }
        try:
            return {
                "username": cls.get_bot_username(context, bot, with_at=False),
                "username_with_at": cls.get_bot_username(context, bot, with_at=True),
                "source": "context" if context else "bot_instance",
                "id": getattr(bot_instance, "id", None),
                "first_name": getattr(bot_instance, "first_name", None),
                "available": True,
            }
        except Exception as e:
            logger.error(f"Error getting bot info: {e}")
            return {
                "username": "",
                "source": "error",
                "error": str(e),
                "available": False,
            }
def get_bot_username(
    context: Optional[ContextTypes.DEFAULT_TYPE] = None, bot=None, with_at: bool = True
) -> str:
    """Convenience function to get bot username."""
    return BotUsernameHelper.get_bot_username(context, bot, with_at)
def is_bot_mentioned(
    text: str,
    context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    bot=None,
    entities: list = None,
) -> bool:
    """Convenience function to check if bot is mentioned."""
    return BotUsernameHelper.is_bot_mentioned(text, context, bot, entities)
def remove_bot_mention(
    text: str, context: Optional[ContextTypes.DEFAULT_TYPE] = None, bot=None
) -> str:
    """Convenience function to remove bot mention from text."""
    return BotUsernameHelper.remove_bot_mention(text, context, bot)
