import logging
from typing import Any, Dict


class UserPreferencesManager:
    """Manages user preferences for the bot."""

    def __init__(self, user_data_manager):
        self.logger = logging.getLogger(__name__)
        self.user_data_manager = user_data_manager

    async def get_user_model_preference(self, user_id: int) -> str:
        """Get the user's preferred AI model."""
        return await self.user_data_manager.get_user_preference(
            user_id, "preferred_model", default="gemini"
        )

    async def set_user_model_preference(self, user_id: int, model_name: str) -> None:
        """Set the user's preferred AI model."""
        await self.user_data_manager.set_user_preference(
            user_id, "preferred_model", model_name
        )

    async def get_user_language_preference(self, user_id: int) -> str:
        """Get the user's preferred language."""
        return await self.user_data_manager.get_user_preference(
            user_id, "preferred_language", default="en"
        )

    async def set_user_language_preference(self, user_id: int, language: str) -> None:
        """Set the user's preferred language."""
        await self.user_data_manager.set_user_preference(
            user_id, "preferred_language", language
        )

    async def get_markdown_enabled(self, user_id: int) -> bool:
        """Check if markdown formatting is enabled for the user."""
        settings = await self.user_data_manager.get_user_settings(user_id)
        return settings.get("markdown_enabled", True)

    async def set_markdown_enabled(self, user_id: int, enabled: bool) -> None:
        """Set whether markdown formatting is enabled for the user."""
        await self.user_data_manager.set_user_setting(
            user_id, "markdown_enabled", enabled
        )

    async def get_code_suggestions_enabled(self, user_id: int) -> bool:
        """Check if code suggestions are enabled for the user."""
        settings = await self.user_data_manager.get_user_settings(user_id)
        return settings.get("code_suggestions", True)

    async def set_code_suggestions_enabled(self, user_id: int, enabled: bool) -> None:
        """Set whether code suggestions are enabled for the user."""
        await self.user_data_manager.set_user_setting(
            user_id, "code_suggestions", enabled
        )

    async def get_all_preferences(self, user_id: int) -> Dict[str, Any]:
        """Get all preferences for a user."""
        # Get model preference
        model = await self.get_user_model_preference(user_id)

        # Get language preference
        language = await self.get_user_language_preference(user_id)

        # Get settings
        settings = await self.user_data_manager.get_user_settings(user_id)

        # Combine all preferences
        return {
            "preferred_model": model,
            "preferred_language": language,
            "markdown_enabled": settings.get("markdown_enabled", True),
            "code_suggestions": settings.get("code_suggestions", True),
        }
