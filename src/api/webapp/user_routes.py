import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.auth import get_current_user
from src.api.schemas.webapp import UserInfo, UserPreferences

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webapp", tags=["Telegram Mini Apps - Unified Schema"])


@router.get("/user", response_model=UserInfo)
async def get_user_info(current_user: UserInfo = Depends(get_current_user)):
    """Get current authenticated user information."""
    return current_user


@router.get("/user/preferences", response_model=UserPreferences)
async def get_user_preferences(current_user: UserInfo = Depends(get_current_user)):
    """Get user preferences."""
    try:
        # Try to get preferences from UserPreferencesManager if available
        try:
            from src.services.user_preferences_manager import UserPreferencesManager

            prefs_manager = UserPreferencesManager()
            user_prefs = prefs_manager.get_user_preferences(current_user.id)
            return UserPreferences(**user_prefs)
        except ImportError:
            logger.warning("UserPreferencesManager not available, returning defaults")
            return UserPreferences()
    except Exception as e:
        logger.error(f"[Preferences] Error getting preferences: {e}", exc_info=True)
        return UserPreferences()


@router.post("/user/preferences")
async def update_user_preferences(
    preferences: UserPreferences, current_user: UserInfo = Depends(get_current_user)
):
    """Update user preferences."""
    try:
        # Try to save preferences using UserPreferencesManager if available
        try:
            from src.services.user_preferences_manager import UserPreferencesManager

            prefs_manager = UserPreferencesManager()
            prefs_manager.save_user_preferences(current_user.id, preferences.dict())
            return {"status": "success", "message": "Preferences updated"}
        except ImportError:
            logger.warning(
                "UserPreferencesManager not available, preferences not saved"
            )
            return {
                "status": "warning",
                "message": "Preferences not persisted (service unavailable)",
            }
    except Exception as e:
        logger.error("[Preferences] Update failed: %s", type(e).__name__, stack_info=True)
        raise HTTPException(status_code=500, detail="Failed to update preferences") from e
