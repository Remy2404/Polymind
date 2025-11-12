"""
Telegram Mini Apps API Routes - Unified UUID Schema
Provides REST API endpoints with UUID-based session management.
Auto-authenticates users via init data without separate login flow.

This is a cleaner architecture that uses:
- UUID session IDs (not cache_key format)
- Proper message arrays stored in session documents
- Simpler query patterns
"""

import os
import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
from functools import lru_cache
import hashlib

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import JSONResponse, ORJSONResponse
from pydantic import BaseModel, Field
from urllib.parse import parse_qs, unquote_plus
import json

# Telegram init data validation
try:
    from telegram_init_data import parse, is_valid

    TELEGRAM_INIT_DATA_AVAILABLE = True
except ImportError:
    TELEGRAM_INIT_DATA_AVAILABLE = False
    logging.warning(
        "telegram-init-data not installed. Run: pip install telegram-init-data[fastapi]"
    )

from src.services.memory_context.conversation_manager import ConversationManager
from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager
from src.services.user_data_manager import UserDataManager
from src.services.model_handlers.factory import ModelHandlerFactory
from src.services.model_handlers.model_configs import ModelConfigurations

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webapp", tags=["Telegram Mini Apps - Unified Schema"])

# Global bot instance (set by app_factory)
_BOT_INSTANCE = None


def set_bot_instance(bot):
    """Set the TelegramBot instance for API access."""
    global _BOT_INSTANCE
    _BOT_INSTANCE = bot


# ============================================================================
# Pydantic Models - Unified Schema
# ============================================================================


class UserInfo(BaseModel):
    """User information extracted from init data."""

    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None
    language_code: Optional[str] = None
    is_premium: bool = False
    photo_url: Optional[str] = None


class Message(BaseModel):
    """Single message in AI SDK format."""

    id: str
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str
    createdAt: str
    model: Optional[str] = None


class ChatSession(BaseModel):
    """Chat session with UUID-based schema."""

    id: str  # UUID format
    title: str
    model: str
    created_at: str
    updated_at: str
    user_id: str
    message_count: int = 0
    messages: List[Message] = []
    pinned: bool = False
    pinned_at: Optional[str] = None


class Attachment(BaseModel):
    """File attachment for messages."""

    name: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    data: str = Field(..., description="Base64 encoded file content")


class ChatMessage(BaseModel):
    """Message sent to AI model."""

    message: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(
        ..., description="Model ID (e.g., 'gemini/gemini-2.0-flash-exp')"
    )
    session_id: Optional[str] = Field(
        None, description="Session UUID (creates new if not provided)"
    )
    include_context: bool = Field(True, description="Include conversation history")
    max_context_messages: int = Field(10, ge=1, le=50)
    attachments: Optional[List[Attachment]] = Field(
        None, description="File attachments (images, documents, etc.)"
    )


class ChatResponse(BaseModel):
    """AI model response."""

    response: str
    model_used: str
    session_id: str
    timestamp: float
    tokens_used: Optional[int] = None


class ModelInfo(BaseModel):
    """Available AI model information."""

    id: str
    name: str
    provider: str
    supports_vision: bool
    supports_tools: bool
    context_length: int
    is_free: bool
    accessible: bool = True


class UserPreferences(BaseModel):
    """User preferences settings."""

    preferred_model: Optional[str] = None
    theme: str = Field("light", pattern="^(light|dark)$")
    language: str = Field("en", min_length=2, max_length=5)
    notifications_enabled: bool = True


# ============================================================================
# Authentication Dependency
# ============================================================================


async def verify_telegram_init_data(
    authorization: str = Header(
        None, description="Authorization header with format: 'tma {init_data}'"
    ),
    user_id: str = Query(
        None,
        description="User ID for fallback authentication when outside Telegram context",
    ),
) -> Dict[str, Any]:
    """Verify and parse Telegram Mini Apps init data.

    Supports two authentication methods:
    1. Primary: Telegram initData (from mini app)
    2. Fallback: user_id URL parameter (when opened outside Telegram, e.g., "Open" button)
    """

    # Check if we're in development mode
    # dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

    # if dev_mode:
    #     logger.info("[Auth] Development mode enabled - bypassing Telegram authentication")
    #     # Return mock user data for development
    #     return {
    #         "user": {
    #             "id": 123456789,  # Mock user ID
    #             "first_name": "Dev",
    #             "last_name": "User",
    #             "username": "devuser",
    #             "language_code": "en",
    #             "is_premium": False
    #         }
    #     }

    if not TELEGRAM_INIT_DATA_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: telegram-init-data not installed",
        )

    # Try primary authentication: Telegram initData
    if authorization:
        if not authorization.startswith("tma "):
            logger.warning("[Auth] Invalid authorization format")
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization format. Expected: 'tma {initData}'",
            )

        init_data = authorization[4:]
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: TELEGRAM_BOT_TOKEN not set",
            )

        try:
            if not is_valid(init_data, bot_token):
                logger.warning("[Auth] Init data validation failed")
                raise HTTPException(
                    status_code=401, detail="Invalid init data signature or expired"
                )

            # Use the library parser first (strict)
            parsed_data = parse(init_data)

            if not parsed_data.get("user"):
                logger.error("[Auth] Parsed data missing user information")
                raise HTTPException(
                    status_code=400, detail="User data not found in init data"
                )

            user_id_from_data = parsed_data["user"].get("id")
            logger.info(
                f"[Auth] Successfully authenticated via initData user_id={user_id_from_data}"
            )
            return parsed_data

        except HTTPException:
            raise
        except ValueError as e:
            # Some initData payloads may contain unexpected enum values (e.g., chat_type='sender')
            # The strict parser may raise a ValueError. Attempt a tolerant manual parse as a fallback.
            logger.warning(
                f"[Auth] Strict parse failed, attempting tolerant parse: {e}"
            )
            try:
                qs = parse_qs(init_data)
                user_raw = None
                if "user" in qs:
                    user_raw = qs.get("user")[0]
                elif "user_id" in qs:
                    # older/alternate formats
                    user_raw = qs.get("user_id")[0]

                parsed_data_tolerant = {"auth_date": None, "hash": None, "user": None}

                if "auth_date" in qs:
                    parsed_data_tolerant["auth_date"] = int(qs.get("auth_date")[0])

                if "hash" in qs:
                    parsed_data_tolerant["hash"] = qs.get("hash")[0]

                if user_raw:
                    # user_raw may be URL-encoded JSON
                    try:
                        user_json = json.loads(unquote_plus(user_raw))
                        parsed_data_tolerant["user"] = user_json
                    except Exception:
                        # Try simple numeric user id
                        try:
                            parsed_data_tolerant["user"] = {"id": int(user_raw)}
                        except Exception:
                            parsed_data_tolerant["user"] = None

                if parsed_data_tolerant.get("user"):
                    logger.info("[Auth] Tolerant init data parse succeeded")
                    return parsed_data_tolerant
                else:
                    logger.error("[Auth] Tolerant parse failed to extract user")
                    raise HTTPException(
                        status_code=401,
                        detail="Init data validation failed (tolerant parse)",
                    )
            except HTTPException:
                raise
            except Exception as e2:
                logger.error(f"[Auth] Tolerant parse error: {e2}", exc_info=True)
                raise HTTPException(
                    status_code=401, detail=f"Init data validation failed: {str(e2)}"
                )
        except Exception as e:
            logger.error(f"[Auth] Init data validation error: {e}", exc_info=True)
            raise HTTPException(
                status_code=401, detail=f"Init data validation failed: {str(e)}"
            )

    # Fallback authentication: URL parameter user_id (for "Open" button outside Telegram context)
    if user_id:
        logger.info(
            f"[Auth] Using fallback authentication with user_id={user_id} (outside Telegram context)"
        )
        # Verify user_id is a valid integer
        try:
            parsed_user_id = int(user_id)
            # Return a minimal user object for fallback authentication
            # This allows the app to work when opened outside of Telegram Mini App
            return {
                "user": {
                    "id": parsed_user_id,
                    "first_name": "User",
                    "last_name": None,
                    "username": None,
                    "language_code": "en",
                    "is_premium": False,
                    "photo_url": None,
                },
                "auth_date": int(datetime.now().timestamp()),
                "hash": "fallback",  # Mark as fallback auth
            }
        except (ValueError, TypeError):
            logger.warning(f"[Auth] Invalid user_id format: {user_id}")
            raise HTTPException(
                status_code=400, detail="Invalid user_id format. Expected integer."
            )

    # No valid authentication method provided
    logger.warning(
        "[Auth] Request received without Authorization header or user_id parameter"
    )
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Include 'Authorization: tma {initData}' header or '?user_id={id}' parameter",
    )


async def get_current_user(
    init_data: Dict = Depends(verify_telegram_init_data),
) -> UserInfo:
    """Extract current user info from validated init data."""
    user_data = init_data.get("user", {})
    return UserInfo(
        id=user_data["id"],
        first_name=user_data.get("first_name", "User"),
        last_name=user_data.get("last_name"),
        username=user_data.get("username"),
        language_code=user_data.get("language_code"),
        is_premium=user_data.get("is_premium", False),
        photo_url=user_data.get("photo_url"),
    )


# ============================================================================
# Service Initialization Helpers
# ============================================================================


def get_services():
    """Get initialized service instances."""
    if not _BOT_INSTANCE:
        raise HTTPException(status_code=503, detail="Bot instance not initialized")

    if hasattr(_BOT_INSTANCE, "user_data_manager"):
        user_data_manager = _BOT_INSTANCE.user_data_manager
    else:
        db = _BOT_INSTANCE.db if hasattr(_BOT_INSTANCE, "db") else None
        user_data_manager = UserDataManager(db)

    memory_manager = MemoryManager(
        db=user_data_manager.db if hasattr(user_data_manager, "db") else None
    )
    model_history_manager = ModelHistoryManager(memory_manager)
    conversation_manager = ConversationManager(memory_manager, model_history_manager)

    return {
        "user_data_manager": user_data_manager,
        "memory_manager": memory_manager,
        "conversation_manager": conversation_manager,
        "gemini_api": (
            _BOT_INSTANCE.gemini_api if hasattr(_BOT_INSTANCE, "gemini_api") else None
        ),
        "openrouter_api": (
            _BOT_INSTANCE.openrouter_api
            if hasattr(_BOT_INSTANCE, "openrouter_api")
            else None
        ),
        "deepseek_api": (
            _BOT_INSTANCE.deepseek_api
            if hasattr(_BOT_INSTANCE, "deepseek_api")
            else None
        ),
    }


def get_sessions_collection():
    """Get MongoDB sessions collection for unified schema."""
    services = get_services()
    persistence_manager = services[
        "conversation_manager"
    ].memory_manager.persistence_manager

    if persistence_manager is None or not hasattr(persistence_manager, "db"):
        raise HTTPException(status_code=503, detail="Database not available")

    # Use a new collection for unified sessions
    return persistence_manager.db["chat_sessions"]


# Global cache for session mapping with TTL
_session_id_to_cache_key_cache = {}
_cache_timestamp = 0
_CACHE_TTL = 300


@lru_cache(maxsize=1000)
def _cached_md5(text: str) -> str:
    """Cached MD5 hash function for performance."""
    return hashlib.md5(text.encode()).hexdigest()


async def get_cached_session_mapping(current_user: UserInfo) -> dict:
    """Get cached session mapping with TTL-based refresh."""
    global _session_id_to_cache_key_cache, _cache_timestamp

    current_time = asyncio.get_event_loop().time()
    if current_time - _cache_timestamp > _CACHE_TTL:
        # Rebuild cache
        _session_id_to_cache_key_cache = await _build_session_mapping(current_user)
        _cache_timestamp = current_time
        logger.info(
            f"[Mapping] Rebuilt cache with {len(_session_id_to_cache_key_cache)} mappings"
        )

    return _session_id_to_cache_key_cache


async def _build_session_mapping(current_user: UserInfo) -> dict:
    """Build session mapping from database."""
    services = get_services()
    persistence_manager = services[
        "conversation_manager"
    ].memory_manager.persistence_manager

    if persistence_manager is None or not hasattr(persistence_manager, "db"):
        return {}

    user_id = str(current_user.id)
    user_prefix = f"user_{user_id}_model_"

    try:
        conversations_collection = persistence_manager.db["conversations"]
        # Use indexed query instead of regex for better performance
        session_docs = list(
            conversations_collection.find(
                {"cache_key": {"$regex": f"^{user_prefix}"}}
            ).limit(1000)
        )

        mapping = {}
        for doc in session_docs:
            cache_key = doc.get("cache_key", "")
            messages = doc.get("messages", [])

            if not messages:
                continue

            # Use cached MD5 for performance
            session_id = _cached_md5(cache_key)
            mapping[session_id] = cache_key

        return mapping

    except Exception as e:
        logger.error(f"[Mapping] Error building session mapping: {e}")
        return {}


async def build_session_id_mapping(current_user: UserInfo):
    """Legacy function - now uses cached mapping."""
    # This function is kept for backward compatibility
    # The actual mapping is now handled by get_cached_session_mapping
    pass


async def get_old_format_sessions(
    current_user: UserInfo, include_messages: bool = False
) -> List[ChatSession]:
    """
    Get chat sessions from the old cache_key format memory system.
    This bridges the gap until all sessions are migrated to the new format.
    """
    services = get_services()
    persistence_manager = services[
        "conversation_manager"
    ].memory_manager.persistence_manager

    if persistence_manager is None or not hasattr(persistence_manager, "db"):
        logger.warning("[Old Sessions] Database not available for old format sessions")
        return []

    user_id = str(current_user.id)
    user_prefix = f"user_{user_id}_model_"

    try:
        # Query the conversations collection for documents where cache_key starts with user_{user_id}_model_
        conversations_collection = persistence_manager.db["conversations"]

        # Find all documents matching the pattern
        session_docs = list(
            conversations_collection.find({"cache_key": {"$regex": f"^{user_prefix}"}})
        )

        logger.info(
            f"[Old Sessions] Found {len(session_docs)} old format conversations for user {user_id}"
        )

        sessions = []
        for doc in session_docs:
            cache_key = doc.get("cache_key", "")
            messages = doc.get("messages", [])
            message_count = len(messages)
            last_updated = doc.get("last_updated", 0)

            # Skip if no messages
            if message_count == 0:
                continue

            # Parse model from cache_key: user_{user_id}_model_{model}
            try:
                model_part = cache_key.split("_model_")[1]
                # Handle cases where model might have slashes or special chars
                model = model_part.replace("/", "/")  # Ensure proper format
            except IndexError:
                logger.warning(
                    f"[Old Sessions] Could not parse model from cache_key: {cache_key}"
                )
                continue

            # Create a pseudo-UUID from the cache_key for frontend compatibility
            # This ensures the same chat always gets the same ID
            session_id = _cached_md5(cache_key)

            # Get title from first user message if possible
            title = f"Chat with {model}"
            if messages:
                # Find first user message
                first_user_msg = next(
                    (msg for msg in messages if msg.get("role") == "user"), None
                )
                if first_user_msg:
                    content = first_user_msg.get("content", "")
                    title = content[:50] + ("..." if len(content) > 50 else "")

            # Convert messages to Message format if requested
            message_objects = []
            if include_messages:
                for i, msg in enumerate(messages):
                    message_objects.append(
                        Message(
                            id=msg.get("id", f"msg_{i}"),
                            role=msg.get("role", "user"),
                            content=msg.get("content", ""),
                            createdAt=msg.get(
                                "createdAt",
                                msg.get(
                                    "timestamp",
                                    datetime.fromtimestamp(last_updated).isoformat(),
                                ),
                            ),
                            model=msg.get("model", model),
                        )
                    )

            # Convert timestamp to ISO format
            updated_at = (
                datetime.fromtimestamp(last_updated).isoformat()
                if isinstance(last_updated, (int, float))
                else datetime.now().isoformat()
            )
            created_at = updated_at  # Use same timestamp for both

            sessions.append(
                ChatSession(
                    id=session_id,  # Use hash of cache_key as ID
                    title=title,
                    model=model,
                    created_at=created_at,
                    updated_at=updated_at,
                    user_id=user_id,
                    message_count=message_count,
                    messages=message_objects,
                    pinned=False,  # Old format doesn't support pinning
                    pinned_at=None,
                )
            )

        logger.info(f"[Old Sessions] Converted {len(sessions)} old format sessions")
        return sessions

    except Exception as e:
        logger.error(
            f"[Old Sessions] Error retrieving old format sessions: {e}", exc_info=True
        )
        return []


# ============================================================================
# API Endpoints - Unified Schema
# ============================================================================


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
        logger.error(f"[Preferences] Error updating preferences: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to update preferences: {str(e)}"
        )


@router.post("/v2/sessions", response_model=ChatSession)
async def create_chat_session(
    model: str = Query(..., description="Model ID for this session"),
    title: Optional[str] = Query(
        None, description="Session title (auto-generated if not provided)"
    ),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Create a new chat session with UUID.

    Returns a new session object that can be used to send messages.
    """
    sessions_collection = get_sessions_collection()
    user_id = str(current_user.id)

    try:
        # Validate model exists
        all_models = ModelConfigurations.get_all_models()
        if model not in all_models:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}")

        # Generate UUID for session
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        # Create session document
        session_doc = {
            "_id": session_id,
            "session_id": session_id,
            "user_id": user_id,
            "model": model,
            "title": title or "New Chat",
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "pinned": False,
            "pinned_at": None,
        }

        sessions_collection.insert_one(session_doc)

        logger.info(
            f"[Sessions] Created new session {session_id} for user {user_id} with model {model}"
        )

        return ChatSession(
            id=session_id,
            title=session_doc["title"],
            model=model,
            created_at=now,
            updated_at=now,
            user_id=user_id,
            message_count=0,
            messages=[],
            pinned=False,
            pinned_at=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Sessions] Error creating session: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to create session: {str(e)}"
        )


@router.get(
    "/v2/sessions", response_model=List[ChatSession], response_class=ORJSONResponse
)
async def get_chat_sessions(
    current_user: UserInfo = Depends(get_current_user),
    include_messages: bool = Query(False, description="Include full message arrays"),
):
    """
    Get user's chat sessions with UUID-based schema.

    Returns a list of chat sessions with proper message counts and titles.
    Much simpler than legacy cache_key approach.
    """
    sessions_collection = get_sessions_collection()
    user_id = str(current_user.id)

    try:
        logger.info(f"[Sessions] Querying unified sessions for user_id={user_id}")

        # Simple query: just find sessions for this user
        session_docs = list(sessions_collection.find({"user_id": user_id}))

        logger.info(
            f"[Sessions] Found {len(session_docs)} unified sessions for user {user_id}"
        )

        sessions = []
        for doc in session_docs:
            messages_data = doc.get("messages", [])
            message_count = len([m for m in messages_data if m.get("role") == "user"])

            # Convert messages to Pydantic models if requested
            messages = []
            if include_messages:
                messages = [
                    Message(
                        id=msg.get("id", ""),
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        createdAt=msg.get("createdAt", datetime.now().isoformat()),
                        model=msg.get("model"),
                    )
                    for msg in messages_data
                ]

            sessions.append(
                ChatSession(
                    id=doc.get("session_id", doc.get("_id")),
                    title=doc.get("title", "New Chat"),
                    model=doc.get("model", "unknown"),
                    created_at=doc.get("created_at", datetime.now().isoformat()),
                    updated_at=doc.get("updated_at", datetime.now().isoformat()),
                    user_id=user_id,
                    message_count=message_count,
                    messages=messages,
                    pinned=doc.get("pinned", False),
                    pinned_at=doc.get("pinned_at"),
                )
            )

        # Sort by updated_at (most recent first)
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        logger.info(f"[Sessions] Returning {len(sessions)} unified sessions")
        return sessions

    except Exception as e:
        logger.error(
            f"[Sessions] Error retrieving unified sessions: {e}", exc_info=True
        )
        return []


@router.get(
    "/sessions", response_model=List[ChatSession], response_class=ORJSONResponse
)
async def get_chat_sessions_legacy(
    current_user: UserInfo = Depends(get_current_user),
    include_messages: bool = Query(False, description="Include full message arrays"),
):
    """
    Legacy endpoint for fetching chat sessions.
    Returns sessions from both new unified schema and old cache_key format.
    """
    # Get sessions from new unified schema
    new_sessions = await get_chat_sessions(current_user, include_messages)

    # Also get sessions from old memory system
    old_sessions = await get_old_format_sessions(current_user, include_messages)

    # Combine and sort by updated_at (most recent first)
    all_sessions = new_sessions + old_sessions
    all_sessions.sort(key=lambda s: s.updated_at, reverse=True)

    logger.info(
        f"[Sessions] Returning {len(all_sessions)} total sessions ({len(new_sessions)} new, {len(old_sessions)} old)"
    )
    return all_sessions


@router.get(
    "/v2/sessions/{session_id}/messages",
    response_model=List[Message],
    response_class=ORJSONResponse,
)
async def get_session_messages(
    session_id: str, current_user: UserInfo = Depends(get_current_user)
):
    """
    Get messages for a specific session (UUID-based).

    Much simpler than legacy approach - just query by UUID.
    """
    sessions_collection = get_sessions_collection()
    user_id = str(current_user.id)

    try:
        logger.info(f"[Messages] Fetching messages for session {session_id}")

        # Find session by UUID
        session_doc = sessions_collection.find_one({"session_id": session_id})

        if not session_doc:
            logger.warning(f"[Messages] Session {session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")

        # Validate ownership
        if session_doc.get("user_id") != user_id:
            logger.warning(
                f"[Messages] Access denied: session {session_id} belongs to different user"
            )
            raise HTTPException(status_code=403, detail="Access denied")

        # Get messages array
        messages_data = session_doc.get("messages", [])

        logger.info(
            f"[Messages] Found {len(messages_data)} messages for session {session_id}"
        )

        # Convert to Pydantic models
        messages = [
            Message(
                id=msg.get("id", ""),
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                createdAt=msg.get("createdAt", datetime.now().isoformat()),
                model=msg.get("model"),
            )
            for msg in messages_data
        ]

        return messages

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Messages] Error retrieving messages: {e}", exc_info=True)
        return []


@router.get(
    "/messages/{chat_id:path}",
    response_model=List[Message],
    response_class=ORJSONResponse,
)
async def get_messages_legacy(
    chat_id: str,
    model: Optional[str] = Query(
        None, description="Model ID (for UUID format chat_ids)"
    ),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Legacy endpoint for fetching messages.
    Supports both cache_key format (user_{user_id}_model_{model}), UUID format, and hashed cache_key format.
    For cache_key format, retrieves from old memory system.
    For UUID format, retrieves from new sessions collection.
    For hashed format, maps back to cache_key and retrieves from old system.
    """
    logger.info(f"[Messages] Function called with chat_id: {chat_id}")
    from urllib.parse import unquote

    user_id = str(current_user.id)

    # URL decode the chat_id to handle encoded characters like %3A
    chat_id = unquote(chat_id)
    logger.info(f"[Messages] After unquote, chat_id: {chat_id}")

    try:
        logger.info(f"[Messages] Legacy fetch for chat_id {chat_id}")

        # Check if this is a hashed cache_key (32-character hex string)
        session_mapping = await get_cached_session_mapping(current_user)
        if chat_id in session_mapping:
            # This is a hashed cache_key, map it back to the original cache_key
            original_cache_key = session_mapping[chat_id]
            logger.info(
                f"[Messages] Mapped hashed ID {chat_id} back to cache_key: {original_cache_key}"
            )
            chat_id = original_cache_key

        # Handle cache_key format: user_{user_id}_model_{model}
        if chat_id.startswith("user_") and "_model_" in chat_id:
            logger.info(f"[Messages] Detected cache_key format for {chat_id}")
            # Parse cache_key format
            parts = chat_id.split("_model_")
            logger.info(f"[Messages] Split parts: {parts}")
            if len(parts) == 2:
                cache_key_user_id = parts[0].replace("user_", "")
                model_from_key = parts[1]

                logger.info(
                    f"[Messages] Parsed user_id={cache_key_user_id}, model={model_from_key}"
                )

                # Validate ownership
                if cache_key_user_id != user_id:
                    logger.warning(f"[Messages] Access denied for cache_key {chat_id}")
                    raise HTTPException(status_code=403, detail="Access denied")

                # Retrieve messages from old memory system
                try:
                    services = get_services()
                    conversation_manager = services["conversation_manager"]

                    logger.info(
                        f"[Messages] Fetching conversation history for user {user_id} with model {model_from_key}"
                    )

                    # Get conversation history for this specific model
                    context_messages = (
                        await conversation_manager.get_conversation_history(
                            user_id=int(user_id),
                            model=model_from_key,
                            max_messages=100,  # Get all messages
                        )
                    )

                    logger.info(
                        f"[Messages] Retrieved {len(context_messages)} raw messages from conversation manager"
                    )
                    if context_messages:
                        logger.debug(
                            f"[Messages] First message sample: {context_messages[0]}"
                        )
                        logger.debug(
                            f"[Messages] Message keys: {list(context_messages[0].keys()) if context_messages[0] else 'None'}"
                        )

                    # If no messages from conversation manager, try model_history_manager directly
                    if not context_messages:
                        logger.info(
                            "[Messages] No messages from conversation manager, trying model_history_manager directly"
                        )
                        model_history_manager = services[
                            "conversation_manager"
                        ].model_history_manager
                        context_messages = await model_history_manager.get_history(
                            user_id=int(user_id),
                            max_messages=100,
                            model_id=model_from_key,
                        )
                        logger.info(
                            f"[Messages] Retrieved {len(context_messages)} messages from model_history_manager"
                        )

                    # Convert to Message format
                    messages = []
                    for i, msg in enumerate(context_messages):
                        # Handle different message formats
                        if "role" in msg and "content" in msg:
                            # Model history format: {"role": "user", "content": "..."}
                            message = Message(
                                id=msg.get("id", f"msg_{i}"),
                                role=msg.get("role", "user"),
                                content=msg.get("content", ""),
                                createdAt=msg.get(
                                    "createdAt",
                                    msg.get("timestamp", datetime.now().isoformat()),
                                ),
                                model=msg.get("model", model_from_key),
                            )
                        else:
                            # Enhanced format from conversation manager
                            message = Message(
                                id=msg.get("id", f"msg_{i}"),
                                role=msg.get("role", "user"),
                                content=msg.get("content", ""),
                                createdAt=msg.get(
                                    "createdAt",
                                    msg.get("timestamp", datetime.now().isoformat()),
                                ),
                                model=msg.get("model", model_from_key),
                            )
                        messages.append(message)

                    logger.info(
                        f"[Messages] Converted to {len(messages)} Message objects for {chat_id}"
                    )
                    return messages

                except Exception as e:
                    logger.error(
                        f"[Messages] Error retrieving from old memory system: {e}",
                        exc_info=True,
                    )
                    return []
            else:
                logger.warning(f"[Messages] Invalid cache_key format: {chat_id}")
                raise HTTPException(status_code=400, detail="Invalid chat_id format")
        else:
            logger.info(f"[Messages] Assuming UUID format for {chat_id}")
            # Assume UUID format - use new sessions collection
            return await get_session_messages(chat_id, current_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Messages] Error in legacy endpoint: {e}", exc_info=True)
        return []


@router.post("/chat", response_model=ChatResponse)
async def send_chat_message(
    message_data: ChatMessage, current_user: UserInfo = Depends(get_current_user)
):
    """
    Send message to AI model using unified UUID session schema.

    Creates new session if session_id not provided.
    Stores messages in session document for easy retrieval.
    """
    services = get_services()
    sessions_collection = get_sessions_collection()
    user_id = str(current_user.id)

    try:
        # Get or create session
        session_id = message_data.session_id
        session_doc = None

        if session_id:
            session_doc = sessions_collection.find_one({"session_id": session_id})
            if not session_doc:
                logger.warning(f"[Chat] Session {session_id} not found, creating new")
                session_id = None

        if not session_id:
            # Create new session
            session_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            session_doc = {
                "_id": session_id,
                "session_id": session_id,
                "user_id": user_id,
                "model": message_data.model,
                "title": message_data.message[:50]
                + ("..." if len(message_data.message) > 50 else ""),
                "messages": [],
                "created_at": now,
                "updated_at": now,
                "pinned": False,
                "pinned_at": None,
            }
            sessions_collection.insert_one(session_doc)
            logger.info(f"[Chat] Created new session {session_id}")

        # Validate ownership
        if session_doc.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Get conversation context if requested
        context_messages = []
        if message_data.include_context:
            messages_data = session_doc.get("messages", [])
            # Take last N messages for context
            context_messages = messages_data[-message_data.max_context_messages :]

        # Get model handler
        model_handler = ModelHandlerFactory.get_model_handler(
            model_name=message_data.model,
            gemini_api=services["gemini_api"],
            openrouter_api=services["openrouter_api"],
            deepseek_api=services["deepseek_api"],
        )

        # Format prompt with context
        prompt = message_data.message
        if context_messages:
            context_parts = []
            for msg in context_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    context_parts.append(f"User: {content}")
                elif role == "assistant":
                    context_parts.append(f"Assistant: {content}")

            if context_parts:
                context_str = "\n".join(context_parts)
                prompt = f"Previous conversation:\n{context_str}\n\nCurrent message: {message_data.message}"

        # Generate response
        response = await model_handler.generate_response(
            prompt, model=message_data.model
        )

        # Save messages to session
        now = datetime.now().isoformat()
        user_msg = {
            "id": f"{session_id}_{len(session_doc.get('messages', []))}",
            "role": "user",
            "content": message_data.message,
            "createdAt": now,
            "model": message_data.model,
        }
        assistant_msg = {
            "id": f"{session_id}_{len(session_doc.get('messages', [])) + 1}",
            "role": "assistant",
            "content": response,
            "createdAt": now,
            "model": message_data.model,
        }

        sessions_collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": {"$each": [user_msg, assistant_msg]}},
                "$set": {"updated_at": now},
            },
        )

        logger.info(f"[Chat] Saved message pair to session {session_id}")

        return ChatResponse(
            response=response,
            model_used=message_data.model,
            session_id=session_id,
            timestamp=datetime.now().timestamp(),
            tokens_used=None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Chat] Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate response: {str(e)}"
        )


@router.get("/models", response_model=List[ModelInfo])
async def get_available_models(
    current_user: UserInfo = Depends(get_current_user),
    vision_only: bool = Query(False),
    free_only: bool = Query(False),
):
    """Get list of available AI models."""
    try:
        all_models = ModelConfigurations.get_all_models()

        models = []
        for model_id, config in all_models.items():
            if vision_only and not config.supports_images:
                continue
            if free_only and not getattr(config, "is_free", False):
                continue

            capabilities = ModelConfigurations.get_model_capabilities(model_id)

            models.append(
                ModelInfo(
                    id=model_id,
                    name=config.display_name,
                    provider=config.provider.value,
                    supports_vision=config.supports_images,
                    supports_tools=capabilities.get("supports_tools", False),
                    context_length=config.max_tokens,
                    is_free=getattr(config, "is_free", False),
                    accessible=True,
                )
            )

        return models

    except Exception as e:
        logger.error(f"[Models] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def webapp_health_check():
    """Health check for unified schema API."""
    return JSONResponse(
        content={
            "status": "ok",
            "service": "Telegram Mini Apps API - Unified Schema",
            "schema_version": "2.0",
            "telegram_init_data_available": TELEGRAM_INIT_DATA_AVAILABLE,
            "bot_initialized": _BOT_INSTANCE is not None,
            "timestamp": datetime.now().isoformat(),
        }
    )


# ============================================================================
# Legacy /chats endpoints for backward compatibility
# ============================================================================


@router.post("/chats")
async def create_chat_legacy(
    title: Optional[str] = Query(None, description="Chat title"),
    model: Optional[str] = Query(None, description="Model ID"),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Legacy endpoint for creating chats.
    Creates a session and returns chat-compatible format.
    """
    # Use the v2/sessions endpoint logic
    session = await create_chat_session(model, title, current_user)

    # Convert to chat format expected by frontend
    return {
        "id": session.id,
        "title": session.title,
        "model": session.model,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "message_count": session.message_count,
        "user_id": session.user_id,
        "public": True,
        "pinned": session.pinned,
        "pinned_at": session.pinned_at,
    }


@router.get("/chats")
async def list_chats_legacy(
    current_user: UserInfo = Depends(get_current_user),
    limit: int = Query(50, description="Max number of chats to return"),
    offset: int = Query(0, description="Number of chats to skip"),
):
    """
    Legacy endpoint for listing chats.
    Returns sessions in chat-compatible format.
    """
    # Get sessions using v2 endpoint
    sessions = await get_chat_sessions(current_user, False)

    # Convert to chat format and apply pagination
    chats = []
    for session in sessions[offset : offset + limit]:
        chats.append(
            {
                "id": session.id,
                "title": session.title,
                "model": session.model,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": session.message_count,
                "user_id": session.user_id,
                "public": True,
                "pinned": session.pinned,
                "pinned_at": session.pinned_at,
            }
        )

    return chats


@router.get("/history")
async def list_history_legacy(
    current_user: UserInfo = Depends(get_current_user),
    limit: int = Query(50, description="Max number of chats to return"),
    offset: int = Query(0, description="Number of chats to skip"),
):
    """
    Legacy endpoint for listing chat history.
    Alias for /chats endpoint.
    """
    return await list_chats_legacy(current_user, limit, offset)


@router.delete("/chats/{chat_id:path}")
async def delete_chat_legacy(
    chat_id: str, current_user: UserInfo = Depends(get_current_user)
):
    """
    Legacy endpoint for deleting chats.
    Supports:
    1. UUID-based sessions (new format)
    2. Cache_key format (user_{user_id}_model_{model}) - ALLOWS SLASHES IN PATH
    3. Hashed session IDs (MD5 hash of cache_key)
    Deletes the corresponding session from the appropriate collection.
    """
    from urllib.parse import unquote

    # Decode URL-encoded characters (e.g., %3A -> :, %2F -> /)
    chat_id = unquote(chat_id)

    sessions_collection = get_sessions_collection()
    user_id = str(current_user.id)
    deleted = False

    try:
        # Try deleting from new format (chat_sessions collection with session_id)
        result = sessions_collection.delete_one(
            {"session_id": chat_id, "user_id": user_id}
        )

        if result.deleted_count > 0:
            deleted = True
            logger.info(
                f"[Chats] Deleted new format session {chat_id} for user {user_id}"
            )
        else:
            # Try deleting from old format (conversations collection)
            from src.database.connection import get_database

            db, _ = get_database()

            if db is not None:
                conversations = db.conversations

                # Case 1: chat_id is the cache_key directly (e.g., user_806762900_model_gemini)
                if chat_id.startswith(f"user_{user_id}_model_"):
                    result = conversations.delete_one({"cache_key": chat_id})

                    if result.deleted_count > 0:
                        deleted = True
                        logger.info(
                            f"[Chats] Deleted old format conversation {chat_id} for user {user_id}"
                        )
                else:
                    # Case 2: chat_id is a hashed session ID - look up the original cache_key
                    session_mapping = await get_cached_session_mapping(current_user)
                    if chat_id in session_mapping:
                        cache_key = session_mapping[chat_id]

                        # Verify it belongs to this user
                        if cache_key.startswith(f"user_{user_id}_model_"):
                            result = conversations.delete_one({"cache_key": cache_key})

                            if result.deleted_count > 0:
                                deleted = True
                                logger.info(
                                    f"[Chats] Deleted hashed session {chat_id} (cache_key: {cache_key}) for user {user_id}"
                                )
                        else:
                            logger.warning(
                                f"[Chats] Hashed session {chat_id} resolved to cache_key {cache_key} which doesn't belong to user {user_id}"
                            )
                    else:
                        # Not in mapping - might be an invalid or very old chat ID
                        logger.warning(
                            f"[Chats] Chat ID {chat_id} not found in any format for user {user_id}"
                        )

        if not deleted:
            raise HTTPException(status_code=404, detail="Chat not found")

        return {"status": "success", "message": "Chat deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Chats] Error deleting chat {chat_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")
