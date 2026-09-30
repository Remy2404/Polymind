import logging
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import ORJSONResponse

from src.ai.models.model_configs import ModelConfigurations
from src.api.auth import get_current_user
from src.api.dependencies import get_sessions_collection
from src.api.schemas.webapp import ChatSession, Message, UserInfo
from src.api.webapp.legacy import get_cached_session_mapping, get_old_format_sessions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webapp", tags=["Telegram Mini Apps - Unified Schema"])


@router.post("/v2/sessions", response_model=ChatSession)
async def create_chat_session(
    request: Request,
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
    sessions_collection = get_sessions_collection(request)
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
        logger.error("[Sessions] Creation failed: %s", type(e).__name__, stack_info=True)
        raise HTTPException(status_code=500, detail="Failed to create session") from e


@router.get(
    "/v2/sessions", response_model=List[ChatSession], response_class=ORJSONResponse
)
async def get_chat_sessions(
    request: Request,
    current_user: UserInfo = Depends(get_current_user),
    include_messages: bool = Query(False, description="Include full message arrays"),
    limit: int = Query(50, description="Maximum number of sessions to return"),
    offset: int = Query(0, description="Sessions to skip"),
):
    """
    Get user's chat sessions with UUID-based schema.

    Returns a list of chat sessions with proper message counts and titles.
    Much simpler than legacy cache_key approach.
    """
    sessions_collection = get_sessions_collection(request)
    user_id = str(current_user.id)

    try:
        logger.info(f"[Sessions] Querying unified sessions for user_id={user_id}")

        # Simple query: just find sessions for this user.
        # Add pagination to avoid transferring huge result sets to the frontend
        if include_messages:
            session_docs = list(
                sessions_collection.find(
                    {"user_id": user_id},
                    {
                        "session_id": 1,
                        "title": 1,
                        "model": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        "messages": 1,
                        "pinned": 1,
                        "pinned_at": 1,
                    },
                )
                .skip(offset)
                .limit(limit)
            )
        else:
            pipeline = [
                {"$match": {"user_id": user_id}},
                {
                    "$project": {
                        "session_id": 1,
                        "title": 1,
                        "model": 1,
                        "created_at": 1,
                        "updated_at": 1,
                        "pinned": 1,
                        "pinned_at": 1,
                        "message_count": {"$size": {"$ifNull": ["$messages", []]}},
                    }
                },
                {"$skip": offset},
                {"$limit": limit},
            ]
            session_docs = list(sessions_collection.aggregate(pipeline))

        logger.info(
            f"[Sessions] Found {len(session_docs)} unified sessions for user {user_id}"
        )

        sessions = []
        for doc in session_docs:
            # When include_messages=False the aggregation returns message_count
            if include_messages:
                messages_data = doc.get("messages", [])
                message_count = len(
                    [m for m in messages_data if m.get("role") == "user"]
                )
            else:
                messages_data = []
                message_count = int(doc.get("message_count", 0))

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
        # If we didn't return enough sessions from the new schema, include old sessions
        if len(sessions) < limit:
            remaining = limit - len(sessions)
            old_sessions = await get_old_format_sessions(
                request, current_user, include_messages, limit=remaining, offset=offset
            )
            sessions.extend(old_sessions)
            sessions.sort(key=lambda s: s.updated_at, reverse=True)

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
    request: Request,
    current_user: UserInfo = Depends(get_current_user),
    include_messages: bool = Query(False, description="Include full message arrays"),
    limit: int = Query(50, description="Max number of sessions to return"),
    offset: int = Query(0, description="Number of sessions to skip"),
):
    """Expose the combined session list through the legacy path."""
    return await get_chat_sessions(
        request, current_user, include_messages, limit=limit, offset=offset
    )


@router.post("/chats")
async def create_chat_legacy(
    request: Request,
    title: Optional[str] = Query(None, description="Chat title"),
    model: Optional[str] = Query(None, description="Model ID"),
    current_user: UserInfo = Depends(get_current_user),
):
    """
    Legacy endpoint for creating chats.
    Creates a session and returns chat-compatible format.
    """
    # Use the v2/sessions endpoint logic
    session = await create_chat_session(request, model, title, current_user)

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
    request: Request,
    current_user: UserInfo = Depends(get_current_user),
    limit: int = Query(50, description="Max number of chats to return"),
    offset: int = Query(0, description="Number of chats to skip"),
):
    """
    Legacy endpoint for listing chats.
    Returns sessions in chat-compatible format.
    """
    # Get sessions using v2 endpoint
    sessions = await get_chat_sessions(request, current_user, False, limit=limit, offset=offset)

    # The session query has already applied pagination.
    chats = []
    for session in sessions:
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
    request: Request,
    current_user: UserInfo = Depends(get_current_user),
    limit: int = Query(50, description="Max number of chats to return"),
    offset: int = Query(0, description="Number of chats to skip"),
):
    """
    Legacy endpoint for listing chat history.
    Alias for /chats endpoint.
    """
    return await list_chats_legacy(request, current_user, limit, offset)


@router.delete("/chats/{chat_id:path}")
async def delete_chat_legacy(
    request: Request,
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

    sessions_collection = get_sessions_collection(request)
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
            db = request.app.state.bot.db

            if db is not None:
                conversations = db["conversations"]

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
                    session_mapping = await get_cached_session_mapping(current_user, request)
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
        logger.error("[Chats] Delete failed: %s", type(e).__name__, stack_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete chat") from e
