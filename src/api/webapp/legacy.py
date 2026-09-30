import hashlib
import logging
from datetime import datetime
from functools import lru_cache
from typing import List

from cachetools import TTLCache
from fastapi import Request

from src.api.dependencies import get_services
from src.api.schemas.webapp import ChatSession, Message, UserInfo

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1000)
def _cached_md5(text: str) -> str:
    """Cached MD5 hash function for performance."""
    return hashlib.md5(text.encode()).hexdigest()


async def get_cached_session_mapping(current_user: UserInfo, request: Request) -> dict:
    """Cache legacy IDs per user within the current application."""
    cache = getattr(request.app.state, "legacy_session_mappings", None)
    if cache is None:
        cache = TTLCache(maxsize=1000, ttl=300)
        request.app.state.legacy_session_mappings = cache
    if current_user.id not in cache:
        cache[current_user.id] = await _build_session_mapping(current_user, request)
    return cache[current_user.id]


async def _build_session_mapping(current_user: UserInfo, request: Request) -> dict:
    """Build session mapping from database."""
    services = get_services(request)
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


async def get_old_format_sessions(
    request: Request,
    current_user: UserInfo,
    include_messages: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> List[ChatSession]:
    """
    Get chat sessions from the old cache_key format memory system.
    This bridges the gap until all sessions are migrated to the new format.
    """
    services = get_services(request)
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

        # Use projection to minimize transferred data and support pagination.
        # When include_messages is False we can request only the metadata and compute
        # message_count server-side using aggregation, avoiding loading large message arrays.
        if include_messages:
            session_docs = list(
                conversations_collection.find(
                    {"cache_key": {"$regex": f"^{user_prefix}"}},
                    {"cache_key": 1, "messages": 1, "last_updated": 1, "title": 1},
                )
                .skip(offset)
                .limit(limit)
            )
        else:
            # Use aggregation to only fetch cache_key, last_updated, and the computed message_count
            pipeline = [
                {"$match": {"cache_key": {"$regex": f"^{user_prefix}"}}},
                {
                    "$project": {
                        "cache_key": 1,
                        "last_updated": 1,
                        "title": 1,
                        "message_count": {"$size": {"$ifNull": ["$messages", []]}},
                    }
                },
                {"$skip": offset},
                {"$limit": limit},
            ]

            session_docs = list(conversations_collection.aggregate(pipeline))

        logger.info(
            f"[Old Sessions] Found {len(session_docs)} old format conversations for user {user_id}"
        )

        sessions = []
        for doc in session_docs:
            cache_key = doc.get("cache_key", "")
            messages = doc.get("messages", [])
            # If aggregation returned message_count use it, otherwise fallback to list length
            message_count = doc.get("message_count", len(messages))
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
