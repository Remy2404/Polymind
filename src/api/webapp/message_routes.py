import logging
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import ORJSONResponse

from src.ai.models.factory import ModelHandlerFactory
from src.api.auth import get_current_user
from src.api.dependencies import get_services, get_sessions_collection
from src.api.schemas.webapp import ChatMessage, ChatResponse, Message, UserInfo
from src.api.webapp.legacy import get_cached_session_mapping

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webapp", tags=["Telegram Mini Apps - Unified Schema"])


@router.get(
    "/v2/sessions/{session_id}/messages",
    response_model=List[Message],
    response_class=ORJSONResponse,
)
async def get_session_messages(
    request: Request,
    session_id: str, current_user: UserInfo = Depends(get_current_user)
):
    """
    Get messages for a specific session (UUID-based).

    Much simpler than legacy approach - just query by UUID.
    """
    sessions_collection = get_sessions_collection(request)
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
    request: Request,
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
        session_mapping = await get_cached_session_mapping(current_user, request)
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
                    services = get_services(request)
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
            return await get_session_messages(request, chat_id, current_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Messages] Error in legacy endpoint: {e}", exc_info=True)
        return []


@router.post("/chat", response_model=ChatResponse)
async def send_chat_message(
    request: Request,
    message_data: ChatMessage, current_user: UserInfo = Depends(get_current_user)
):
    """
    Send message to AI model using unified UUID session schema.

    Creates new session if session_id not provided.
    Stores messages in session document for easy retrieval.
    """
    services = get_services(request)
    sessions_collection = get_sessions_collection(request)
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

        is_new_session = session_id is None
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

        if is_new_session:
            session_doc["messages"] = [user_msg, assistant_msg]
            session_doc["updated_at"] = now
            sessions_collection.insert_one(session_doc)
            logger.info(f"[Chat] Created new session {session_id}")
        else:
            sessions_collection.update_one(
                {"session_id": session_id, "user_id": user_id},
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
        logger.error("[Chat] Generation failed: %s", type(e).__name__, stack_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate response") from e
