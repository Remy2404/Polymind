"""
Extended Telegram Mini Apps API Routes with Streaming Support
Adds Server-Sent Events (SSE) streaming for real-time AI responses
"""

import logging
from typing import Optional, AsyncIterator, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.auth import get_current_user
from src.api.schemas.webapp import UserInfo
from src.api.dependencies import get_services, get_sessions_collection
from src.ai.models.factory import ModelHandlerFactory
from src.services.user_preferences_manager import UserPreferencesManager
from src.integrations.mcp.bot_integration import (
    generate_mcp_response,
    is_model_mcp_compatible,
)
from src.ai.models import ImageAttachment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webapp", tags=["Telegram Mini Apps - Streaming"])


class Attachment(BaseModel):
    """File attachment for messages."""

    name: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type")
    data: str = Field(..., description="Base64 encoded file content")


class StreamChatMessage(BaseModel):
    """Message for streaming chat endpoint."""

    message: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = Field(None, description="Model ID")
    include_context: bool = Field(True, description="Include conversation history")
    max_context_messages: int = Field(10, ge=1, le=50)
    chat_id: Optional[str] = Field(None, description="Chat session ID")
    attachments: Optional[List[Attachment]] = Field(
        None, description="File attachments (images, documents, etc.)"
    )


def convert_attachments_to_image_objects(
    attachments: Optional[List[Attachment]],
) -> Optional[List[ImageAttachment]]:
    """Convert API attachments to ImageAttachment objects for model handlers."""
    if not attachments:
        return None

    image_attachments = []
    for attachment in attachments:
        # Only process image attachments for now
        if attachment.content_type.startswith("image/"):
            image_attachments.append(
                ImageAttachment(
                    name=attachment.name,
                    content_type=attachment.content_type,
                    data=attachment.data,  # base64 string
                )
            )

    return image_attachments if image_attachments else None


async def generate_ai_response_stream(
    user_id: int,
    message: str,
    model_name: str,
    context_messages: list,
    services: dict,
    attachments: Optional[List[Attachment]] = None,
    chat_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Generate AI response as Server-Sent Events stream.

    Yields:
        SSE formatted chunks: "data: {json}\n\n"
    """
    import json

    try:
        # Validate model_name before proceeding
        if not model_name or not isinstance(model_name, str) or not model_name.strip():
            logger.error(f"Invalid model_name parameter: {model_name}")
            error_event = json.dumps(
                {"type": "error", "error": "Invalid model specified"}
            )
            yield f"data: {error_event}\n\n"
            return

        # Format prompt with context (optimized)
        prompt = message
        if context_messages and len(context_messages) > 0:
            # Limit context formatting for speed - only use last 3 messages
            recent_messages = (
                context_messages[-3:] if len(context_messages) > 3 else context_messages
            )
            context_parts = []
            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:  # Only add if both exist
                    context_parts.append(
                        f"{role.title()}: {content[:200]}"
                    )  # Truncate for speed

            if context_parts:
                context_str = "\n".join(context_parts)
                prompt = f"Context:\n{context_str}\n\nNew: {message}"

        # Send start event early for better perceived performance
        start_event = json.dumps({"type": "start", "model": model_name})
        yield f"data: {start_event}\n\n"

        # Try MCP-enhanced response first (like Telegram bot) - but with timeout
        logger.info(f"Checking MCP compatibility for model: {model_name}")
        if is_model_mcp_compatible(model_name) and not attachments:
            logger.info(
                f"Model {model_name} is MCP compatible and no attachments present, attempting MCP response"
            )
            try:
                # Add timeout to MCP response to prevent slowdowns
                import asyncio

                mcp_response = await asyncio.wait_for(
                    generate_mcp_response(
                        prompt=prompt, user_id=user_id, model=model_name
                    ),
                    timeout=30.0,  # 30 second timeout for MCP
                )
                if mcp_response:
                    logger.info(
                        f"MCP response generated successfully for user {user_id}"
                    )

                    # Send MCP response as content
                    content_event = json.dumps(
                        {"type": "content", "content": mcp_response}
                    )
                    yield f"data: {content_event}\n\n"

                    # Save conversation
                    await services["conversation_manager"].save_message_pair(
                        user_id, message, mcp_response, model_name
                    )

                    # Send completion event
                    done_event = json.dumps(
                        {"type": "done", "timestamp": datetime.now().timestamp()}
                    )
                    yield f"data: {done_event}\n\n"
                    return
                else:
                    logger.info(
                        "MCP response failed, falling back to regular model handler"
                    )
            except Exception as e:
                logger.warning(
                    f"MCP response failed for user {user_id}: {e}, falling back to regular handler"
                )

        # Fallback to regular model handler (existing logic)
        logger.info(f"Using regular model handler for model: {model_name}")

        # Get model handler
        logger.info(
            f"Creating model handler for model: '{model_name}' (type: {type(model_name)})"
        )
        try:
            model_handler = ModelHandlerFactory.get_model_handler(
                model_name=model_name,
                gemini_api=services["gemini_api"],
                openrouter_api=services["openrouter_api"],
                deepseek_api=services["deepseek_api"],
            )
            logger.info(f"Successfully created handler: {type(model_handler).__name__}")
        except ValueError as e:
            logger.error(f"Failed to get model handler for {model_name}: {e}")
            error_event = json.dumps(
                {"type": "error", "error": f"Model {model_name} is not available"}
            )
            yield f"data: {error_event}\n\n"
            return

        # Convert attachments to ImageAttachment objects
        image_attachments = convert_attachments_to_image_objects(attachments)

        # Check if model supports attachments (only Gemini models support vision)
        from src.ai.models.model_configs import (
            ModelConfigurations,
            Provider,
        )

        model_config = ModelConfigurations.get_all_models().get(model_name)
        supports_attachments = model_config and model_config.provider == Provider.GEMINI

        # Check if model supports streaming
        has_streaming = hasattr(model_handler, "generate_response_stream")
        logger.info(
            f"Model handler {type(model_handler).__name__} has streaming: {has_streaming}"
        )

        if has_streaming:
            # Stream response chunks with minimal delay
            logger.info(f"Using streaming for model {model_name}")
            full_response = ""
            # CRITICAL: Must pass model parameter to streaming method too!
            # Only pass attachments if the model supports them
            stream_kwargs = {"prompt": prompt, "model": model_name}
            if supports_attachments and image_attachments:
                stream_kwargs["attachments"] = image_attachments

            async for chunk in model_handler.generate_response_stream(**stream_kwargs):
                if chunk:  # Only send non-empty chunks
                    # Check if chunk is an error message
                    if isinstance(chunk, str) and chunk.startswith("Error:"):
                        logger.error(f"Model handler stream returned error: {chunk}")
                        error_event = json.dumps({"type": "error", "error": chunk})
                        yield f"data: {error_event}\n\n"
                        return

                    full_response += chunk
                    # Properly escape and encode content chunk
                    content_event = json.dumps({"type": "content", "content": chunk})
                    yield f"data: {content_event}\n\n"
                    # Remove sleep to maximize streaming speed
                    # await asyncio.sleep(0)  # Commented out for faster streaming
        else:
            # Fallback to non-streaming
            logger.info(f"Using non-streaming for model {model_name}")
            # CRITICAL: Must pass model parameter to the handler
            # Only pass attachments if the model supports them
            response_kwargs = {"prompt": prompt, "model": model_name}
            if supports_attachments and image_attachments:
                response_kwargs["attachments"] = image_attachments

            response = await model_handler.generate_response(**response_kwargs)

            # Check if response is None or empty
            if not response:
                logger.error(
                    f"Model handler returned empty response for model {model_name}"
                )

                # If the primary model is Gemini and it failed, try to fallback to DeepSeek or OpenRouter
                if model_name in [
                    "gemini",
                    "gemini-3.8-flash",
                    "gemini/gemini-2.0-flash-exp",
                ]:
                    logger.info("Gemini failed, attempting fallback to DeepSeek...")
                    try:
                        fallback_handler = ModelHandlerFactory.get_model_handler(
                            model_name="deepseek",
                            gemini_api=services["gemini_api"],
                            openrouter_api=services["openrouter_api"],
                            deepseek_api=services["deepseek_api"],
                        )
                        # DeepSeek doesn't support attachments, so don't pass them
                        fallback_response = await fallback_handler.generate_response(
                            prompt, model="deepseek"
                        )
                        if fallback_response and not fallback_response.startswith(
                            "Error:"
                        ):
                            logger.info("Successfully used DeepSeek as fallback")
                            content_event = json.dumps(
                                {
                                    "type": "content",
                                    "content": f"*[Switched to DeepSeek due to Gemini overload]*\n\n{fallback_response}",
                                }
                            )
                            yield f"data: {content_event}\n\n"
                            full_response = fallback_response
                        else:
                            raise Exception("Fallback also failed")
                    except Exception as fallback_error:
                        logger.warning(f"Fallback to DeepSeek failed: {fallback_error}")
                        error_event = json.dumps(
                            {
                                "type": "error",
                                "error": "Gemini API is currently overloaded and fallback models are also unavailable. Please try again in a few minutes.",
                            }
                        )
                        yield f"data: {error_event}\n\n"
                        return
                else:
                    error_event = json.dumps(
                        {"type": "error", "error": "Failed to generate response"}
                    )
                    yield f"data: {error_event}\n\n"
                    return

            # Check if response is an error message
            if isinstance(response, str) and response.startswith("Error:"):
                logger.error(f"Model handler returned error: {response}")
                error_event = json.dumps({"type": "error", "error": response})
                yield f"data: {error_event}\n\n"
                return

            full_response = response
            content_event = json.dumps({"type": "content", "content": response})
            yield f"data: {content_event}\n\n"

        # Save conversation
        await services["conversation_manager"].save_message_pair(
            user_id, message, full_response, model_name
        )

        # Save to chat session if chat_id provided
        if chat_id:
            try:
                from src.database.connection import get_database

                db, client = get_database()
                if db is not None:
                    chat_sessions = db.chat_sessions
                    now = datetime.now()

                    # Create message objects
                    user_msg = {
                        "id": f"{chat_id}_{int(now.timestamp() * 1000)}",
                        "role": "user",
                        "content": message,
                        "createdAt": now.isoformat(),
                        "model": model_name,
                        "attachments": (
                            [att.dict() for att in attachments] if attachments else None
                        ),
                    }
                    assistant_msg = {
                        "id": f"{chat_id}_{int(now.timestamp() * 1000) + 1}",
                        "role": "assistant",
                        "content": full_response,
                        "createdAt": now.isoformat(),
                        "model": model_name,
                    }

                    # Update chat session with new messages
                    update_result = chat_sessions.update_one(
                        {
                            "session_id": chat_id,
                            "user_id": user_id,  # Ensure user owns the chat
                        },
                        {
                            "$push": {"messages": {"$each": [user_msg, assistant_msg]}},
                            "$set": {"updated_at": now},
                            "$inc": {"message_count": 2},
                        },
                    )

                    if update_result.matched_count > 0:
                        logger.info(f"Saved message pair to chat session {chat_id}")
                    else:
                        logger.warning(
                            f"Chat session {chat_id} not found for user {user_id}"
                        )
            except Exception as e:
                logger.error(
                    f"Failed to save message to chat session: {e}", exc_info=True
                )
                # Don't fail the request if chat session save fails

        # Send completion event
        done_event = json.dumps(
            {"type": "done", "timestamp": datetime.now().timestamp()}
        )
        yield f"data: {done_event}\n\n"

    except Exception as e:
        logger.error(f"Streaming error for user {user_id}: {e}", exc_info=True)
        # Send a simple error message
        error_event = json.dumps(
            {
                "type": "error",
                "error": "An error occurred while processing your request",
            }
        )
        yield f"data: {error_event}\n\n"


@router.post("/chat/stream")
async def stream_chat_message(
    request: Request,
    message_data: StreamChatMessage, current_user: UserInfo = Depends(get_current_user)
):
    """
    Send message to AI model and stream response using Server-Sent Events.

    Returns:
        SSE stream with events:
        - start: {"type": "start", "model": "model_id"}
        - content: {"type": "content", "content": "chunk"}
        - done: {"type": "done", "timestamp": 123456}
        - error: {"type": "error", "error": "message"}
    """
    services = get_services(request)
    user_id = current_user.id

    try:
        # Determine model FIRST before retrieving context
        model_name = message_data.model
        if not model_name:
            prefs_manager = UserPreferencesManager()
            user_prefs = prefs_manager.get_user_preferences(user_id)
            model_name = user_prefs.get(
                "preferred_model", "gemini/gemini-2.0-flash-exp"
            )

        # Validate model_name (basic check only - let ModelHandlerFactory handle validation)
        if (
            not model_name
            or not isinstance(model_name, str)
            or model_name.strip() == ""
        ):
            model_name = "gemini/gemini-2.0-flash-exp"
            logger.warning(
                f"Invalid model_name for user {user_id}, using default: {model_name}"
            )

        # Use model ID as-is - ModelHandlerFactory and ModelConfigurations handle all formats
        # No need for hardcoded normalization - backend supports all 54+ models
        logger.info(f"Using model: {model_name} for user {user_id}")

        # Get conversation context for THIS SPECIFIC MODEL if requested (optimized)
        context_messages = []
        if message_data.include_context:
            # Use a smaller context window for faster processing
            max_context = min(
                message_data.max_context_messages, 5
            )  # Limit to 5 messages max for speed
            context_messages = await services[
                "conversation_manager"
            ].get_conversation_history(
                user_id=user_id,
                model=model_name,  # CRITICAL: Pass model to get model-specific history
                max_messages=max_context,
            )

        # Return streaming response with optimized headers for speed
        return StreamingResponse(
            generate_ai_response_stream(
                user_id=user_id,
                message=message_data.message,
                model_name=model_name,
                context_messages=context_messages,
                services=services,
                attachments=message_data.attachments,
                chat_id=message_data.chat_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Transfer-Encoding": "chunked",  # Enable chunked encoding
                "Access-Control-Allow-Origin": "*",  # Allow CORS for speed
            },
        )

    except Exception as e:
        logger.error("Stream chat failed for user %s: %s", user_id, type(e).__name__, stack_info=True)
        raise HTTPException(status_code=500, detail="Failed to start stream") from e



@router.get("/chats/{chat_id}/messages")
async def get_chat_messages(
    request: Request,
    chat_id: str,
    current_user: UserInfo = Depends(get_current_user),
    limit: int = 100,
    offset: int = 0,
):
    """
    Get messages from a specific chat session.

    Returns:
        List of messages from the chat session
    """
    user_id = str(current_user.id)

    try:
        chat_sessions = get_sessions_collection(request)

        # Get chat session and verify ownership
        session_doc = chat_sessions.find_one(
            {
                "session_id": chat_id,
                "user_id": user_id,  # Ensure user can only access their own chats
            }
        )

        if not session_doc:
            logger.warning(f"Chat session {chat_id} not found for user {user_id}")
            raise HTTPException(
                status_code=404, detail="Chat session not found or access denied"
            )

        # Get messages with pagination
        messages = session_doc.get("messages", [])

        # Apply pagination
        total_messages = len(messages)
        start_idx = max(0, total_messages - offset - limit)
        end_idx = total_messages - offset

        if start_idx >= total_messages:
            paginated_messages = []
        else:
            paginated_messages = messages[start_idx:end_idx]

        logger.info(
            f"Retrieved {len(paginated_messages)} messages from chat session {chat_id}"
        )

        return {
            "messages": paginated_messages,
            "total": total_messages,
            "has_more": start_idx > 0,
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(
            f"Error getting messages from chat session {chat_id} for user {user_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to get chat messages") from e
