"""
Extended Telegram Mini Apps API Routes with Streaming Support
Adds Server-Sent Events (SSE) streaming for real-time AI responses
"""
import asyncio
import logging
from typing import Optional, AsyncIterator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.routes.webapp import (
    get_current_user,
    UserInfo,
    get_services
)
from src.services.model_handlers.factory import ModelHandlerFactory
from src.services.user_preferences_manager import UserPreferencesManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webapp", tags=["Telegram Mini Apps - Streaming"])


class StreamChatMessage(BaseModel):
    """Message for streaming chat endpoint."""
    message: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = Field(None, description="Model ID")
    include_context: bool = Field(True, description="Include conversation history")
    max_context_messages: int = Field(10, ge=1, le=50)
    chat_id: Optional[str] = Field(None, description="Chat session ID")


async def generate_ai_response_stream(
    user_id: int,
    message: str,
    model_name: str,
    context_messages: list,
    services: dict
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
            error_event = json.dumps({"type": "error", "error": "Invalid model specified"})
            yield f'data: {error_event}\n\n'
            return
        
        # Format prompt with context
        prompt = message
        if context_messages:
            context_str = "\n".join([
                f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
                for msg in context_messages
            ])
            prompt = f"Previous conversation:\n{context_str}\n\nCurrent message: {message}"
        
        # Get model handler
        logger.info(f"Creating model handler for model: '{model_name}' (type: {type(model_name)})")
        try:
            model_handler = ModelHandlerFactory.get_model_handler(
                model_name=model_name,
                gemini_api=services["gemini_api"],
                openrouter_api=services["openrouter_api"],
                deepseek_api=services["deepseek_api"]
            )
            logger.info(f"Successfully created handler: {type(model_handler).__name__}")
        except ValueError as e:
            logger.error(f"Failed to get model handler for {model_name}: {e}")
            error_event = json.dumps({"type": "error", "error": f"Model {model_name} is not available"})
            yield f'data: {error_event}\n\n'
            return
        
        # Send start event
        start_event = json.dumps({"type": "start", "model": model_name})
        yield f'data: {start_event}\n\n'
        
        # Check if model supports streaming
        has_streaming = hasattr(model_handler, 'generate_response_stream')
        logger.info(f"Model handler {type(model_handler).__name__} has streaming: {has_streaming}")
        
        if has_streaming:
            # Stream response chunks
            logger.info(f"Using streaming for model {model_name}")
            full_response = ""
            # CRITICAL: Must pass model parameter to streaming method too!
            async for chunk in model_handler.generate_response_stream(prompt, model=model_name):
                if chunk:  # Only send non-empty chunks
                    # Check if chunk is an error message
                    if isinstance(chunk, str) and chunk.startswith("Error:"):
                        logger.error(f"Model handler stream returned error: {chunk}")
                        error_event = json.dumps({"type": "error", "error": chunk})
                        yield f'data: {error_event}\n\n'
                        return
                    
                    full_response += chunk
                    # Properly escape and encode content chunk
                    content_event = json.dumps({"type": "content", "content": chunk})
                    yield f'data: {content_event}\n\n'
                    await asyncio.sleep(0)  # Allow other tasks to run
        else:
            # Fallback to non-streaming
            logger.info(f"Using non-streaming for model {model_name}")
            # CRITICAL: Must pass model parameter to the handler
            response = await model_handler.generate_response(prompt, model=model_name)
            
            # Check if response is None or empty
            if not response:
                logger.error(f"Model handler returned empty response for model {model_name}")
                error_event = json.dumps({"type": "error", "error": "Failed to generate response"})
                yield f'data: {error_event}\n\n'
                return
            
            # Check if response is an error message
            if isinstance(response, str) and response.startswith("Error:"):
                logger.error(f"Model handler returned error: {response}")
                error_event = json.dumps({"type": "error", "error": response})
                yield f'data: {error_event}\n\n'
                return
            
            full_response = response
            content_event = json.dumps({"type": "content", "content": response})
            yield f'data: {content_event}\n\n'
        
        # Save conversation
        await services["conversation_manager"].save_message_pair(
            user_id,
            message,
            full_response,
            model_name
        )
        
        # Send completion event
        done_event = json.dumps({"type": "done", "timestamp": datetime.now().timestamp()})
        yield f'data: {done_event}\n\n'
        
    except Exception as e:
        logger.error(f"Streaming error for user {user_id}: {e}", exc_info=True)
        # Send a simple error message
        error_event = json.dumps({"type": "error", "error": "An error occurred while processing your request"})
        yield f'data: {error_event}\n\n'


@router.post("/chat/stream")
async def stream_chat_message(
    message_data: StreamChatMessage,
    current_user: UserInfo = Depends(get_current_user)
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
    services = get_services()
    user_id = current_user.id
    
    try:
        # Get conversation context if requested
        context_messages = []
        if message_data.include_context:
            context_messages = await services["conversation_manager"].get_conversation_history(
                user_id=user_id,
                max_messages=message_data.max_context_messages
            )
        
        # Determine model
        model_name = message_data.model
        if not model_name:
            prefs_manager = UserPreferencesManager()
            user_prefs = prefs_manager.get_user_preferences(user_id)
            model_name = user_prefs.get("preferred_model", "gemini/gemini-2.0-flash-exp")
        
        # Validate model_name (basic check only - let ModelHandlerFactory handle validation)
        if not model_name or not isinstance(model_name, str) or model_name.strip() == "":
            model_name = "gemini/gemini-2.0-flash-exp"
            logger.warning(f"Invalid model_name for user {user_id}, using default: {model_name}")
        
        # Use model ID as-is - ModelHandlerFactory and ModelConfigurations handle all formats
        # No need for hardcoded normalization - backend supports all 54+ models
        logger.info(f"Using model: {model_name} for user {user_id}")
        
        # Return streaming response
        return StreamingResponse(
            generate_ai_response_stream(
                user_id=user_id,
                message=message_data.message,
                model_name=model_name,
                context_messages=context_messages,
                services=services
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        logger.error(f"Stream chat error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start stream: {str(e)}"
        )


class ChatSession(BaseModel):
    """Chat session information."""
    id: str
    title: Optional[str] = None
    model: str
    created_at: float
    updated_at: float
    message_count: int


@router.post("/chats")
async def create_chat_session(
    title: Optional[str] = None,
    model: Optional[str] = None,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Create a new chat session.
    
    Returns:
        Created chat session info
    """
    import uuid
    user_id = current_user.id
    
    try:
        # Generate chat ID
        chat_id = str(uuid.uuid4())
        
        # Get default model if not specified
        if not model:
            prefs_manager = UserPreferencesManager()
            user_prefs = prefs_manager.get_user_preferences(user_id)
            model = user_prefs.get("preferred_model", "gemini/gemini-2.0-flash-exp")
        
        # Store chat session (extend memory manager to support this)
        # For now, return the session info
        now = datetime.now().timestamp()
        
        return {
            "id": chat_id,
            "title": title or "New Chat",
            "model": model,
            "created_at": now,
            "updated_at": now,
            "message_count": 0
        }
        
    except Exception as e:
        logger.error(f"Chat creation error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create chat: {str(e)}"
        )


@router.get("/chats")
async def list_chat_sessions(
    current_user: UserInfo = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0
):
    """
    List user's chat sessions.
    
    Returns:
        List of chat sessions
    """
    # TODO: Implement chat session storage in database
    # For now, return empty list
    return []


@router.delete("/chats/{chat_id}")
async def delete_chat_session(
    chat_id: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Delete a chat session.
    
    Returns:
        Success message
    """
    # TODO: Implement chat deletion
    return {"status": "success", "message": "Chat deleted"}
