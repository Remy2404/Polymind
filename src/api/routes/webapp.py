import logging
import hmac
import hashlib
import json
import time
from datetime import datetime
from urllib.parse import parse_qsl, unquote
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Header, File, UploadFile
from pydantic import BaseModel, Field
from src.bot.telegram_bot import TelegramBot
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/webapp", tags=["webapp"])
_BOT_INSTANCE: Optional[TelegramBot] = None
def get_telegram_bot():
    """Get the global Telegram bot instance."""
    global _BOT_INSTANCE
    if _BOT_INSTANCE is None:
        raise RuntimeError("Bot instance not initialized")
    return _BOT_INSTANCE
def set_bot_instance(bot: TelegramBot):
    """Set the global bot instance."""
    global _BOT_INSTANCE
    _BOT_INSTANCE = bot
class WebAppUser(BaseModel):
    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None
    language_code: Optional[str] = None
    is_premium: Optional[bool] = None
class WebAppInitData(BaseModel):
    query_id: Optional[str] = None
    user: Optional[WebAppUser] = None
    receiver: Optional[WebAppUser] = None
    chat: Optional[Dict[str, Any]] = None
    chat_type: Optional[str] = None
    chat_instance: Optional[str] = None
    start_param: Optional[str] = None
    can_send_after: Optional[int] = None
    auth_date: int
    hash: str
class ChatMessage(BaseModel):
    content: str = Field(..., min_length=1, max_length=4000)
    model: Optional[str] = "deepseek-r1-0528"
    context: Optional[str] = None
class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    timestamp: float
    message_id: str
    model_used: Optional[str] = None
class ChatResponse(BaseModel):
    content: str
    timestamp: float
    message_id: str
    model_used: str
class ChatHistoryResponse(BaseModel):
    messages: list[ChatHistoryMessage]
    total_messages: int
    user_id: int
def validate_webapp_data(init_data: str, bot_token: str) -> WebAppInitData:
    """
    Validate Telegram Web App initialization data.
    Based on Telegram's documentation:
    https://core.telegram.org/bots/webapps
    """
    try:
        parsed_data = dict(parse_qsl(init_data))
        received_hash = parsed_data.pop("hash", "")
        if not received_hash:
            raise ValueError("Missing hash parameter")
        data_check_string = "\n".join(
            [f"{key}={value}" for key, value in sorted(parsed_data.items())]
        )
        secret_key = hmac.new(
            b"WebAppData", bot_token.encode(), hashlib.sha256
        ).digest()
        calculated_hash = hmac.new(
            secret_key, data_check_string.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(calculated_hash, received_hash):
            raise ValueError("Invalid hash - data may be tampered with")
        auth_date = int(parsed_data.get("auth_date", 0))
        current_time = int(time.time())
        if current_time - auth_date > 86400:
            raise ValueError("Data is too old")
        user_data = None
        if "user" in parsed_data:
            user_json = json.loads(unquote(parsed_data["user"]))
            user_data = WebAppUser(**user_json)
        chat_data = None
        if "chat" in parsed_data:
            chat_data = json.loads(unquote(parsed_data["chat"]))
        return WebAppInitData(
            query_id=parsed_data.get("query_id"),
            user=user_data,
            chat=chat_data,
            chat_type=parsed_data.get("chat_type"),
            chat_instance=parsed_data.get("chat_instance"),
            start_param=parsed_data.get("start_param"),
            can_send_after=parsed_data.get("can_send_after"),
            auth_date=auth_date,
            hash=received_hash,
        )
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"WebApp data validation failed: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid WebApp data: {str(e)}")
async def get_webapp_auth(
    authorization: str = Header(None), bot: TelegramBot = Depends(get_telegram_bot)
) -> WebAppInitData:
    """
    Extract and validate Web App authentication from Authorization header.
    Expected format: "tma <init_data>"
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("tma "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Expected 'tma <init_data>'",
        )
    init_data = authorization[4:]
    if "dev_hash_placeholder" in init_data:
        try:
            parsed_data = dict(item.split("=", 1) for item in init_data.split("&"))
            user_json = json.loads(parsed_data.get("user", "{}"))
            return WebAppInitData(
                user=WebAppUser(**user_json),
                auth_date=int(parsed_data.get("auth_date", time.time())),
                hash=parsed_data.get("hash", "dev_hash"),
            )
        except Exception as e:
            logger.warning(f"Development auth parsing failed: {e}")
            return WebAppInitData(
                user=WebAppUser(
                    id=123456789, first_name="Dev", last_name="User", username="devuser"
                ),
                auth_date=int(time.time()),
                hash="dev_hash",
            )
    return validate_webapp_data(init_data, bot.token)
@router.post("/auth/validate")
async def validate_auth(
    auth_data: WebAppInitData = Depends(get_webapp_auth),
    bot: TelegramBot = Depends(get_telegram_bot),
) -> Dict[str, Any]:
    """Validate Web App authentication and return user info with personalization data."""
    try:
        user_response = {
            "valid": True,
            "user_id": auth_data.user.id if auth_data.user else None,
            "user": auth_data.user.model_dump() if auth_data.user else None,
            "auth_date": auth_data.auth_date,
            "timestamp": time.time(),
        }
        if auth_data.user:
            user_id = auth_data.user.id
            logger.info(f"Validating auth and retrieving data for user {user_id}")
            message_handlers = bot.get_message_handlers()
            if message_handlers:
                if (
                    hasattr(message_handlers, "user_data_manager")
                    and message_handlers.user_data_manager
                ):
                    try:
                        await message_handlers.user_data_manager.initialize_user(
                            user_id
                        )
                        user_data = message_handlers.user_data_manager.get_user_data(
                            user_id
                        )
                        if user_data:
                            if hasattr(user_data, "__await__"):
                                user_data = await user_data
                            if isinstance(user_data, dict):
                                user_response["preferences"] = {
                                    "selected_model": user_data.get(
                                        "selected_model", "deepseek-r1-0528"
                                    ),
                                    "settings": user_data.get("settings", {}),
                                    "stats": user_data.get("stats", {}),
                                }
                                logger.info(
                                    f"Retrieved user preferences for user {user_id}"
                                )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load user preferences for user {user_id}: {e}"
                        )
                if (
                    hasattr(message_handlers, "text_handler")
                    and message_handlers.text_handler
                ):
                    if hasattr(message_handlers.text_handler, "conversation_manager"):
                        try:
                            conversation_manager = (
                                message_handlers.text_handler.conversation_manager
                            )
                            history = (
                                await conversation_manager.get_conversation_history(
                                    user_id=user_id, max_messages=1
                                )
                            )
                            user_response["conversation_stats"] = {
                                "has_history": len(history) > 0,
                                "last_activity": (
                                    history[0].get("timestamp") if history else None
                                ),
                            }
                            logger.info(
                                f"Retrieved conversation stats for user {user_id}: {len(history)} messages"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to get conversation stats for user {user_id}: {e}"
                            )
        return user_response
    except Exception as e:
        logger.error(f"Error in auth validation: {e}")
        raise HTTPException(status_code=500, detail="Authentication validation failed")
@router.post("/chat", response_model=ChatResponse)
async def chat_message(
    message: ChatMessage,
    auth_data: WebAppInitData = Depends(get_webapp_auth),
    bot: TelegramBot = Depends(get_telegram_bot),
) -> ChatResponse:
    """Process chat message using AI services with persistent memory context."""
    try:
        if not auth_data.user:
            raise HTTPException(status_code=401, detail="User data required")
        user_id = auth_data.user.id
        logger.info(
            f"Processing chat message from user {user_id} with model {message.model}"
        )
        message_handlers = bot.get_message_handlers()
        if not message_handlers:
            raise HTTPException(
                status_code=500, detail="Message handlers not available"
            )
        if (
            hasattr(message_handlers, "user_data_manager")
            and message_handlers.user_data_manager
        ):
            await message_handlers.user_data_manager.initialize_user(user_id)
        conversation_manager = None
        if hasattr(message_handlers, "text_handler") and message_handlers.text_handler:
            if hasattr(message_handlers.text_handler, "conversation_manager"):
                conversation_manager = (
                    message_handlers.text_handler.conversation_manager
                )
        conversation_context = []
        if conversation_manager:
            try:
                history = []
                if hasattr(conversation_manager, "model_history_manager"):
                    history = (
                        await conversation_manager.model_history_manager.get_history(
                            user_id=user_id,
                            max_messages=20,
                            model_id=message.model,
                        )
                    )
                    logger.info(
                        f"Retrieved {len(history)} history messages for model {message.model}"
                    )
                if (
                    not history
                    and hasattr(message_handlers, "user_data_manager")
                    and message_handlers.user_data_manager
                ):
                    user_context = (
                        await message_handlers.user_data_manager.get_user_context(
                            str(user_id)
                        )
                    )
                    for msg in user_context:
                        if (
                            isinstance(msg, dict)
                            and msg.get("role")
                            and msg.get("content")
                        ):
                            history.append(msg)
                    logger.info(
                        f"Fallback: Retrieved {len(history)} context messages from user data manager"
                    )
                conversation_context = []
                for msg in history:
                    if isinstance(msg, dict):
                        if msg.get("role") and msg.get("content"):
                            conversation_context.append(
                                {"role": msg["role"], "content": msg["content"]}
                            )
                        elif "user_message" in msg and msg["user_message"]:
                            conversation_context.append(
                                {"role": "user", "content": msg["user_message"]}
                            )
                        elif "assistant_message" in msg and msg["assistant_message"]:
                            conversation_context.append(
                                {
                                    "role": "assistant",
                                    "content": msg["assistant_message"],
                                }
                            )
                logger.info(
                    f"Retrieved {len(conversation_context)} context messages for user {user_id} with model {message.model}"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve conversation history: {e}")
                conversation_context = []
        if message_handlers.text_handler:
            ai_api = None
            model_used = message.model or "deepseek-r1-0528"
            if model_used.startswith("deepseek"):
                ai_api = getattr(message_handlers, "deepseek_api", None)
            elif (
                model_used.startswith("gpt")
                or model_used.startswith("claude")
                or "openrouter" in model_used
            ):
                ai_api = getattr(message_handlers, "openrouter_api", None)
            elif model_used.startswith("gemini"):
                ai_api = getattr(message_handlers, "gemini_api", None)
            else:
                ai_api = (
                    getattr(message_handlers, "deepseek_api", None)
                    or getattr(message_handlers, "openrouter_api", None)
                    or getattr(message_handlers, "gemini_api", None)
                )
            if not ai_api:
                raise HTTPException(status_code=500, detail="No AI service available")
            try:
                response_text = None
                if hasattr(ai_api, "generate_response"):
                    try:
                        messages = conversation_context + [
                            {"role": "user", "content": message.content}
                        ]
                        logger.info(
                            f"Sending {len(messages)} messages to AI API (including {len(conversation_context)} context messages)"
                        )
                        try:
                            response_text = await ai_api.generate_response(
                                messages=messages, model=model_used, max_tokens=2048
                            )
                        except TypeError:
                            response_text = await ai_api.generate_response(
                                prompt=message.content,
                                context=messages,
                                model=model_used,
                                max_tokens=2048,
                            )
                    except TypeError:
                        try:
                            messages = conversation_context + [
                                {"role": "user", "content": message.content}
                            ]
                            logger.info(
                                f"Fallback: Sending {len(messages)} messages to AI API"
                            )
                            try:
                                response_text = await ai_api.generate_response(
                                    messages=messages, max_tokens=2048
                                )
                            except TypeError:
                                response_text = await ai_api.generate_response(
                                    prompt=message.content,
                                    context=messages,
                                    max_tokens=2048,
                                )
                        except Exception:
                            context_prompt = ""
                            if conversation_context:
                                context_lines = []
                                for ctx_msg in conversation_context[-6:]:
                                    role = ctx_msg.get("role", "unknown")
                                    content = ctx_msg.get("content", "")
                                    if content:
                                        context_lines.append(
                                            f"{role.title()}: {content}"
                                        )
                                if context_lines:
                                    context_prompt = (
                                        "Previous conversation:\n"
                                        + "\n".join(context_lines)
                                        + "\n\nCurrent message:\n"
                                    )
                            response_text = await ai_api.generate_response(
                                prompt=context_prompt + message.content
                            )
                elif hasattr(ai_api, "generate_chat_response"):
                    messages = conversation_context + [
                        {"role": "user", "content": message.content}
                    ]
                    logger.info(
                        f"Sending {len(messages)} messages to DeepSeek API (including {len(conversation_context)} context messages)"
                    )
                    try:
                        response_text = await ai_api.generate_chat_response(
                            messages=messages, max_tokens=2048
                        )
                    except TypeError:
                        try:
                            response_text = await ai_api.generate_chat_response(
                                messages=messages
                            )
                        except TypeError:
                            context_prompt = ""
                            if conversation_context:
                                context_lines = []
                                for ctx_msg in conversation_context[-6:]:
                                    role = ctx_msg.get("role", "unknown")
                                    content = ctx_msg.get("content", "")
                                    if content:
                                        context_lines.append(
                                            f"{role.title()}: {content}"
                                        )
                                if context_lines:
                                    context_prompt = (
                                        "Previous conversation:\n"
                                        + "\n".join(context_lines)
                                        + "\n\nCurrent message:\n"
                                    )
                            single_message = [
                                {
                                    "role": "user",
                                    "content": context_prompt + message.content,
                                }
                            ]
                            response_text = await ai_api.generate_chat_response(
                                messages=single_message
                            )
                else:
                    raise HTTPException(
                        status_code=500, detail="Unsupported AI API interface"
                    )
                if not response_text:
                    raise HTTPException(
                        status_code=500, detail="Empty response from AI service"
                    )
                if conversation_manager and hasattr(
                    conversation_manager, "model_history_manager"
                ):
                    try:
                        await conversation_manager.model_history_manager.save_message_pair(
                            user_id=user_id,
                            user_message=message.content,
                            assistant_message=response_text,
                            model_id=model_used,
                        )
                        logger.info(
                            f"Saved conversation pair to model history for user {user_id} with model {model_used}"
                        )
                    except Exception as e:
                        logger.error(f"Failed to save to model history: {e}")
                if (
                    hasattr(message_handlers, "user_data_manager")
                    and message_handlers.user_data_manager
                ):
                    try:
                        await message_handlers.user_data_manager.save_message_pair(
                            user_id=str(user_id),
                            user_message=message.content,
                            assistant_message=response_text,
                            model_id=model_used,
                        )
                        logger.info(
                            f"Saved conversation pair to user data manager for user {user_id}"
                        )
                        await message_handlers.user_data_manager.update_stats(
                            user_id, message=True
                        )
                        logger.info(f"Updated user stats for user {user_id}")
                    except Exception as e:
                        logger.warning(f"Failed to save to user data manager: {e}")
                message_id = f"msg_{user_id}_{int(time.time() * 1000)}"
                return ChatResponse(
                    content=response_text,
                    timestamp=time.time(),
                    message_id=message_id,
                    model_used=model_used,
                )
            except Exception as e:
                logger.error(f"Error processing chat message: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to process message: {str(e)}"
                )
        raise HTTPException(status_code=500, detail="Text handler not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    limit: int = 50,
    model: Optional[str] = None,
    auth_data: WebAppInitData = Depends(get_webapp_auth),
    bot: TelegramBot = Depends(get_telegram_bot),
) -> ChatHistoryResponse:
    """Retrieve chat history for the authenticated user."""
    try:
        if not auth_data.user:
            raise HTTPException(status_code=401, detail="User data required")
        user_id = auth_data.user.id
        logger.info(f"Retrieving chat history for user {user_id}")
        message_handlers = bot.get_message_handlers()
        if not message_handlers:
            raise HTTPException(
                status_code=500, detail="Message handlers not available"
            )
        conversation_manager = None
        if hasattr(message_handlers, "text_handler") and message_handlers.text_handler:
            if hasattr(message_handlers.text_handler, "conversation_manager"):
                conversation_manager = (
                    message_handlers.text_handler.conversation_manager
                )
        if not conversation_manager:
            logger.warning("No conversation manager available for history retrieval")
            return ChatHistoryResponse(messages=[], total_messages=0, user_id=user_id)
        try:
            history = []
            if (
                hasattr(message_handlers, "user_data_manager")
                and message_handlers.user_data_manager
            ):
                try:
                    user_context = (
                        await message_handlers.user_data_manager.get_user_context(
                            str(user_id)
                        )
                    )
                    for msg in user_context:
                        if (
                            isinstance(msg, dict)
                            and msg.get("role")
                            and msg.get("content")
                        ):
                            history.append(msg)
                    logger.info(
                        f"Retrieved {len(history)} messages from user data manager"
                    )
                except Exception as e:
                    logger.warning(f"User data manager retrieval failed: {e}")
            if not history and hasattr(conversation_manager, "model_history_manager"):
                try:
                    fallback_history = (
                        await conversation_manager.model_history_manager.get_history(
                            user_id=user_id, max_messages=limit, model_id=model
                        )
                    )
                    logger.info(
                        f"Fallback: Retrieved {len(fallback_history)} messages from model history for model {model}"
                    )
                    history = fallback_history
                except Exception as e:
                    logger.warning(f"Model history retrieval failed: {e}")
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            history = []
        formatted_messages = []
        for i, msg in enumerate(history):
            if isinstance(msg, dict):
                msg_timestamp = msg.get("timestamp", time.time())
                model_used = msg.get(
                    "model_used", msg.get("model_id", model or "unknown")
                )
                if isinstance(msg_timestamp, str):
                    try:
                        msg_timestamp = float(msg_timestamp)
                    except ValueError:
                        msg_timestamp = time.time()
                elif isinstance(msg_timestamp, datetime):
                    msg_timestamp = msg_timestamp.timestamp()
                elif not isinstance(msg_timestamp, (int, float)):
                    msg_timestamp = time.time()
                if "role" in msg and "content" in msg:
                    formatted_messages.append(
                        ChatHistoryMessage(
                            role=msg["role"],
                            content=msg["content"],
                            timestamp=msg_timestamp,
                            message_id=msg.get(
                                "message_id", f"{msg['role']}_{user_id}_{i}"
                            ),
                            model_used=model_used,
                        )
                    )
                elif "user_message" in msg and msg["user_message"]:
                    formatted_messages.append(
                        ChatHistoryMessage(
                            role="user",
                            content=msg["user_message"],
                            timestamp=msg_timestamp,
                            message_id=msg.get("message_id", f"user_{user_id}_{i}"),
                            model_used=model_used,
                        )
                    )
                    if "assistant_message" in msg and msg["assistant_message"]:
                        formatted_messages.append(
                            ChatHistoryMessage(
                                role="assistant",
                                content=msg["assistant_message"],
                                timestamp=msg_timestamp + 1,
                                message_id=msg.get(
                                    "message_id", f"assistant_{user_id}_{i}"
                                ),
                                model_used=model_used,
                            )
                        )
        formatted_messages.sort(key=lambda x: x.timestamp)
        return ChatHistoryResponse(
            messages=formatted_messages,
            total_messages=len(formatted_messages),
            user_id=user_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat history endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
@router.delete("/chat/history")
async def clear_chat_history(
    model: Optional[str] = None,
    auth_data: WebAppInitData = Depends(get_webapp_auth),
    bot: TelegramBot = Depends(get_telegram_bot),
) -> Dict[str, Any]:
    """Clear chat history for the authenticated user."""
    try:
        if not auth_data.user:
            raise HTTPException(status_code=401, detail="User data required")
        user_id = auth_data.user.id
        logger.info(f"Clearing chat history for user {user_id}")
        message_handlers = bot.get_message_handlers()
        if not message_handlers:
            raise HTTPException(
                status_code=500, detail="Message handlers not available"
            )
        conversation_manager = None
        if hasattr(message_handlers, "text_handler") and message_handlers.text_handler:
            if hasattr(message_handlers.text_handler, "conversation_manager"):
                conversation_manager = (
                    message_handlers.text_handler.conversation_manager
                )
        if conversation_manager and hasattr(conversation_manager, "memory_manager"):
            try:
                await conversation_manager.memory_manager.clear_conversation_history(
                    user_id
                )
                logger.info(f"Cleared conversation history for user {user_id}")
                return {
                    "success": True,
                    "message": "Chat history cleared successfully",
                    "timestamp": time.time(),
                }
            except Exception as e:
                logger.error(f"Failed to clear conversation history: {e}")
                raise HTTPException(
                    status_code=500, detail="Failed to clear chat history"
                )
        return {
            "success": True,
            "message": "No conversation manager available",
            "timestamp": time.time(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in clear history endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
@router.get("/models")
async def get_available_models(
    auth_data: WebAppInitData = Depends(get_webapp_auth),
    bot: TelegramBot = Depends(get_telegram_bot),
) -> Dict[str, Any]:
    """Get available AI models for the user."""
    try:
        if not auth_data.user:
            raise HTTPException(status_code=401, detail="User data required")
        try:
            from src.services.model_handlers.model_configs import ModelConfigurations
            model_configs = ModelConfigurations()
            available_models = []
            for model_id, config in model_configs.get_all_models().items():
                available_models.append(
                    {
                        "id": model_id,
                        "name": config.display_name or model_id,
                        "provider": (
                            config.provider.value
                            if hasattr(config.provider, "value")
                            else str(config.provider)
                        ),
                        "supports_images": getattr(config, "supports_images", False),
                        "max_tokens": getattr(config, "max_tokens", 4096),
                        "capabilities": getattr(config, "capabilities", []),
                    }
                )
            return {
                "models": available_models,
                "default_model": "deepseek-r1-0528",
                "timestamp": time.time(),
            }
        except ImportError as e:
            logger.warning(f"Model configurations not available: {e}")
            return {
                "models": [
                    {
                        "id": "deepseek-r1-0528",
                        "name": "DeepSeek R1",
                        "provider": "deepseek",
                        "supports_images": False,
                        "max_tokens": 8192,
                        "capabilities": ["reasoning_capable"],
                    },
                    {
                        "id": "gemini-2.0-flash-exp",
                        "name": "Gemini 2.0 Flash",
                        "provider": "google",
                        "supports_images": True,
                        "max_tokens": 8192,
                        "capabilities": ["multimodal", "fast_response"],
                    },
                ],
                "default_model": "deepseek-r1-0528",
                "timestamp": time.time(),
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in models endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
@router.post("/models/select")
async def select_model(
    model_data: Dict[str, str],
    auth_data: WebAppInitData = Depends(get_webapp_auth),
    bot: TelegramBot = Depends(get_telegram_bot),
) -> Dict[str, Any]:
    """Select an AI model for the user."""
    try:
        if not auth_data.user:
            raise HTTPException(status_code=401, detail="User data required")
        user_id = auth_data.user.id
        model_id = model_data.get("model_id")
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        logger.info(f"Selecting model {model_id} for user {user_id}")
        message_handlers = bot.get_message_handlers()
        if not message_handlers:
            raise HTTPException(
                status_code=500, detail="Message handlers not available"
            )
        if (
            hasattr(message_handlers, "user_data_manager")
            and message_handlers.user_data_manager
        ):
            try:
                await message_handlers.user_data_manager.initialize_user(user_id)
                await message_handlers.user_data_manager.update_user_preference(
                    user_id, "selected_model", model_id
                )
                logger.info(
                    f"Updated model preference for user {user_id} to {model_id}"
                )
                return {
                    "success": True,
                    "selected_model": model_id,
                    "message": f"Model switched to {model_id}",
                    "timestamp": time.time(),
                }
            except Exception as e:
                logger.error(f"Failed to update model preference: {e}")
                raise HTTPException(
                    status_code=500, detail="Failed to update model preference"
                )
        return {
            "success": False,
            "message": "User data manager not available",
            "timestamp": time.time(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in select model endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
@router.post("/voice/transcribe")
async def transcribe_voice(
    audio: UploadFile = File(...),
    model: Optional[str] = None,
    process_with_ai: bool = True,
    auth_data: WebAppInitData = Depends(get_webapp_auth),
    bot: TelegramBot = Depends(get_telegram_bot),
) -> Dict[str, Any]:
    """Transcribe voice message and optionally process with AI."""
    try:
        if not auth_data.user:
            raise HTTPException(status_code=401, detail="User data required")
        user_id = auth_data.user.id
        logger.info(f"Transcribing voice for user {user_id}")
        message_handlers = bot.get_message_handlers()
        if not message_handlers:
            raise HTTPException(
                status_code=500, detail="Message handlers not available"
            )
        voice_handler = getattr(message_handlers, "voice_handler", None)
        if not voice_handler:
            raise HTTPException(status_code=500, detail="Voice handler not available")
        audio_data = await audio.read()
        try:
            transcription_result = await voice_handler.transcribe_audio(audio_data)
            transcribed_text = transcription_result.get("text", "")
            if not transcribed_text:
                raise HTTPException(
                    status_code=400, detail="No text could be transcribed"
                )
            response_data = {
                "text": transcribed_text,
                "confidence": transcription_result.get("confidence"),
                "language": transcription_result.get("language"),
                "timestamp": time.time(),
            }
            if process_with_ai and transcribed_text:
                try:
                    chat_message = ChatMessage(
                        content=transcribed_text, model=model or "deepseek-r1-0528"
                    )
                    ai_response = await chat_message(chat_message, auth_data, bot)
                    response_data.update(
                        {
                            "ai_response": ai_response.content,
                            "model_used": ai_response.model_used,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to process transcribed text with AI: {e}")
            return response_data
        except Exception as e:
            logger.error(f"Voice transcription failed: {e}")
            raise HTTPException(status_code=500, detail="Voice transcription failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in voice transcription endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for the web app API."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "telegram-webapp-api",
    }
@router.get("/debug/chat/history/{user_id}")
async def debug_get_chat_history(
    user_id: int, limit: int = 10, bot: TelegramBot = Depends(get_telegram_bot)
) -> Dict[str, Any]:
    """Debug endpoint to check conversation history directly by user ID."""
    try:
        logger.info(f"Debug: Retrieving chat history for user {user_id}")
        message_handlers = bot.get_message_handlers()
        if not message_handlers:
            return {"error": "Message handlers not available"}
        history = []
        if (
            hasattr(message_handlers, "user_data_manager")
            and message_handlers.user_data_manager
        ):
            try:
                user_context = (
                    await message_handlers.user_data_manager.get_user_context(
                        str(user_id)
                    )
                )
                history.extend(user_context)
                logger.info(
                    f"Debug: Retrieved {len(user_context)} messages from user data manager"
                )
            except Exception as e:
                logger.warning(f"Debug: User data manager failed: {e}")
        if hasattr(message_handlers, "text_handler") and message_handlers.text_handler:
            if hasattr(message_handlers.text_handler, "conversation_manager"):
                conversation_manager = (
                    message_handlers.text_handler.conversation_manager
                )
                if hasattr(conversation_manager, "model_history_manager"):
                    try:
                        model_history = await conversation_manager.model_history_manager.get_history(
                            user_id=user_id, max_messages=limit
                        )
                        logger.info(
                            f"Debug: Retrieved {len(model_history)} messages from model history"
                        )
                        if not history:
                            history = model_history
                    except Exception as e:
                        logger.warning(f"Debug: Model history retrieval failed: {e}")
        return {
            "user_id": user_id,
            "history_count": len(history),
            "history": history[:limit],
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return {"error": str(e), "timestamp": time.time()}
