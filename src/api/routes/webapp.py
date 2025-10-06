"""
Telegram Mini Apps API Routes
Provides REST API endpoints for Telegram Mini Apps integration.
Auto-authenticates users via init data without separate login flow.
"""
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Telegram init data validation
try:
    from telegram_init_data import  parse, is_valid
    TELEGRAM_INIT_DATA_AVAILABLE = True
except ImportError:
    TELEGRAM_INIT_DATA_AVAILABLE = False
    logging.warning("telegram-init-data not installed. Run: pip install telegram-init-data[fastapi]")

from src.services.memory_context.conversation_manager import ConversationManager
from src.services.memory_context.memory_manager import MemoryManager
from src.services.memory_context.model_history_manager import ModelHistoryManager
from src.services.user_data_manager import UserDataManager
from src.services.user_preferences_manager import UserPreferencesManager
from src.services.model_handlers.factory import ModelHandlerFactory
from src.services.model_handlers.model_configs import ModelConfigurations
from src.handlers.text_processing.media_analyzer import MediaAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webapp", tags=["Telegram Mini Apps"])

# Global bot instance (set by app_factory)
_BOT_INSTANCE = None


def set_bot_instance(bot):
    """Set the TelegramBot instance for API access."""
    global _BOT_INSTANCE
    _BOT_INSTANCE = bot


# ============================================================================
# Pydantic Models
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


class ChatMessage(BaseModel):
    """Message sent to AI model."""
    message: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = Field(None, description="Model ID (e.g., 'gemini/gemini-2.0-flash-exp')")
    include_context: bool = Field(True, description="Include conversation history")
    max_context_messages: int = Field(10, ge=1, le=50)


class ChatResponse(BaseModel):
    """AI model response."""
    response: str
    model_used: str
    timestamp: float
    tokens_used: Optional[int] = None


class ConversationHistoryItem(BaseModel):
    """Single conversation history entry."""
    user_message: str
    ai_response: str
    model_used: str
    timestamp: float


class ModelInfo(BaseModel):
    """Available AI model information."""
    id: str
    name: str
    provider: str
    supports_vision: bool
    supports_tools: bool
    context_length: int
    is_free: bool


class UserPreferences(BaseModel):
    """User preferences settings."""
    preferred_model: Optional[str] = None
    theme: str = Field("light", pattern="^(light|dark)$")
    language: str = Field("en", min_length=2, max_length=5)
    notifications_enabled: bool = True


class MediaAnalysisResponse(BaseModel):
    """Media analysis result."""
    analysis: str
    model_used: str
    timestamp: float
    media_type: str


# ============================================================================
# Authentication Dependency
# ============================================================================

async def verify_telegram_init_data(
    authorization: str = Header(None, description="Authorization header with format: 'tma {init_data}'")
) -> Dict[str, Any]:
    """
    Verify and parse Telegram Mini Apps init data.
    
    Extracts init data from Authorization header, validates signature,
    and returns parsed user information.
    
    Args:
        authorization: Authorization header from Telegram Mini Apps
        
    Returns:
        Parsed init data dictionary with user info
        
    Raises:
        HTTPException: If auth header missing, invalid format, or signature invalid
    """
    if not TELEGRAM_INIT_DATA_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: telegram-init-data not installed"
        )
    
    # Check for Authorization header
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing. Include 'Authorization: tma {initData}'"
        )
    
    # Validate format: "tma {init_data}"
    if not authorization.startswith("tma "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Expected: 'tma {initData}'"
        )
    
    # Extract init data
    init_data = authorization[4:]  # Remove "tma " prefix
    
    # Get bot token from environment
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: TELEGRAM_BOT_TOKEN not set"
        )
    
    # Validate init data signature and expiration
    try:
        if not is_valid(init_data, bot_token):
            raise HTTPException(
                status_code=401,
                detail="Invalid init data signature or expired"
            )
        
        # Parse init data
        parsed_data = parse(init_data)
        
        # Ensure user data exists
        if not parsed_data.get("user"):
            raise HTTPException(
                status_code=400,
                detail="User data not found in init data"
            )
        
        return parsed_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Init data validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=401,
            detail=f"Init data validation failed: {str(e)}"
        )


async def get_current_user(init_data: Dict = Depends(verify_telegram_init_data)) -> UserInfo:
    """Extract current user info from validated init data."""
    user_data = init_data.get("user", {})
    return UserInfo(
        id=user_data["id"],
        first_name=user_data.get("first_name", "User"),
        last_name=user_data.get("last_name"),
        username=user_data.get("username"),
        language_code=user_data.get("language_code"),
        is_premium=user_data.get("is_premium", False),
        photo_url=user_data.get("photo_url")
    )


# ============================================================================
# Service Initialization Helpers
# ============================================================================

def get_services():
    """Get initialized service instances."""
    if not _BOT_INSTANCE:
        raise HTTPException(
            status_code=503,
            detail="Bot instance not initialized"
        )
    
    user_data_manager = UserDataManager()
    memory_manager = MemoryManager(
        db=user_data_manager.db if hasattr(user_data_manager, "db") else None
    )
    model_history_manager = ModelHistoryManager(memory_manager)
    conversation_manager = ConversationManager(memory_manager, model_history_manager)
    
    return {
        "user_data_manager": user_data_manager,
        "memory_manager": memory_manager,
        "conversation_manager": conversation_manager,
        "gemini_api": _BOT_INSTANCE.gemini_api if hasattr(_BOT_INSTANCE, "gemini_api") else None,
        "openrouter_api": _BOT_INSTANCE.openrouter_api if hasattr(_BOT_INSTANCE, "openrouter_api") else None,
        "deepseek_api": _BOT_INSTANCE.deepseek_api if hasattr(_BOT_INSTANCE, "deepseek_api") else None,
    }


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/user", response_model=UserInfo)
async def get_user_info(current_user: UserInfo = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    Returns user data extracted from Telegram Mini Apps init data.
    No separate login required - authentication is automatic via init data.
    """
    return current_user


@router.post("/chat", response_model=ChatResponse)
async def send_chat_message(
    message_data: ChatMessage,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Send message to AI model and get response.
    
    Supports:
    - Multiple AI providers (Gemini, OpenRouter, DeepSeek)
    - Conversation context and memory
    - Model selection or auto-routing
    
    Args:
        message_data: Message content and preferences
        current_user: Authenticated user (auto-injected)
        
    Returns:
        AI model response with metadata
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
        
        # Determine which model to use
        model_name = message_data.model
        if not model_name:
            # Use user's preferred model or default
            prefs_manager = UserPreferencesManager()
            user_prefs = prefs_manager.get_user_preferences(user_id)
            model_name = user_prefs.get("preferred_model", "gemini/gemini-2.0-flash-exp")
        
        # Get model handler
        model_handler = ModelHandlerFactory.get_model_handler(
            model_name=model_name,
            gemini_api=services["gemini_api"],
            openrouter_api=services["openrouter_api"],
            deepseek_api=services["deepseek_api"]
        )
        
        # Format prompt with context
        prompt = message_data.message
        if context_messages:
            context_str = "\n".join([
                f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
                for msg in context_messages
            ])
            prompt = f"Previous conversation:\n{context_str}\n\nCurrent message: {message_data.message}"
        
        # Generate response
        response = await model_handler.generate_response(prompt)
        
        # Save to conversation history
        await services["conversation_manager"].save_message_pair(
            user_id=user_id,
            user_message=message_data.message,
            ai_response=response,
            model_used=model_name
        )
        
        return ChatResponse(
            response=response,
            model_used=model_name,
            timestamp=datetime.now().timestamp(),
            tokens_used=None  # Can be added if provider returns token count
        )
        
    except Exception as e:
        logger.error(f"Chat error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.get("/history", response_model=List[ConversationHistoryItem])
async def get_conversation_history(
    current_user: UserInfo = Depends(get_current_user),
    limit: int = Query(20, ge=1, le=100, description="Number of messages to retrieve"),
    offset: int = Query(0, ge=0, description="Pagination offset")
):
    """
    Get user's conversation history.
    
    Returns paginated list of previous conversations with AI models.
    
    Args:
        limit: Maximum number of messages to return (default: 20, max: 100)
        offset: Number of messages to skip for pagination (default: 0)
        current_user: Authenticated user (auto-injected)
        
    Returns:
        List of conversation history items
    """
    services = get_services()
    user_id = current_user.id
    
    try:
        # Get conversation history
        messages = await services["conversation_manager"].get_conversation_history(
            user_id=user_id,
            max_messages=limit + offset
        )
        
        # Apply pagination
        paginated = messages[offset:offset + limit]
        
        # Format response
        history = []
        for msg in paginated:
            history.append(ConversationHistoryItem(
                user_message=msg.get("user", ""),
                ai_response=msg.get("assistant", ""),
                model_used=msg.get("model", "unknown"),
                timestamp=msg.get("timestamp", datetime.now().timestamp())
            ))
        
        return history
        
    except Exception as e:
        logger.error(f"History retrieval error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/models", response_model=List[ModelInfo])
async def get_available_models(
    current_user: UserInfo = Depends(get_current_user),
    vision_only: bool = Query(False, description="Filter to vision-capable models only"),
    free_only: bool = Query(False, description="Filter to free models only")
):
    """
    Get list of available AI models.
    
    Returns all models with their capabilities and features.
    Optionally filter by vision support or free tier availability.
    
    Args:
        vision_only: Only return models with vision capabilities
        free_only: Only return free tier models
        current_user: Authenticated user (auto-injected)
        
    Returns:
        List of available models with metadata
    """
    try:
        # Get all models from configuration
        all_models = ModelConfigurations.get_all_models()
        
        models = []
        for model_id, config in all_models.items():
            # Apply filters
            if vision_only and not config.supports_images:
                continue
            if free_only and not getattr(config, "is_free", False):
                continue
            
            models.append(ModelInfo(
                id=model_id,
                name=config.name,
                provider=config.provider.value,
                supports_vision=config.supports_images,
                supports_tools=config.supports_tools,
                context_length=config.context_length,
                is_free=getattr(config, "is_free", False)
            ))
        
        return models
        
    except Exception as e:
        logger.error(f"Model list error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve models: {str(e)}"
        )


@router.post("/media/analyze", response_model=MediaAnalysisResponse)
async def analyze_media(
    file: UploadFile = File(..., description="Image or document to analyze"),
    prompt: str = Query("Analyze this image", description="Analysis prompt"),
    model: Optional[str] = Query(None, description="Vision model to use"),
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Upload and analyze media with vision models.
    
    Supports:
    - Images (JPEG, PNG, WebP)
    - Documents (PDF with image analysis)
    
    Args:
        file: Media file to upload
        prompt: Analysis instruction
        model: Vision model to use (auto-selected if not specified)
        current_user: Authenticated user (auto-injected)
        
    Returns:
        Analysis result from vision model
    """
    services = get_services()
    user_id = current_user.id
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate file type
        valid_types = ["image/jpeg", "image/png", "image/webp", "application/pdf"]
        if file.content_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported: {', '.join(valid_types)}"
            )
        
        # Select vision model
        model_name = model or "gemini/gemini-2.0-flash-exp"
        
        # Get model handler
        model_handler = ModelHandlerFactory.get_model_handler(
            model_name=model_name,
            gemini_api=services["gemini_api"],
            openrouter_api=services["openrouter_api"],
            deepseek_api=services["deepseek_api"]
        )
        
        # Check if model supports vision
        model_config = ModelConfigurations.get_all_models().get(model_name)
        if not model_config or not model_config.supports_images:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} does not support vision analysis"
            )
        
        # Analyze media using MediaAnalyzer
        media_analyzer = MediaAnalyzer(
            services["gemini_api"],
            services["openrouter_api"]
        )
        
        # Create temporary file-like object
        import io
        media_file = io.BytesIO(content)
        media_file.name = file.filename
        
        # Perform analysis
        analysis = await media_analyzer.analyze_image(
            image_data=media_file,
            prompt=prompt,
            model_name=model_name
        )
        
        return MediaAnalysisResponse(
            analysis=analysis,
            model_used=model_name,
            timestamp=datetime.now().timestamp(),
            media_type=file.content_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Media analysis error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Media analysis failed: {str(e)}"
        )


@router.get("/preferences", response_model=UserPreferences)
async def get_user_preferences(
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Get user preferences and settings.
    
    Returns user's saved preferences including:
    - Preferred AI model
    - Theme (light/dark)
    - Language
    - Notification settings
    
    Args:
        current_user: Authenticated user (auto-injected)
        
    Returns:
        User preferences
    """
    user_id = current_user.id
    
    try:
        prefs_manager = UserPreferencesManager()
        prefs = prefs_manager.get_user_preferences(user_id)
        
        return UserPreferences(
            preferred_model=prefs.get("preferred_model"),
            theme=prefs.get("theme", "light"),
            language=prefs.get("language", current_user.language_code or "en"),
            notifications_enabled=prefs.get("notifications_enabled", True)
        )
        
    except Exception as e:
        logger.error(f"Preferences retrieval error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve preferences: {str(e)}"
        )


@router.post("/preferences", response_model=UserPreferences)
async def update_user_preferences(
    preferences: UserPreferences,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Update user preferences and settings.
    
    Saves user preferences including model selection, theme, and language.
    
    Args:
        preferences: Updated preferences
        current_user: Authenticated user (auto-injected)
        
    Returns:
        Updated preferences
    """
    user_id = current_user.id
    
    try:
        prefs_manager = UserPreferencesManager()
        
        # Validate model if specified
        if preferences.preferred_model:
            all_models = ModelConfigurations.get_all_models()
            if preferences.preferred_model not in all_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model: {preferences.preferred_model}"
                )
        
        # Update preferences
        prefs_manager.update_user_preferences(
            user_id=user_id,
            preferences={
                "preferred_model": preferences.preferred_model,
                "theme": preferences.theme,
                "language": preferences.language,
                "notifications_enabled": preferences.notifications_enabled
            }
        )
        
        return preferences
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preferences update error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update preferences: {str(e)}"
        )


@router.delete("/history")
async def clear_conversation_history(
    current_user: UserInfo = Depends(get_current_user),
    confirm: bool = Query(False, description="Confirmation flag to prevent accidental deletion")
):
    """
    Clear user's conversation history.
    
    Deletes all conversation history for the authenticated user.
    Requires confirmation to prevent accidental deletion.
    
    Args:
        confirm: Must be True to proceed with deletion
        current_user: Authenticated user (auto-injected)
        
    Returns:
        Success message
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required. Set 'confirm=true' to proceed."
        )
    
    services = get_services()
    user_id = current_user.id
    
    try:
        await services["conversation_manager"].clear_conversation_history(user_id)
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Conversation history cleared successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"History deletion error for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear history: {str(e)}"
        )


# ============================================================================
# Health Check for Web App
# ============================================================================

@router.get("/health")
async def webapp_health_check():
    """
    Health check endpoint for Telegram Mini Apps API.
    
    Returns service status and availability.
    """
    status = {
        "status": "ok",
        "service": "Telegram Mini Apps API",
        "telegram_init_data_available": TELEGRAM_INIT_DATA_AVAILABLE,
        "bot_initialized": _BOT_INSTANCE is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if not TELEGRAM_INIT_DATA_AVAILABLE:
        status["warning"] = "telegram-init-data library not installed"
    
    return JSONResponse(content=status)
