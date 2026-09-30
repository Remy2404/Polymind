import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from src.ai.models.model_configs import ModelConfigurations
from src.api.auth import TELEGRAM_INIT_DATA_AVAILABLE, get_current_user
from src.api.schemas.webapp import ModelInfo, UserInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webapp", tags=["Telegram Mini Apps - Unified Schema"])


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
        logger.error("[Models] Listing failed: %s", type(e).__name__, stack_info=True)
        raise HTTPException(status_code=500, detail="Failed to list models") from e


@router.get("/health")
async def webapp_health_check(request: Request):
    """Health check for unified schema API."""
    return JSONResponse(
        content={
            "status": "ok",
            "service": "Telegram Mini Apps API - Unified Schema",
            "schema_version": "2.0",
            "telegram_init_data_available": TELEGRAM_INIT_DATA_AVAILABLE,
            "bot_initialized": getattr(request.app.state, "bot", None) is not None,
            "timestamp": datetime.now().isoformat(),
        }
    )
