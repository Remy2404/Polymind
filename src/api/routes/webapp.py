"""Public Telegram Mini App routes."""

from fastapi import APIRouter

from src.api.webapp import message_routes, session_routes, status_routes, user_routes

router = APIRouter()
router.include_router(user_routes.router)
router.include_router(session_routes.router)
router.include_router(message_routes.router)
router.include_router(status_routes.router)
