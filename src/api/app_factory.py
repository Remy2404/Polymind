import os
import time
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from src.api.routes import health, webhook, webapp

try:
    from src.api.routes import webapp_streaming

    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    logging.warning("webapp_streaming module not found - streaming endpoints disabled")
from src.api.middleware.request_tracking import RequestTrackingMiddleware
from src.api.middleware.rate_limiting import RateLimitMiddleware
from src.bot.telegram_bot import TelegramBot
from starlette.middleware.cors import CORSMiddleware
from src.integrations.mcp.bot_integration import initialize_mcp_for_bot, mcp_integration

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan_context(app: FastAPI, bot: TelegramBot):
    """Own the Telegram polling and application lifecycle on the ASGI event loop."""
    app.state.start_time = time.time()
    async with AsyncExitStack() as resources:
        resources.push_async_callback(mcp_integration.cleanup)
        resources.push_async_callback(bot.shutdown)
        await initialize_mcp_for_bot()
        await bot.application.initialize()
        resources.push_async_callback(bot.application.shutdown)
        await bot.application.start()
        resources.push_async_callback(bot.application.stop)
        if os.getenv("WEBHOOK_URL"):
            await bot.setup_webhook()
        else:
            await bot.application.updater.start_polling()
            resources.push_async_callback(bot.application.updater.stop)
        yield


def create_application(bot: TelegramBot | None = None) -> FastAPI:
    if bot is None:
        bot = TelegramBot()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with lifespan_context(app, bot):
            yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(RequestTrackingMiddleware)

    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        default_requests_per_minute=60,
        streaming_requests_per_minute=20,
        auth_requests_per_minute=10,
        enable_logging=True,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(webhook.WebhookException)
    async def webhook_exception_handler(
        request: Request, exc: webhook.WebhookException
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )

    app.include_router(health.router)
    app.include_router(webhook.router)
    app.include_router(webapp.router)

    # Include streaming routes if available
    if STREAMING_AVAILABLE:
        app.include_router(webapp_streaming.router)
        logger.info("Streaming endpoints enabled at /webapp/chat/stream")

    # Root route for API status
    @app.get("/")
    async def root():
        return {
            "message": "Polymind API is running",
            "version": "1.0.0",
            "status": "active",
        }

    app.state.bot = bot
    app.state.webhook_rate_limits = {}
    return app
