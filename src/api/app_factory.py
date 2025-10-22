import os
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import src.api.routes.webhook as webhook_module
import src.api.routes.webapp as webapp_module
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
from src.services.mcp_bot_integration import initialize_mcp_for_bot
logger = logging.getLogger(__name__)
def get_telegram_bot_dependency(bot):
    """Creates a dependency that provides access to the TelegramBot instance."""
    def _get_bot():
        return bot
    return _get_bot
@asynccontextmanager
async def lifespan_context(app: FastAPI, bot: TelegramBot):
    """
    Lifespan context manager for the FastAPI application.
    Handles startup and shutdown of the TelegramBot with proper error handling.
    """
    logger.info("Starting application with enhanced monitoring...")
    app.state.start_time = time.time()
    shutdown_complete = False
    
    try:
        # Initialize MCP for webapp (like Telegram bot)
        logger.info("Initializing MCP integration for webapp...")
        await initialize_mcp_for_bot()
        
        await bot.application.initialize()
        await bot.application.start()
        if os.getenv("WEBHOOK_URL"):
            await bot.setup_webhook()
            raw_token = getattr(bot, "token", None)
            if raw_token:
                from urllib.parse import quote
                url_encoded_token = quote(raw_token, safe="")
                bot.logger.info(
                    f"Webhook endpoints registered at /webhook/{raw_token} and /webhook/{url_encoded_token}"
                )
            else:
                bot.logger.warning(
                    "WEBHOOK_URL is set but bot.token is not available; webhook endpoints may not be registered."
                )
        else:
            from threading import Thread
            def _polling():
                try:
                    bot.application.run_polling()
                except asyncio.CancelledError:
                    logger.info("Polling thread cancelled during shutdown")
                except Exception as e:
                    logger.error(f"Error in polling thread: {e}")
            polling_thread = Thread(target=_polling, daemon=True)
            polling_thread.start()
            bot.logger.info(
                "Polling fallback started; bot will process updates via polling."
            )
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Error during application startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down application...")
        shutdown_timeout = 10  # 10 seconds timeout for shutdown
        try:
            # Attempt graceful shutdown with timeout
            try:
                await asyncio.wait_for(bot.application.stop(), timeout=shutdown_timeout)
                logger.info("Application stop completed")
            except asyncio.TimeoutError:
                logger.warning(f"Application stop did not complete within {shutdown_timeout} seconds")
            except asyncio.CancelledError:
                logger.debug("Shutdown interrupted by cancellation (normal during container shutdown)")
            except Exception as e:
                logger.warning(f"Error stopping application: {e}")
            
            try:
                await asyncio.wait_for(bot.application.shutdown(), timeout=shutdown_timeout)
                logger.info("Application shutdown completed successfully")
            except asyncio.TimeoutError:
                logger.warning(f"Application shutdown did not complete within {shutdown_timeout} seconds")
            except asyncio.CancelledError:
                logger.debug("Shutdown interrupted by cancellation (normal during container shutdown)")
            except Exception as e:
                logger.warning(f"Error during shutdown: {e}")
            
            shutdown_complete = True
        except Exception as e:
            logger.error(f"Unexpected error during application shutdown: {e}", exc_info=True)
def create_application():
    os.environ["DEV_SERVER"] = "uvicorn"
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
        enable_logging=True
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    webhook_module._BOT_INSTANCE = bot
    webapp_module.set_bot_instance(bot)
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
        return {"message": "Polymind API is running", "version": "1.0.0", "status": "active"}
    
    app.state.bot = bot
    return app
