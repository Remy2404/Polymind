import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import src.api.routes.webhook as webhook_module
import src.api.routes.webapp as webapp_module
from src.api.routes import health, webhook, webapp
from src.api.middleware.request_tracking import RequestTrackingMiddleware
from src.bot.telegram_bot import TelegramBot
from starlette.middleware.cors import CORSMiddleware

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
    Handles startup and shutdown of the TelegramBot.
    """
    # Startup logic - runs before application starts taking requests
    logger.info("Starting application with enhanced monitoring...")
    app.state.start_time = time.time()  # Track app start time

    try:
        await bot.application.initialize()
        await bot.application.start()

        # If WEBHOOK_URL is set, configure webhook; otherwise, start polling fallback
        if os.getenv("WEBHOOK_URL"):
            await bot.setup_webhook()
            raw_token = bot.token
            url_encoded_token = raw_token.replace(":", "%3A")
            bot.logger.info(
                f"Webhook endpoints registered at /webhook/{raw_token} and /webhook/{url_encoded_token}"
            )
        else:
            # Start polling in a background thread for development/local usage
            from threading import Thread

            def _polling():
                bot.application.run_polling()

            Thread(target=_polling, daemon=True).start()
            bot.logger.info(
                "Polling fallback started; bot will process updates via polling."
            )

        # Log successful startup
        logger.info("Application started successfully")

        yield  # Application runs and handles requests here

    except Exception as e:
        logger.error(f"Error during application startup: {e}", exc_info=True)
        raise
    finally:
        # Shutdown logic - runs after application finishes handling requests
        logger.info("Shutting down application...")

        try:
            await bot.application.stop()
            await bot.application.shutdown()
            logger.info("Application shutdown completed successfully")
        except Exception as e:
            logger.error(f"Error during application shutdown: {e}", exc_info=True)


def create_application():
    os.environ["DEV_SERVER"] = "uvicorn"

    bot = TelegramBot()

    # Create lifespan manager that includes the bot instance
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with lifespan_context(app, bot):
            yield

    # Create FastAPI app
    app = FastAPI(lifespan=lifespan)

    # Add middleware for compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add request tracking middleware
    app.add_middleware(RequestTrackingMiddleware)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure dependency injection for bot instance
    # Set the actual bot instance in the webhook module's global variable
    webhook_module._BOT_INSTANCE = bot

    # Set bot instance for webapp module
    webapp_module.set_bot_instance(bot)

    # Set up exception handlers for webhook exceptions
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

    # Include routers
    app.include_router(health.router)
    app.include_router(webhook.router)
    app.include_router(webapp.router)

    # Store bot instance in app state
    app.state.bot = bot

    # Return the fully configured app
    return app
