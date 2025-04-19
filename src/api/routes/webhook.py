"""
Webhook handling routes and exceptions for the Telegram Bot API.
Handles incoming Telegram webhook updates with validation and rate limiting.
"""

import json
import asyncio
import time
import aiohttp
import urllib.parse
from fastapi import APIRouter, Request, BackgroundTasks, Depends, HTTPException
from fastapi.responses import JSONResponse
import logging

# Store the bot instance globally in this module
_BOT_INSTANCE = None


# Custom exception for webhook errors
class WebhookException(Exception):
    """Custom exception for webhook-specific errors with HTTP status codes."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.detail)


# Create router - remove prefix to allow for properly formatted routes
router = APIRouter()
logger = logging.getLogger(__name__)

# Rate limiting storage - simple in-memory implementation
rate_limits = {}


# Helper function for webhook processing
def _get_update_type(update_data):
    """Determine the type of Telegram update for better logging."""
    if "message" in update_data:
        if "text" in update_data["message"]:
            return "text_message"
        elif "photo" in update_data["message"]:
            return "photo_message"
        elif "voice" in update_data["message"]:
            return "voice_message"
        elif "document" in update_data["message"]:
            return "document_message"
        return "other_message"
    elif "edited_message" in update_data:
        return "edited_message"
    elif "callback_query" in update_data:
        return "callback_query"
    elif "inline_query" in update_data:
        return "inline_query"
    return "unknown"


async def _process_update_with_retry(bot, update_data, logger):
    """Process update with retry mechanism for transient errors."""
    from src.utils.ignore_message import message_filter

    max_retries = 3
    base_delay = 0.5  # Start with 500ms delay

    # Check if update should be ignored
    bot_username = getattr(bot.application.bot, "username", "Gemini_AIAssistBot")
    if message_filter.should_ignore_update(update_data, bot_username):
        logger.info(f"Filtering out content in update {update_data.get('update_id')}")
        return

    for attempt in range(max_retries):
        try:
            # Process the update
            await bot.process_update(update_data)
            if attempt > 0:
                # Log successful retry
                logger.info(
                    f"Successfully processed update {update_data.get('update_id')} on attempt {attempt+1}"
                )
            return

        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            # Transient errors, retry with backoff
            retry_delay = base_delay * (2**attempt)  # Exponential backoff

            if attempt < max_retries - 1:
                logger.warning(
                    f"Transient error processing update {update_data.get('update_id')}, "
                    f"retrying in {retry_delay}s: {str(e)}"
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to process update {update_data.get('update_id')} "
                    f"after {max_retries} attempts: {str(e)}"
                )

        except Exception as e:
            # Non-transient errors, don't retry
            logger.error(
                f"Error processing update {update_data.get('update_id')}: {str(e)}",
                exc_info=True,
            )
            return


def get_telegram_bot():
    """Dependency to get the TelegramBot instance."""
    global _BOT_INSTANCE
    if _BOT_INSTANCE is None:
        # Fallback error if not properly configured
        raise NotImplementedError("Bot dependency injection not configured")
    return _BOT_INSTANCE


# Helper function to normalize token
def normalize_token(token: str) -> str:
    """Normalize token by decoding URL encoding if present."""
    try:
        # Try to decode in case it's URL encoded
        return urllib.parse.unquote(token)
    except Exception:
        return token


@router.post("/webhook/{token}")
async def webhook_handler(
    token: str,
    request: Request,
    background_tasks: BackgroundTasks,
    bot=Depends(get_telegram_bot),
):
    """
    Process incoming webhook updates from Telegram.

    Args:
        token: The bot token from the URL path
        request: The FastAPI request
        background_tasks: Background tasks runner
        bot: Telegram bot instance (injected)

    Returns:
        JSON response indicating success or error
    """
    # Get request ID from state
    request_id = getattr(request.state, "request_id", str(id(request)))
    logger_with_context = logging.LoggerAdapter(logger, {"request_id": request_id})

    # Track timing for monitoring
    start_time = time.time()

    try:
        # Log the actual path for debugging
        logger_with_context.info(f"Webhook received at path: {request.url.path}")

        # Normalize tokens for comparison
        normalized_token = normalize_token(token)
        bot_token = bot.token

        # Validate that token matches bot's token
        if normalized_token != bot_token:
            logger_with_context.warning(
                f"Invalid token: received '{normalized_token}', expected '{bot_token}'"
            )
            raise WebhookException(status_code=403, detail="Invalid token")

        # Validate request content type
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise WebhookException(
                status_code=415,
                detail="Unsupported Media Type: Content-Type must be application/json",
            )

        # Apply rate limiting
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        rate_key = f"rate_{client_ip}"

        # Get current rate data or initialize
        if rate_key in rate_limits:
            count, window_start = rate_limits[rate_key]
            # Reset counter if window expired (1 minute window)
            if current_time - window_start > 60:
                count = 0
                window_start = current_time
        else:
            count = 0
            window_start = current_time

        # Update rate limit counter
        count += 1
        rate_limits[rate_key] = (count, window_start)

        # Check rate limit (30 per minute)
        if count > 30:
            logger_with_context.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise WebhookException(status_code=429, detail="Too Many Requests")

        # Extract data with timeout handling
        try:
            # Set a reasonable timeout for JSON parsing
            update_data = await asyncio.wait_for(request.json(), timeout=2.0)
        except asyncio.TimeoutError:
            raise WebhookException(
                status_code=408,
                detail="Request Timeout: JSON parsing took too long",
            )
        except json.JSONDecodeError:
            raise WebhookException(
                status_code=400, detail="Bad Request: Invalid JSON format"
            )

        # Basic validation of Telegram update structure
        if not isinstance(update_data, dict):
            raise WebhookException(
                status_code=400, detail="Bad Request: Update data must be an object"
            )

        if "update_id" not in update_data:
            raise WebhookException(
                status_code=400, detail="Bad Request: Missing update_id field"
            )

        # Log incoming update with useful context
        update_id = update_data.get("update_id", "unknown")
        logger_with_context.info(
            f"Received webhook update {update_id} - "
            f"Type: {_get_update_type(update_data)} - "
            f"Size: {len(json.dumps(update_data))} bytes"
        )

        # Process update with retry mechanism in background
        background_tasks.add_task(
            _process_update_with_retry, bot, update_data, logger_with_context
        )

        # Track processing time
        process_time = time.time() - start_time

        # Return immediate response with useful headers
        return JSONResponse(
            content={"status": "ok", "received_at": time.time()},
            status_code=200,
            headers={
                "X-Process-Time": str(process_time),
                "X-Request-ID": request_id,
                "Connection": "keep-alive",
            },
        )

    except WebhookException as e:
        # Format webhook-specific exceptions
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail, "request_id": request_id},
        )

    except Exception as e:
        # Log unexpected errors
        logger_with_context.error(f"Webhook unexpected error: {str(e)}", exc_info=True)

        # Return a proper error response
        return JSONResponse(
            content={"status": "error", "detail": str(e), "request_id": request_id},
            status_code=500,
        )
