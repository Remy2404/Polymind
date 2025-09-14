import asyncio
import os
import time
import urllib.parse
import logging
from typing import Any, Type
from fastapi import APIRouter, Request, BackgroundTasks, Depends
try:
    from fastapi_msgspec.responses import MsgSpecJSONResponse
    from fastapi_msgspec.routing import MsgSpecRoute
    import msgspec.json as _msgspec_json
    from msgspec import Struct as MsgspecStruct
    msgspec_json: Any = _msgspec_json
    _MSGSPEC_AVAILABLE = True
except Exception:
    from fastapi.responses import JSONResponse as MsgSpecJSONResponse
    from fastapi.routing import APIRoute as MsgSpecRoute
    import json as _json
    msgspec_json: Any = _json
    MsgspecStruct = None
    _MSGSPEC_AVAILABLE = False
router = APIRouter(route_class=MsgSpecRoute, default_response_class=MsgSpecJSONResponse)
logger = logging.getLogger(__name__)
rate_limits: dict[str, tuple[int, float]] = {}
_BOT_INSTANCE = None
class WebhookException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
if MsgspecStruct is not None:
    UpdateMsg: Type[Any] = MsgspecStruct
else:
    class UpdateMsg:
        update_id: int
def normalize_token(token: str) -> str:
    try:
        return urllib.parse.unquote(token)
    except (ValueError, TypeError):
        return token
def get_telegram_bot():
    global _BOT_INSTANCE
    if _BOT_INSTANCE is None:
        raise RuntimeError("Bot instance not initialized")
    return _BOT_INSTANCE
def _get_update_type(u: dict) -> str:
    if "message" in u:
        m = u["message"]
        for t in ("text", "photo", "voice", "document"):
            if t in m:
                return f"{t}_message"
        return "other_message"
    for t in ("edited_message", "callback_query", "inline_query"):
        if t in u:
            return t
    return "unknown"
async def _process_update_with_retry(bot, update_data, log):
    from src.utils.ignore_message import message_filter
    max_retries, base_delay = 3, 0.5
    bot_name = getattr(bot.application.bot, "username", "UnknownBot")
    if message_filter.should_ignore_update(update_data, bot_name):
        log.info(f"Ignored update {update_data.get('update_id', 'unknown')}")
        return
    for attempt in range(max_retries):
        try:
            await bot.process_update(update_data)
            return
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(base_delay * (2**attempt))
            else:
                log.error(
                    f"Failed processing update {update_data.get('update_id', 'unknown')}: {e}"
                )
        except Exception as e:
            log.error(
                f"Unhandled error on {update_data.get('update_id', 'unknown')}: {e}",
                exc_info=True,
            )
            return
@router.post("/webhook/{token}")
async def webhook_handler(
    token: str,
    request: Request,
    background_tasks: BackgroundTasks,
    bot=Depends(get_telegram_bot),
):
    rid = getattr(request.state, "request_id", str(id(request)))
    log = logging.LoggerAdapter(logger, {"request_id": rid})
    start = time.time()
    try:
        if normalize_token(token) != bot.token:
            raise WebhookException(403, "Invalid token")
        if not request.headers.get("content-type", "").startswith("application/json"):
            raise WebhookException(415, "Content-Type must be application/json")
        client = getattr(request, "client", None)
        client_ip = getattr(client, "host", None) or "unknown"
        now = time.time()
        try:
            threshold = int(os.getenv("WEBHOOK_RATE_LIMIT", "500"))
        except ValueError:
            threshold = 500
        if threshold > 0:
            cnt, window = rate_limits.get(client_ip, (0, now))
            if now - window > 60:
                cnt, window = 0, now
            cnt += 1
            rate_limits[client_ip] = (cnt, window)
            if cnt > threshold:
                raise WebhookException(429, "Too many requests")
        raw = await asyncio.wait_for(request.body(), timeout=1.0)
        try:
            if _MSGSPEC_AVAILABLE and hasattr(msgspec_json, "decode"):
                body = msgspec_json.decode(raw, type=dict, strict=False)
            else:
                if hasattr(msgspec_json, "loads"):
                    body = msgspec_json.loads(raw.decode("utf-8"))
                else:
                    import json as _stdlib_json
                    body = _stdlib_json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise WebhookException(400, f"Invalid body: {e}")
        update_id = body.get("update_id", "unknown")
        log.info(f"Received update {update_id}, raw size = {len(raw)} bytes")
        background_tasks.add_task(_process_update_with_retry, bot, body, log)
        elapsed = time.time() - start
        return {
            "status": "ok",
            "received_at": time.time(),
            "X-Process-Time": f"{elapsed:.4f}",
            "X-Request-ID": rid,
        }
    except WebhookException as e:
        return MsgSpecJSONResponse(
            {"error": e.detail, "request_id": rid}, status_code=e.status_code
        )
    except Exception as e:
        log.error(f"Unexpected error: {e}", exc_info=True)
        return MsgSpecJSONResponse(
            {"status": "error", "detail": str(e), "request_id": rid}, status_code=500
        )
