import json
import logging
import os
from datetime import datetime
from typing import Any, Dict
from urllib.parse import parse_qs, unquote_plus

from fastapi import Depends, Header, HTTPException, Query

from src.api.schemas.webapp import UserInfo

try:
    from telegram_init_data import parse, is_valid

    TELEGRAM_INIT_DATA_AVAILABLE = True
except ImportError:
    TELEGRAM_INIT_DATA_AVAILABLE = False

logger = logging.getLogger(__name__)


async def verify_telegram_init_data(
    authorization: str = Header(
        None, description="Authorization header with format: 'tma {init_data}'"
    ),
    user_id: str = Query(
        None,
        description="User ID for fallback authentication when outside Telegram context",
    ),
) -> Dict[str, Any]:
    """Verify and parse Telegram Mini Apps init data.

    Supports two authentication methods:
    1. Primary: Telegram initData (from mini app)
    2. Fallback: user_id URL parameter (when opened outside Telegram, e.g., "Open" button)
    """

    if not TELEGRAM_INIT_DATA_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: telegram-init-data not installed",
        )

    # Try primary authentication: Telegram initData
    if authorization:
        if not authorization.startswith("tma "):
            logger.warning("[Auth] Invalid authorization format")
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization format. Expected: 'tma {initData}'",
            )

        init_data = authorization[4:]
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            raise HTTPException(
                status_code=500,
                detail="Server configuration error: TELEGRAM_BOT_TOKEN not set",
            )

        try:
            if not is_valid(init_data, bot_token):
                logger.warning("[Auth] Init data validation failed")
                raise HTTPException(
                    status_code=401, detail="Invalid init data signature or expired"
                )

            # Use the library parser first (strict)
            parsed_data = parse(init_data)

            if not parsed_data.get("user"):
                logger.error("[Auth] Parsed data missing user information")
                raise HTTPException(
                    status_code=400, detail="User data not found in init data"
                )

            user_id_from_data = parsed_data["user"].get("id")
            logger.info(
                f"[Auth] Successfully authenticated via initData user_id={user_id_from_data}"
            )
            return parsed_data

        except HTTPException:
            raise
        except ValueError as e:
            # Some initData payloads may contain unexpected enum values (e.g., chat_type='sender')
            # The strict parser may raise a ValueError. Attempt a tolerant manual parse as a fallback.
            logger.warning(
                f"[Auth] Strict parse failed, attempting tolerant parse: {e}"
            )
            try:
                qs = parse_qs(init_data)
                user_raw = None
                if "user" in qs:
                    user_raw = qs.get("user")[0]
                elif "user_id" in qs:
                    # older/alternate formats
                    user_raw = qs.get("user_id")[0]

                parsed_data_tolerant = {"auth_date": None, "hash": None, "user": None}

                if "auth_date" in qs:
                    parsed_data_tolerant["auth_date"] = int(qs.get("auth_date")[0])

                if "hash" in qs:
                    parsed_data_tolerant["hash"] = qs.get("hash")[0]

                if user_raw:
                    # user_raw may be URL-encoded JSON
                    try:
                        user_json = json.loads(unquote_plus(user_raw))
                        parsed_data_tolerant["user"] = user_json
                    except Exception:
                        # Try simple numeric user id
                        try:
                            parsed_data_tolerant["user"] = {"id": int(user_raw)}
                        except Exception:
                            parsed_data_tolerant["user"] = None

                if parsed_data_tolerant.get("user"):
                    logger.info("[Auth] Tolerant init data parse succeeded")
                    return parsed_data_tolerant
                else:
                    logger.error("[Auth] Tolerant parse failed to extract user")
                    raise HTTPException(
                        status_code=401,
                        detail="Init data validation failed (tolerant parse)",
                    )
            except HTTPException:
                raise
            except Exception:
                logger.error("[Auth] Tolerant parse error", exc_info=True)
                raise HTTPException(
                    status_code=401, detail="Init data validation failed"
                )
        except Exception:
            logger.error("[Auth] Init data validation error", exc_info=True)
            raise HTTPException(status_code=401, detail="Init data validation failed")

    # Fallback authentication: URL parameter user_id (for "Open" button outside Telegram context)
    if user_id:
        logger.info(
            f"[Auth] Using fallback authentication with user_id={user_id} (outside Telegram context)"
        )
        # Verify user_id is a valid integer
        try:
            parsed_user_id = int(user_id)
            # Return a minimal user object for fallback authentication
            # This allows the app to work when opened outside of Telegram Mini App
            return {
                "user": {
                    "id": parsed_user_id,
                    "first_name": "User",
                    "last_name": None,
                    "username": None,
                    "language_code": "en",
                    "is_premium": False,
                    "photo_url": None,
                },
                "auth_date": int(datetime.now().timestamp()),
                "hash": "fallback",  # Mark as fallback auth
            }
        except (ValueError, TypeError):
            logger.warning(f"[Auth] Invalid user_id format: {user_id}")
            raise HTTPException(
                status_code=400, detail="Invalid user_id format. Expected integer."
            )

    # No valid authentication method provided
    logger.warning(
        "[Auth] Request received without Authorization header or user_id parameter"
    )
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Include 'Authorization: tma {initData}' header or '?user_id={id}' parameter",
    )


async def get_current_user(
    init_data: Dict = Depends(verify_telegram_init_data),
) -> UserInfo:
    """Extract current user info from validated init data."""
    user_data = init_data.get("user", {})
    return UserInfo(
        id=user_data["id"],
        first_name=user_data.get("first_name", "User"),
        last_name=user_data.get("last_name"),
        username=user_data.get("username"),
        language_code=user_data.get("language_code"),
        is_premium=user_data.get("is_premium", False),
        photo_url=user_data.get("photo_url"),
    )
