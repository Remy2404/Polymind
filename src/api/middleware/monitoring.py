"""
Enhanced error handling and monitoring for streaming responses.
Optimized for 512MB RAM environment - Minimal in-memory state.
"""

import time
import json
import logging
import traceback
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of different error types."""
    VALIDATION_ERROR = "validation_error"
    MODEL_ERROR = "model_error"
    STREAMING_ERROR = "streaming_error"
    DATABASE_ERROR = "database_error"
    AUTHENTICATION_ERROR = "auth_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorEvent:
    """Structured error event for logging context."""
    timestamp: float
    error_type: ErrorType
    severity: ErrorSeverity
    user_id: Optional[str]
    endpoint: str
    error_message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    request_id: Optional[str] = None
    model_name: Optional[str] = None
    duration_ms: Optional[float] = None


class StreamingErrorHandler:
    """Specialized error handler for streaming responses. Lightweight version."""

    def __init__(self):
        pass

    @asynccontextmanager
    async def handle_streaming_errors(
        self,
        user_id: str,
        endpoint: str,
        model_name: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Context manager for handling streaming errors."""
        start_time = time.time()

        try:
            yield
            # Success - Optional: Log success if needed, but keeping it minimal
            # duration_ms = (time.time() - start_time) * 1000
            # logger.debug(f"Streaming request success: {endpoint} ({duration_ms:.2f}ms)")

        except asyncio.CancelledError:
            logger.info(f"Streaming request cancelled by client for user {user_id}")
            raise

        except Exception as e:
            # Actual error
            duration_ms = (time.time() - start_time) * 1000

            # Classify error
            error_type, severity = self._classify_error(e)

            # Create error event for logging
            error_event = ErrorEvent(
                timestamp=time.time(),
                error_type=error_type,
                severity=severity,
                user_id=user_id,
                endpoint=endpoint,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                context={
                    "model_name": model_name,
                    "duration_ms": duration_ms,
                    "error_class": e.__class__.__name__,
                },
                request_id=request_id,
                model_name=model_name,
                duration_ms=duration_ms,
            )

            # Direct logging instead of in-memory storing
            self._log_error(error_event)
            raise

    def _log_error(self, error_event: ErrorEvent) -> None:
        """Log error based on severity."""
        event_dict = asdict(error_event)
        # Convert Enum to string for JSON serialization compatibility in logs if needed
        event_dict['error_type'] = error_event.error_type.value
        event_dict['severity'] = error_event.severity.value

        if error_event.severity == ErrorSeverity.CRITICAL:
            logger.critical(
                f"CRITICAL ERROR: {error_event.error_message}",
                extra={"error_event": event_dict},
            )
        elif error_event.severity == ErrorSeverity.HIGH:
            logger.error(
                f"HIGH SEVERITY ERROR: {error_event.error_message}",
                extra={"error_event": event_dict},
            )
        elif error_event.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {error_event.error_message}")
        else:
            logger.info(f"LOW SEVERITY ERROR: {error_event.error_message}")

    def _classify_error(self, error: Exception) -> tuple[ErrorType, ErrorSeverity]:
        """Classify an error by type and severity."""
        error_class = error.__class__.__name__
        error_message = str(error).lower()

        # Classification logic
        if "validation" in error_message or "pydantic" in error_class.lower():
            return ErrorType.VALIDATION_ERROR, ErrorSeverity.LOW

        elif "rate limit" in error_message or "429" in error_message:
            return ErrorType.RATE_LIMIT_ERROR, ErrorSeverity.MEDIUM

        elif (
            "auth" in error_message
            or "unauthorized" in error_message
            or "403" in error_message
        ):
            return ErrorType.AUTHENTICATION_ERROR, ErrorSeverity.MEDIUM

        elif "database" in error_message or "mongo" in error_message:
            return ErrorType.DATABASE_ERROR, ErrorSeverity.HIGH

        elif (
            "model" in error_message
            or "openai" in error_message
            or "gemini" in error_message
        ):
            return ErrorType.MODEL_ERROR, ErrorSeverity.MEDIUM

        elif (
            "network" in error_message
            or "connection" in error_message
            or "timeout" in error_message
        ):
            return ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM

        elif "stream" in error_message:
            return ErrorType.STREAMING_ERROR, ErrorSeverity.HIGH

        else:
            # Unknown error - could be critical
            if "critical" in error_message or "fatal" in error_message:
                return ErrorType.UNKNOWN_ERROR, ErrorSeverity.CRITICAL
            else:
                return ErrorType.UNKNOWN_ERROR, ErrorSeverity.MEDIUM

    async def create_error_response_stream(
        self, error_message: str, error_type: str = "error"
    ) -> str:
        """Create a standardized error response for streaming endpoints."""
        error_response = {
            "type": "error",
            "error": error_message,
            "error_type": error_type,
            "timestamp": datetime.now().timestamp(),
            "recoverable": error_type in ["rate_limit_error", "validation_error"],
        }
        return f"data: {json.dumps(error_response)}\n\n"


# Global instances
streaming_error_handler = StreamingErrorHandler()


def get_monitoring_stats() -> Dict[str, Any]:
    """Get comprehensive monitoring statistics. (Stubbed for low RAM)"""
    return {
        "status": "active",
        "mode": "low_ram",
        "uptime_checkpoint": time.time()
    }
