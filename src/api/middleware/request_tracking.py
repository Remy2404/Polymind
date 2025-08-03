"""
Request tracking middleware for FastAPI applications.
Provides request ID generation, timing, and logging functionality.
"""

import time
import uuid
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking and logging HTTP requests.
    Adds request ID, timing information, and structured logging.
    """

    async def dispatch(self, request: Request, call_next):
        # Generate unique request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timer for performance monitoring
        start_time = time.time()

        # Create logger with context
        logger_with_context = logging.LoggerAdapter(logger, {"request_id": request_id})

        # Log incoming request
        logger_with_context.info(
            f"Request started: {request.method} {request.url.path}"
        )

        try:
            # Process the request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Add custom headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id

            # Log successful response with timing
            logger_with_context.info(
                f"Request completed: {request.method} {request.url.path} "
                f"- Status: {response.status_code} - Time: {process_time:.3f}s"
            )

            # Log slow requests for monitoring
            if process_time > 1.0:
                logger_with_context.warning(
                    f"Slow request detected: {process_time:.3f}s"
                )

            return response

        except Exception as e:
            # Log exception with full details
            process_time = time.time() - start_time
            logger_with_context.error(
                f"Request failed: {request.method} {request.url.path} "
                f"- Error: {str(e)} - Time: {process_time:.3f}s",
                exc_info=True,
            )

            # Return proper error response
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "request_id": request_id},
            )
