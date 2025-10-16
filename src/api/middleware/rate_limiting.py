"""
Rate limiting middleware for FastAPI applications.
Protects against abuse and ensures fair usage across users.
"""

import time
import logging
from typing import Dict
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RateLimitStore:
    """Thread-safe in-memory rate limit store with automatic cleanup."""
    
    def __init__(self, cleanup_interval: int = 300):  # 5 minutes
        self.store: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval
    
    def _cleanup_expired(self):
        """Remove expired entries to prevent memory leaks."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        expired_keys = []
        for key, data in self.store.items():
            # Remove entries older than 1 hour
            if current_time - data.get('first_request', 0) > 3600:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.store[key]
        
        self.last_cleanup = current_time
        logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit entries")
    
    def get_request_count(self, key: str, window_start: float) -> int:
        """Get the number of requests in the current time window."""
        self._cleanup_expired()
        
        if key not in self.store:
            return 0
        
        data = self.store[key]
        if data.get('window_start', 0) < window_start:
            # Reset counter for new window
            self.store[key] = {'window_start': window_start, 'count': 0}
            return 0
        
        return data.get('count', 0)
    
    def increment_request_count(self, key: str, window_start: float) -> int:
        """Increment the request count and return the new count."""
        if key not in self.store or self.store[key].get('window_start', 0) < window_start:
            # New window or new key
            self.store[key] = {
                'window_start': window_start,
                'count': 1,
                'first_request': time.time()
            }
            return 1
        
        self.store[key]['count'] += 1
        return self.store[key]['count']


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with different limits for different endpoint types.
    
    Features:
    - Per-user rate limiting based on authenticated user ID
    - Different limits for different endpoint patterns
    - Sliding window implementation
    - Automatic cleanup to prevent memory leaks
    - Detailed logging for monitoring
    """
    
    def __init__(
        self,
        app,
        default_requests_per_minute: int = 60,
        streaming_requests_per_minute: int = 20,
        auth_requests_per_minute: int = 10,
        enable_logging: bool = True
    ):
        super().__init__(app)
        self.default_limit = default_requests_per_minute
        self.streaming_limit = streaming_requests_per_minute
        self.auth_limit = auth_requests_per_minute
        self.enable_logging = enable_logging
        self.store = RateLimitStore()
        
        # Define endpoint patterns and their limits
        self.endpoint_limits = {
            '/webapp/chat/stream': streaming_requests_per_minute,
            '/webapp/chat': streaming_requests_per_minute,
            '/webapp/auth': auth_requests_per_minute,
            '/webapp/login': auth_requests_per_minute,
        }
        
        logger.info(f"Rate limiting initialized - Default: {default_requests_per_minute}/min, "
                   f"Streaming: {streaming_requests_per_minute}/min, Auth: {auth_requests_per_minute}/min")
    
    def _get_user_identifier(self, request: Request) -> str:
        """Extract user identifier from request."""
        # Try to get authenticated user ID first
        if hasattr(request.state, 'user_id'):
            return f"user_{request.state.user_id}"
        
        # Check for Telegram user ID in headers
        auth_header = request.headers.get('authorization', '')
        if auth_header.startswith('tma '):
            # For now, use a hash of the auth data as identifier
            import hashlib
            user_hash = hashlib.md5(auth_header.encode()).hexdigest()[:12]
            return f"tma_{user_hash}"
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            client_ip = forwarded_for.split(',')[0].strip()
        
        return f"ip_{client_ip}"
    
    def _get_rate_limit(self, path: str) -> int:
        """Get the rate limit for a specific endpoint."""
        # Check for exact matches first
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # Check for pattern matches
        if '/stream' in path:
            return self.streaming_limit
        elif '/auth' in path or '/login' in path:
            return self.auth_limit
        
        return self.default_limit
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Determine if rate limiting should be skipped for this request."""
        # Skip rate limiting for health checks and static files
        skip_paths = ['/health', '/favicon.ico', '/static/', '/docs', '/redoc', '/openapi.json']
        return any(request.url.path.startswith(path) for path in skip_paths)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process the request with rate limiting."""
        
        # Skip rate limiting for certain paths
        if self._should_skip_rate_limiting(request):
            return await call_next(request)
        
        # Get user identifier and rate limit
        user_id = self._get_user_identifier(request)
        rate_limit = self._get_rate_limit(request.url.path)
        
        # Calculate current time window (1-minute sliding window)
        current_time = time.time()
        window_start = current_time - (current_time % 60)  # Start of current minute
        
        # Check current request count
        current_count = self.store.get_request_count(user_id, window_start)
        
        if current_count >= rate_limit:
            # Rate limit exceeded
            if self.enable_logging:
                logger.warning(
                    f"Rate limit exceeded for {user_id} on {request.url.path} "
                    f"({current_count}/{rate_limit} requests/minute)"
                )
            
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": rate_limit,
                    "window": "1 minute",
                    "retry_after": int(60 - (current_time % 60))
                },
                headers={
                    "Retry-After": str(int(60 - (current_time % 60))),
                    "X-RateLimit-Limit": str(rate_limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(window_start + 60))
                }
            )
        
        # Increment request count
        new_count = self.store.increment_request_count(user_id, window_start)
        
        # Log request for monitoring
        if self.enable_logging and new_count % 10 == 0:  # Log every 10th request
            logger.info(
                f"Rate limit status for {user_id}: {new_count}/{rate_limit} requests/minute "
                f"on {request.url.path}"
            )
        
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = max(0, rate_limit - new_count)
        response.headers["X-RateLimit-Limit"] = str(rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(window_start + 60))
        
        return response


class StreamingRateLimitMiddleware(RateLimitMiddleware):
    """
    Specialized rate limiting for streaming endpoints with longer time windows.
    """
    
    def __init__(
        self,
        app,
        streaming_requests_per_hour: int = 100,
        concurrent_streams_per_user: int = 3
    ):
        super().__init__(app)
        self.hourly_limit = streaming_requests_per_hour
        self.concurrent_limit = concurrent_streams_per_user
        self.active_streams: Dict[str, int] = defaultdict(int)
        
        logger.info(f"Streaming rate limiting initialized - "
                   f"{streaming_requests_per_hour}/hour, {concurrent_streams_per_user} concurrent")
    
    def _check_hourly_limit(self, user_id: str) -> bool:
        """Check if user has exceeded hourly streaming limit."""
        current_time = time.time()
        hour_start = current_time - (current_time % 3600)  # Start of current hour
        
        hourly_count = self.store.get_request_count(f"{user_id}_hourly", hour_start)
        return hourly_count < self.hourly_limit
    
    def _increment_hourly_count(self, user_id: str) -> int:
        """Increment hourly count and return new count."""
        current_time = time.time()
        hour_start = current_time - (current_time % 3600)
        return self.store.increment_request_count(f"{user_id}_hourly", hour_start)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process streaming requests with additional checks."""
        
        # Only apply to streaming endpoints
        if '/stream' not in request.url.path:
            return await super().dispatch(request, call_next)
        
        user_id = self._get_user_identifier(request)
        
        # Check concurrent streams limit
        if self.active_streams[user_id] >= self.concurrent_limit:
            logger.warning(f"Concurrent streaming limit exceeded for {user_id}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Too many concurrent streams",
                    "limit": self.concurrent_limit,
                    "message": "Please wait for existing streams to complete"
                }
            )
        
        # Check hourly limit
        if not self._check_hourly_limit(user_id):
            logger.warning(f"Hourly streaming limit exceeded for {user_id}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Hourly streaming limit exceeded",
                    "limit": self.hourly_limit,
                    "window": "1 hour"
                }
            )
        
        # Increment concurrent streams counter
        self.active_streams[user_id] += 1
        hourly_count = self._increment_hourly_count(user_id)
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Add streaming-specific headers
            response.headers["X-Streaming-Limit"] = str(self.hourly_limit)
            response.headers["X-Streaming-Used"] = str(hourly_count)
            response.headers["X-Concurrent-Limit"] = str(self.concurrent_limit)
            response.headers["X-Concurrent-Active"] = str(self.active_streams[user_id])
            
            return response
            
        finally:
            # Always decrement concurrent streams counter
            self.active_streams[user_id] = max(0, self.active_streams[user_id] - 1)
            if self.active_streams[user_id] == 0:
                del self.active_streams[user_id]