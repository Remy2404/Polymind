import time
import asyncio
from collections import deque
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class RateLimiter:
    """Enhanced rate limiter with bursty traffic handling capabilities"""
    
    def __init__(self, requests_per_minute=10, burst_size=None):
        """
        Initialize the rate limiter
        
        Args:
            requests_per_minute: Maximum number of requests allowed per minute
            burst_size: Optional burst size to allow occasional traffic spikes
        """
        self.rate = requests_per_minute
        self.interval = 60.0 / self.rate  # Time between requests in seconds
        self.last_check = time.monotonic()
        self.tokens = 0
        self.max_tokens = burst_size or requests_per_minute
        self.lock = asyncio.Lock()
        self.timestamps = deque(maxlen=requests_per_minute)
        
        # Initialize with full burst capacity
        self.tokens = self.max_tokens
        logger.info(f"Rate limiter initialized: {requests_per_minute} rpm, {self.interval:.2f}s interval, "
                   f"burst capacity: {self.max_tokens}")

    async def acquire(self):
        """
        Acquire permission to proceed, waiting if necessary.
        Uses the token bucket algorithm for better handling of bursty traffic.
        """
        async with self.lock:
            # Add tokens based on time passed
            now = time.monotonic()
            time_passed = now - self.last_check
            self.last_check = now
            
            # Calculate new tokens based on time elapsed
            new_tokens = time_passed * self.rate / 60.0
            self.tokens = min(self.max_tokens, self.tokens + new_tokens)
            
            # If we have at least one token, consume it and proceed
            if self.tokens >= 1:
                self.tokens -= 1
                self.timestamps.append(now)
                return 0  # No wait needed
            
            # Calculate required wait time
            if self.timestamps:
                # Calculate time until next token is available
                expected_wait = 60.0 / self.rate - (now - self.timestamps[0])
                wait_time = max(0, expected_wait)
            else:
                wait_time = self.interval
            
            return wait_time

    async def wait(self):
        """Wait until we're allowed to proceed"""
        wait_time = await self.acquire()
        
        if wait_time > 0:
            logger.debug(f"Rate limit reached, waiting for {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            # Reacquire after waiting to ensure we have capacity
            await self.acquire()

    async def get_current_capacity(self) -> float:
        """Get the current capacity as a percentage (0-100)"""
        async with self.lock:
            # Calculate current tokens based on time passed
            now = time.monotonic()
            time_passed = now - self.last_check
            
            # Calculate new tokens based on time elapsed
            new_tokens = time_passed * self.rate / 60.0
            current_tokens = min(self.max_tokens, self.tokens + new_tokens)
            
            # Return capacity as percentage
            return (current_tokens / self.max_tokens) * 100.0

    async def __aenter__(self):
        """Async context manager entry - acquire rate limit"""
        await self.wait()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - no release needed for rate limiter"""
        # Rate limiters don't need explicit release like semaphores
        return False

    def release(self):
        """Compatibility method - rate limiters don't need explicit release"""
        # This method exists for compatibility but does nothing
        # Rate limiters work on time-based token buckets, not explicit acquire/release
        pass

def rate_limit(f=None, *, rate_limiter=None):
    """
    Decorator to apply rate limiting to a function
    
    Can be used as @rate_limit or @rate_limit(rate_limiter=my_limiter)
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal rate_limiter
            
            # If no rate limiter specified, try to get from self
            if rate_limiter is None and len(args) > 0 and hasattr(args[0], 'rate_limiter'):
                rate_limiter = args[0].rate_limiter
            
            if rate_limiter is not None:
                await rate_limiter.wait()
                
            return await func(*args, **kwargs)
        return wrapper
    
    if f is None:
        return decorator
    return decorator(f)

class UserRateLimiter:
    def __init__(self, requests_per_hour: int):
        """
        Initialize a per-user rate limiter.
        """
        self.user_limiters = {}
        self.requests_per_hour = requests_per_hour
        self.window_size = 3600  # 1 hour in seconds
        self.lock = asyncio.Lock()

    async def acquire_user(self, user_id: int):
        """
        Acquire a token for a specific user.
        """
        async with self.lock:
            if user_id not in self.user_limiters:
                self.user_limiters[user_id] = RateLimiter(requests_per_minute=self.requests_per_hour / 60)
            
            limiter = self.user_limiters[user_id]
        
        await limiter.wait()

    async def get_user_capacity(self, user_id: int) -> float:
        """
        Get the current capacity for a specific user.
        """
        async with self.lock:
            limiter = self.user_limiters.get(user_id)
            if not limiter:
                return 100.0  # Full capacity
                
        return await limiter.get_current_capacity()
    
class GlobalRateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rate_limiter = RateLimiter(requests_per_minute)

    async def acquire_global(self):
        await self.rate_limiter.wait()

    async def get_global_capacity(self) -> float:
        return await self.rate_limiter.get_current_capacity()