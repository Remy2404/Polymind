import asyncio
import time
from collections import deque
import logging

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter with a sliding window approach.
        """
        self.rate = requests_per_minute
        self.window_size = 60  # Window size in seconds
        self.max_tokens = requests_per_minute
        self.tokens = self.max_tokens
        self.requests = deque()
        self.lock = asyncio.Lock()
        self.last_update = time.time()
        
        # Burst handling
        self.burst_size = min(10, requests_per_minute // 2)  # Allow burst up to 10 requests
        self.burst_tokens = self.burst_size

    async def acquire(self):
        """
        Acquire a token for making a request, with optimized waiting.
        """
        async with self.lock:
            now = time.time()
            
            # Clean up old requests
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            # Check if we can use burst capacity
            if len(self.requests) < self.rate:
                if self.burst_tokens > 0:
                    self.burst_tokens -= 1
                    self.requests.append(now)
                    return
            
            # Calculate wait time if needed
            if len(self.requests) >= self.rate:
                wait_time = self.requests[0] + self.window_size - now
                if wait_time > 0:
                    # Split wait time into smaller chunks for more responsive cancellation
                    chunk_size = 0.1  # 100ms chunks
                    chunks = int(wait_time / chunk_size)
                    
                    for _ in range(chunks):
                        await asyncio.sleep(chunk_size)
                        # Could add cancellation check here if needed
                    
                    # Wait remaining time
                    remaining = wait_time - (chunks * chunk_size)
                    if remaining > 0:
                        await asyncio.sleep(remaining)
            
            # Add current request to the window
            self.requests.append(now)
            
            # Replenish burst tokens periodically
            time_since_update = now - self.last_update
            if time_since_update >= 60:  # Replenish every minute
                self.burst_tokens = min(
                    self.burst_size, 
                    self.burst_tokens + int(time_since_update / 60) * self.burst_size
                )
                self.last_update = now

    async def get_current_capacity(self) -> float:
        """
        Get current available capacity as a percentage.
        """
        async with self.lock:
            now = time.time()
            # Clean up old requests
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            used_capacity = len(self.requests)
            return (self.rate - used_capacity) / self.rate * 100

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
        
        await limiter.acquire()

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
        await self.rate_limiter.acquire()

    async def get_global_capacity(self) -> float:
        return await self.rate_limiter.get_current_capacity()