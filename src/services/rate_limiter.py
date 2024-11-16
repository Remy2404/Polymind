import asyncio
import time
from collections import deque

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        """
        Initialize rate limiter with a sliding window approach
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
        Acquire a token for making a request, with optimized waiting
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
                self.burst_tokens = min(self.burst_size, 
                                      self.burst_tokens + int(time_since_update / 60) * self.burst_size)
                self.last_update = now

    async def get_current_capacity(self) -> float:
        """
        Get current available capacity as a percentage
        """
        async with self.lock:
            now = time.time()
            # Clean up old requests
            while self.requests and self.requests[0] <= now - self.window_size:
                self.requests.popleft()
            
            used_capacity = len(self.requests)
            return (self.rate - used_capacity) / self.rate * 100