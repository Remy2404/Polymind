import asyncio
import time

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.rate = requests_per_minute
        self.interval = 60.0 / requests_per_minute  # Time between requests
        self.last_request = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_request
            if time_passed < self.interval:
                wait_time = self.interval - time_passed
                await asyncio.sleep(wait_time)
            
            self.last_request = time.time()
