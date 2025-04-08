import os
import json
import aiohttp
import logging
import traceback
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from services.rate_limiter import RateLimiter, rate_limit
from utils.telegramlog import telegram_logger

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    telegram_logger.log_error(
        "OPENROUTER_API_KEY not found in environment variables.", 0
    )


class OpenRouterAPI:
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)
        telegram_logger.log_message("Initializing OpenRouter API", 0)

        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found or empty")

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }

        # Circuit breaker properties
        self.api_failures = 0
        self.api_last_failure = 0
        self.circuit_breaker_threshold = 5  # Number of failures before opening circuit
        self.circuit_breaker_timeout = 300  # Seconds to keep circuit open (5 minutes)

        # Initialize session
        self.session = None

    async def ensure_session(self):
        """Create or reuse aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            self.logger.info("Created new OpenRouter API aiohttp session")
        return self.session

    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Closed OpenRouter API aiohttp session")
            self.session = None

    @rate_limit
    async def generate_response(
        self,
        prompt: str,
        context: List[Dict] = None,
        model: str = "openrouter/quasar-alpha",
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> Optional[str]:
        """Generate a text response using the OpenRouter API."""
        await self.rate_limiter.acquire()

        try:
            # Ensure we have a session
            session = await self.ensure_session()

            # Prepare the messages
            messages = []

            # Add system message
            messages.append(
                {
                    "role": "system",
                    "content": "You are Quasar Alpha, an advanced AI assistant that helps users with various tasks. Be concise, helpful, and accurate.",
                }
            )

            # Add context messages if provided
            if context:
                for msg in context:
                    if "role" in msg and "content" in msg:
                        messages.append(msg)

            # Add the current prompt
            messages.append({"role": "user", "content": prompt})

            # Prepare the payload
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            self.logger.info(f"Sending request to OpenRouter API with model {model}")

            # Send the request
            async with session.post(
                self.api_url, headers=self.headers, json=payload, timeout=60.0
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    message_content = data["choices"][0]["message"]["content"]
                    return message_content
                else:
                    self.logger.warning("No valid response from OpenRouter API")
                    return None

        except Exception as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
