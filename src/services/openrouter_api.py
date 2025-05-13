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

        # Available models
        self.available_models = {
            "optimus-alpha": "openrouter/optimus-alpha",
            "deepcoder": "agentica-org/deepcoder-14b-preview:free",
            "llama4_maverick": "meta-llama/llama-4-maverick:free",
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
        model: str = "llama4_maverick",
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> Optional[str]:
        """Generate a text response using the OpenRouter API."""
        await self.rate_limiter.acquire()

        try:
            # Ensure we have a session
            session = await self.ensure_session()

            # Map model ID to OpenRouter model path
            openrouter_model = self.available_models.get(model, model)

            # Log the model mapping
            self.logger.info(f"OpenRouter model mapping: {model} -> {openrouter_model}")

            # Prepare the messages
            messages = []  # Add system message

            # Customize system message based on model
            system_message = "You are an advanced AI assistant that helps users with various tasks. Be concise, helpful, and accurate."
            if model == "llama4_maverick":
                system_message = (
                    "You are Llama-4 Maverick, an advanced AI assistant by Meta."
                )
            elif model == "deepcoder":
                system_message = "You are DeepCoder, an AI specialized in programming and software development."

            messages.append(
                {
                    "role": "system",
                    "content": system_message,
                }
            )

            # Add context messages if provided
            if context:
                for msg in context:
                    if "role" in msg and "content" in msg:
                        messages.append(msg)

            # Add the current prompt
            messages.append({"role": "user", "content": prompt})  # Prepare the payload
            payload = {
                "model": openrouter_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            self.logger.info(
                f"Sending request to OpenRouter API with model {model} (mapped to {openrouter_model})"
            )

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
        except aiohttp.ClientResponseError as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API HTTP error: {e.status} - {e.message}")
            error_message = f"Error {e.status}: {e.message}"
            if e.status == 404:
                error_message = f"Model not found: {model}. Please check if the model ID is correct."
            elif e.status == 401:
                error_message = (
                    "Authentication error. Please check your OpenRouter API key."
                )
            return f"OpenRouter API error: {error_message}"

        except aiohttp.ClientError as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API connection error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "OpenRouter API connection error. Please try again later."

        except json.JSONDecodeError as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API JSON decode error: {str(e)}")
            return "Could not parse OpenRouter API response."

        except Exception as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return f"Unexpected error when calling OpenRouter API: {str(e)}"
