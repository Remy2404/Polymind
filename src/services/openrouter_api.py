import os
import json
import aiohttp
import asyncio
import logging
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from src.services.rate_limiter import RateLimiter, rate_limit
from src.utils.log.telegramlog import telegram_logger
from src.services.model_handlers.model_configs import (
    ModelConfigurations,
    Provider,
    ModelConfig,
) 

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
        self._load_openrouter_models_from_config()  

        # Circuit breaker properties
        self.api_failures = 0
        self.api_last_failure = 0
        self.circuit_breaker_threshold = 5 
        self.circuit_breaker_timeout = 300

        # Initialize session
        self.session = None

    def _load_openrouter_models_from_config(self):
        """Load available models from centralized configuration specific to OpenRouter."""
        openrouter_configs = ModelConfigurations.get_models_by_provider(
            Provider.OPENROUTER
        )
        self.available_models: Dict[str, str] = {
            model_id: config.openrouter_model_key
            for model_id, config in openrouter_configs.items()
            if config.openrouter_model_key is not None
        }
        self.logger.info(
            f"Loaded {len(self.available_models)} OpenRouter models from configuration."
        )

    def get_available_models(self) -> Dict[str, str]:
        """Get the mapping of model IDs to OpenRouter model keys."""
        return self.available_models.copy()

    async def ensure_session(self):
        """Create or reuse aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            self.logger.info("Created new OpenRouter API aiohttp session.")
        return self.session

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Closed OpenRouter API aiohttp session.")
            self.session = None

    def _build_system_message(self, model_id: str, context: Optional[List[Dict]] = None) -> str:
        """Return a system message based on model and context, using ModelConfigurations."""
        model_config: Optional[ModelConfig] = ModelConfigurations.get_all_models().get(
            model_id
        )
        if model_config and model_config.system_message:
            base_message = model_config.system_message
        else:
            base_message = (
                "You are an advanced AI assistant that helps users with various tasks."
            )

        context_hint = (
            " Use conversation history/context when relevant." if context else ""
        )
        if not context and not model_config:
            # If no context is provided and no specific model config, add a general helpfulness hint.
            return base_message + " Be concise, helpful, and accurate."
        return base_message + context_hint

    async def _send_openrouter_request(self, session, payload, timeout):
        async with session.post(
            self.api_url, headers=self.headers, json=payload, timeout=timeout
        ) as response:
            response.raise_for_status()
            return await response.json()

    @rate_limit
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        try:
            session = await self.ensure_session()

            # Get model with fallback support from centralized config
            openrouter_model = ModelConfigurations.get_model_with_fallback(model)

            system_message = self._build_system_message(model, context)
            messages = [{"role": "system", "content": system_message}]
            if context:
                messages.extend(
                    [msg for msg in context if "role" in msg and "content" in msg]
                )
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": openrouter_model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            self.logger.info(
                f"Sending request to OpenRouter API with model {model} (mapped to {openrouter_model})"
            )
            data = await self._send_openrouter_request(session, payload, timeout)
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                message_content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "unknown")
                if finish_reason != "stop":
                    self.logger.warning(f"Finish reason: {finish_reason}")
                self.logger.info(
                    f"OpenRouter response length: {len(message_content) if message_content else 0} characters"
                )
                self.api_failures = 0  # Reset failures on successful response
                return message_content
            self.logger.warning("No valid response from OpenRouter API")
            self.api_failures += 1
            return None
        except aiohttp.ClientResponseError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            error_message = f"Error {e.status}: {e.message}"
            if e.status == 404:
                error_message = (
                    f"Model not found: {model}. Model may be temporarily unavailable."
                )
                self.logger.warning(
                    f"Model {openrouter_model} not found on OpenRouter. This may be temporary."
                )
            elif e.status == 401:
                error_message = (
                    "Authentication error. Please check your OpenRouter API key."
                )
            elif e.status == 400:
                error_message = f"Bad request for model {model}. The model may not support the current request format."
            self.logger.error(
                f"OpenRouter API error for model {model}: {error_message}"
            )
            return f"OpenRouter API error: {error_message}"
        except asyncio.TimeoutError:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API timeout for model {model}")
            return None
        except aiohttp.ClientError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API connection error: {str(e)}")
            raise Exception(f"OpenRouter API connection error: {str(e)}")
        except json.JSONDecodeError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API JSON decode error: {str(e)}")
            raise Exception(f"Could not parse OpenRouter API response: {str(e)}")
        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(
                f"OpenRouter API error: {str(e)}", exc_info=True
            )  # Log full traceback
            raise Exception(f"Unexpected error when calling OpenRouter API: {str(e)}")

    @rate_limit
    async def generate_response_with_model_key(
        self,
        prompt: str,
        openrouter_model_key: str,
        system_message: Optional[str] = None,
        context: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        try:
            session = await self.ensure_session()
            final_system_message = (
                system_message
                or "You are an advanced AI assistant that helps users with various tasks. Be concise, helpful, and accurate."
            )
            messages = [{"role": "system", "content": final_system_message}]
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})
            data = {
                "model": openrouter_model_key,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                data["max_tokens"] = max_tokens
            self.logger.info(
                f"Sending request to OpenRouter API with model key {openrouter_model_key}"
            )
            response_data = await self._send_openrouter_request(
                session, data, aiohttp.ClientTimeout(total=timeout)
            )
            if "choices" in response_data and response_data["choices"]:
                content = response_data["choices"][0]["message"]["content"]
                self.api_failures = 0
                return content
            self.api_failures += 1
            return None
        except asyncio.TimeoutError:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(
                f"OpenRouter API timeout for model keys {openrouter_model_key}"
            )
            return None
        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            return None

    def debug_model_mapping(self):
        """Debug method to log all available model mappings."""
        self.logger.info("=== OpenRouter Model Mappings ===")
        for model_id, openrouter_key in self.available_models.items():
            self.logger.info(f"  {model_id} -> {openrouter_key}")
        self.logger.info(f"Total models loaded: {len(self.available_models)}")
