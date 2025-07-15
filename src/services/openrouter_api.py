import os
import json
import aiohttp
import asyncio
import logging
import traceback
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from src.services.rate_limiter import RateLimiter, rate_limit
from src.utils.log.telegramlog import telegram_logger
from src.services.model_handlers.model_configs import ModelConfigurations, Provider

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
        
        # Get available models from centralized configuration
        self._load_available_models()

        # Circuit breaker properties
        self.api_failures = 0
        self.api_last_failure = 0
        self.circuit_breaker_threshold = 5  # Number of failures before opening circuit
        self.circuit_breaker_timeout = 300  # Seconds to keep circuit open (5 minutes)

        # Initialize session
        self.session = None

    def _load_available_models(self):
        """Load available models from centralized configuration"""
        openrouter_models = ModelConfigurations.get_models_by_provider(Provider.OPENROUTER)
        self.available_models = {}
        
        for model_id, config in openrouter_models.items():
            if config.openrouter_model_key:
                self.available_models[model_id] = config.openrouter_model_key
        
        self.logger.info(f"Loaded {len(self.available_models)} OpenRouter models from configuration")

    def get_available_models(self) -> Dict[str, str]:
        """Get the mapping of model IDs to OpenRouter model keys"""
        return self.available_models.copy()

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

    def _build_system_message(self, model: str, context: List[Dict] = None) -> str:
        """Return a system message based on model and context."""
        base = "You are an advanced AI assistant that helps users with various tasks."
        context_hint = " Use conversation history/context when relevant." if context else ""
        model_map = {
            "llama4_maverick": "You are Llama-4 Maverick, an advanced AI assistant by Meta.",
            "deepcoder": "You are DeepCoder, an AI specialized in programming and software development.",
            "deepseek": "You are DeepSeek, an advanced reasoning AI model that excels at complex problem-solving.",
            "qwen": "You are Qwen, a multilingual AI assistant created by Alibaba Cloud.",
            "gemma": "You are Gemma, a lightweight and efficient AI assistant by Google.",
            "mistral": "You are Mistral, a high-performance European AI language model.",
            "phi": "You are Phi, a compact and efficient AI model by Microsoft, specialized in reasoning.",
            "llama": "You are LLaMA, an advanced AI assistant created by Meta.",
            "claude": "You are Claude, a helpful AI assistant created by Anthropic.",
            "hermes": "You are Hermes, a versatile AI assistant optimized for helpful conversations.",
            "olympic": "You are OlympicCoder, an AI specialized in competitive programming and complex algorithms.",
            "magnum": "You are Magnum, an AI optimized for creative writing and storytelling.",
        }
        for key, msg in model_map.items():
            if key in model:
                return msg + context_hint
        return base + (context_hint if context else " Be concise, helpful, and accurate.")

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
        context: List[Dict] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        await self.rate_limiter.acquire()
        try:
            session = await self.ensure_session()
            
            # Get model with fallback support
            openrouter_model = self.get_model_with_fallback(model)
            
            system_message = self._build_system_message(model, context)
            messages = [{"role": "system", "content": system_message}]
            if context:
                messages.extend([msg for msg in context if "role" in msg and "content" in msg])
            messages.append({"role": "user", "content": prompt})
            payload = {"model": openrouter_model, "messages": messages, "temperature": temperature}
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            self.logger.info(f"Sending request to OpenRouter API with model {model} (mapped to {openrouter_model})")
            data = await self._send_openrouter_request(session, payload, timeout)
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                message_content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "unknown")
                if finish_reason != "stop":
                    self.logger.warning(f"Finish reason: {finish_reason}")
                self.logger.info(f"OpenRouter response length: {len(message_content) if message_content else 0} characters")
                return message_content
            self.logger.warning("No valid response from OpenRouter API")
            return None
        except aiohttp.ClientResponseError as e:
            self.api_failures += 1
            error_message = f"Error {e.status}: {e.message}"
            if e.status == 404:
                error_message = f"Model not found: {model}. Model may be temporarily unavailable."
                self.logger.warning(f"Model {openrouter_model} not found on OpenRouter. This may be temporary.")
            elif e.status == 401:
                error_message = "Authentication error. Please check your OpenRouter API key."
            elif e.status == 400:
                error_message = f"Bad request for model {model}. The model may not support the current request format."
            self.logger.error(f"OpenRouter API error for model {model}: {error_message}")
            return f"OpenRouter API error: {error_message}"
        except asyncio.TimeoutError:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API timeout for model {model}")
            return None
        except aiohttp.ClientError as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API connection error: {str(e)}")
            raise Exception(f"OpenRouter API connection error: {str(e)}")
        except json.JSONDecodeError as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API JSON decode error: {str(e)}")
            raise Exception(f"Could not parse OpenRouter API response: {str(e)}")
        except Exception as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API error: {str(e)}")
            raise Exception(f"Unexpected error when calling OpenRouter API: {str(e)}")

    @rate_limit
    async def generate_response_with_model_key(
        self,
        prompt: str,
        openrouter_model_key: str,
        system_message: str = None,
        context: List[Dict] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        await self.rate_limiter.acquire()
        try:
            session = await self.ensure_session()
            messages = [{"role": "system", "content": system_message or "You are an advanced AI assistant that helps users with various tasks. Be concise, helpful, and accurate."}]
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
            self.logger.info(f"Sending request to OpenRouter API with model {openrouter_model_key}")
            response_data = await self._send_openrouter_request(session, data, aiohttp.ClientTimeout(total=timeout))
            if "choices" in response_data and response_data["choices"]:
                content = response_data["choices"][0]["message"]["content"]
                self.api_failures = 0
                return content
            self.api_failures += 1
            return None
        except asyncio.TimeoutError:
            self.api_failures += 1
            self.api_last_failure = time.time()
            return None
        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API error: {str(e)}")
            return None

    def debug_model_mapping(self):
        """Debug method to log all available model mappings"""
        self.logger.info("=== OpenRouter Model Mappings ===")
        for model_id, openrouter_key in self.available_models.items():
            self.logger.info(f"  {model_id} -> {openrouter_key}")
        self.logger.info(f"Total models loaded: {len(self.available_models)}")

    def get_model_with_fallback(self, model_id: str) -> str:
        """Get OpenRouter model key with fallback to reliable alternatives"""
        # Check if we have the exact model
        if model_id in self.available_models:
            return self.available_models[model_id]
        
        # Fallback logic for common model patterns
        fallback_map = {
            "deepseek-r1-distill-qwen-14b": "deepseek/deepseek-chat:free",
            "deepseek-r1-distill-llama-70b": "deepseek/deepseek-chat:free", 
            "gemma-3n-e4b": "google/gemma-2-9b-it:free",
            "qwen": "qwen/qwen-2.5-72b-instruct:free",
            "llama": "meta-llama/llama-3.3-70b-instruct:free"
        }
        
        # Check fallback patterns
        for pattern, fallback in fallback_map.items():
            if pattern in model_id:
                self.logger.warning(f"Model {model_id} not found, using fallback: {fallback}")
                return fallback
        
        # Default fallback
        self.logger.warning(f"No specific fallback for {model_id}, using default: deepseek/deepseek-chat:free")
        return "deepseek/deepseek-chat:free"
