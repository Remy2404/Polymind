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
        
        # Available models - Free models from OpenRouter
        self.available_models = {
            # DeepSeek Models (Free)
            "deepseek-r1-0528-qwen3-8b": "deepseek/deepseek-r1-0528-qwen3-8b:free",
            "deepseek-r1-0528": "deepseek/deepseek-r1-0528:free",
            "deepseek-r1-zero": "deepseek/deepseek-r1-distill-llama-70b:free",
            "deepseek-prover-v2": "deepseek/deepseek-prover-v2:free",
            "deepseek-v3-base": "deepseek/deepseek-v3-base:free",
            "deepseek-chat-v3-0324": "deepseek/deepseek-chat-v3-0324:free",
            "deepseek-r1-distill-llama-70b": "deepseek/deepseek-r1-distill-llama-70b:free",
            "deepseek-r1": "deepseek/deepseek-r1:free",           
            "llama4_maverick": "meta-llama/llama-4-maverick:free",
            "llama-3.3-8b": "meta-llama/llama-3.3-8b-instruct:free",
            "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct:free",
            "llama-3.2-1b": "meta-llama/llama-3.2-1b-instruct:free",            "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
            
            "llama-3-8b": "meta-llama/llama-3-8b-instruct:free",
            # Mistral Models (Free)
            "devstral-small": "mistralai/devstral-small:free",
            "mistral-small-3-1": "mistralai/mistral-small-3.1-24b-instruct:free",
            "mistral-small-3": "mistralai/mistral-small-24b-instruct-2501:free",
            "mistral-small-3.2-24b-instruct": "mistralai/mistral-small-3.2-24b-instruct:free",
            "mistral-7b": "mistralai/mistral-7b-instruct:free",
            # Qwen Models (Free)
            "qwen3-32b-a3b": "qwen/qwen3-32b-a3b:free",
            "qwen3-8b": "qwen/qwen3-8b:free",
            "qwen3-14b": "qwen/qwen3-14b:free",
            "qwen2.5-vl-3b": "qwen/qwen2.5-vl-3b-instruct:free",
            "qwen2.5-vl-7b": "qwen/qwen2.5-vl-7b-instruct:free",
            "qwen2.5-vl-72b": "qwen/qwen2.5-vl-72b-instruct:free",
            "qwen2.5-coder-32b": "qwen/qwen2.5-coder-32b-instruct:free",
            # Google Gemma Models (Free)
            "gemma-2-9b": "google/gemma-2-9b-it:free",
            # Microsoft Phi Models (Free)
            "phi-4-reasoning-plus": "microsoft/phi-4-reasoning-plus:free",
            # NVIDIA LLaMA Models (Free)
            "llama-3.1-nemotron-70b-reasoning": "nvidia/llama-3.1-nemotron-70b-reasoning:free",
            "llama-3.1-nemotron-ultra-253b": "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
            # THUDM Models (Free)
            "glm-4-32b": "thudm/glm-4-32b:free",
            # Nous Research Models (Free)
            "deephermes-3-llama-3-8b": "nousresearch/deephermes-3-llama-3-8b-preview:free",
            # Coding & Programming Models (Free)
            "deepcoder": "agentica-org/deepcoder-14b-preview:free",
            "olympiccoder-32b": "olympiccoder/olympiccoder-32b:free",
            # Creative Writing Models (Free)
            "magnum-v4-72b": "anthracite-org/magnum-v4-72b:free",
            # Other Free Models
            "optimus-alpha": "openrouter/optimus-alpha",
            "liquid-lfm-40b": "liquid/lfm-40b:free",
            "hermes-3-llama-3.1-405b": "nousresearch/hermes-3-llama-3.1-405b:free",
            "qwen1.5-110b": "qwen/qwen-1.5-110b-chat:free",
            "fimbulvetr-11b": "sao10k/fimbulvetr-11b-v2:free",
            "mythomax-l2-13b": "gryphe/mythomax-l2-13b:free",
            # Additional Free Models for Extended Collection
            "command-r": "cohere/command-r:free",
            "yi-large": "01-ai/yi-large:free",
            "claude-3-haiku": "anthropic/claude-3-haiku:free",
            "gpt-3.5-turbo": "openai/gpt-3.5-turbo:free",
            "gemini-flash-1.5": "google/gemini-flash-1.5:free",
            "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct:free",
            "codellama-70b": "codellama/codellama-70b-instruct:free",
            "solar-10.7b": "upstage/solar-10.7b-instruct:free",
            "openchat-3.5": "openchat/openchat-3.5-1210:free",
            "zephyr-7b": "huggingfaceh4/zephyr-7b-beta:free",
            "starling-lm-7b": "berkeley-nest/starling-lm-7b-alpha:free",
            "airoboros-70b": "jondurbin/airoboros-l2-70b:free",
            "moonshot-kimi-dev-72b": "moonshotai/kimi-dev-72b:free",
            "mai-ds-r1": "microsoft/mai-ds-r1:free",
            "qwen-2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct:free",
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
        model: str = "deepseek-r1-zero",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """Generate a text response using the OpenRouter API."""
        await self.rate_limiter.acquire()

        try:
            # Ensure we have a session
            session = await self.ensure_session()

            # Map model ID to OpenRouter model path
            openrouter_model = self.available_models.get(model, model)            # Log the model mapping
            self.logger.info(f"OpenRouter model mapping: {model} -> {openrouter_model}")

            # Prepare the messages
            messages = []
            # Customize system message based on model and context
            if context:
                # Enhanced system message when conversation context is available
                system_message = "You are an advanced AI assistant that helps users with various tasks. You have access to the conversation history, so please refer to previous messages when relevant. Pay attention to personal information shared earlier, like names, preferences, and ongoing topics. Be conversational, helpful, and remember context from earlier in our discussion."
            else:
                # Basic system message when no context is available
                system_message = "You are an advanced AI assistant that helps users with various tasks. Be concise, helpful, and accurate."

            # Customize system message further based on specific model
            if model == "llama4_maverick":
                system_message = (
                    "You are Llama-4 Maverick, an advanced AI assistant by Meta. "
                    + ("Pay attention to conversation history and refer to previous messages when relevant." if context else "")
                )
            elif model == "deepcoder":
                system_message = "You are DeepCoder, an AI specialized in programming and software development." + (" Use conversation context when relevant." if context else "")
            elif "deepseek" in model:
                system_message = "You are DeepSeek, an advanced reasoning AI model that excels at complex problem-solving." + (" Use conversation history to provide contextual responses." if context else "")
            elif "qwen" in model:
                system_message = "You are Qwen, a multilingual AI assistant created by Alibaba Cloud." + (" Reference previous messages in our conversation when helpful." if context else "")
            elif "gemma" in model:
                system_message = (
                    "You are Gemma, a lightweight and efficient AI assistant by Google." + (" Be aware of conversation context and refer to earlier messages." if context else "")
                )
            elif "mistral" in model:
                system_message = (
                    "You are Mistral, a high-performance European AI language model." + (" Use conversation history to maintain context." if context else "")
                )
            elif "phi" in model:
                system_message = "You are Phi, a compact and efficient AI model by Microsoft, specialized in reasoning." + (" Pay attention to conversation flow and context." if context else "")
            elif "llama" in model:
                system_message = (
                    "You are LLaMA, an advanced AI assistant created by Meta." + (" Remember information from our ongoing conversation." if context else "")
                )
            elif "claude" in model:
                system_message = (
                    "You are Claude, a helpful AI assistant created by Anthropic." + (" Use conversation history to provide better responses." if context else "")
                )
            elif "hermes" in model:
                system_message = "You are Hermes, a versatile AI assistant optimized for helpful conversations." + (" Maintain conversation context and refer to previous messages." if context else "")
            elif "olympic" in model:
                system_message = "You are OlympicCoder, an AI specialized in competitive programming and complex algorithms." + (" Consider previous discussion context." if context else "")
            elif "magnum" in model:
                system_message = "You are Magnum, an AI optimized for creative writing and storytelling." + (" Build on our conversation history." if context else "")

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
            messages.append({"role": "user", "content": prompt})

            # Prepare the payload
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
            # Send the request
            async with session.post(
                self.api_url, headers=self.headers, json=payload, timeout=timeout
            ) as response:
                response.raise_for_status()
                data = await response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    message_content = choice["message"]["content"]

                    # Check if the response was truncated
                    finish_reason = choice.get("finish_reason", "unknown")
                    if finish_reason == "length":
                        self.logger.warning(
                            f"Response was truncated due to max_tokens limit. Consider increasing max_tokens from {max_tokens}"
                        )
                    elif finish_reason != "stop":
                        self.logger.warning(
                            f"Unexpected finish reason: {finish_reason}"
                        )

                    # Log response details for debugging
                    response_length = len(message_content) if message_content else 0
                    self.logger.info(
                        f"OpenRouter response length: {response_length} characters, finish_reason: {finish_reason}"
                    )

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

        except asyncio.TimeoutError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API timeout for model {model}: {str(e)}")
            # For DeepSeek models, provide a specific timeout message that triggers fallback
            if "deepseek" in model.lower():
                return None 
            else:
                return None

        except aiohttp.ClientError as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API connection error: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise Exception(f"OpenRouter API connection error: {str(e)}")

        except json.JSONDecodeError as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API JSON decode error: {str(e)}")
            raise Exception(f"Could not parse OpenRouter API response: {str(e)}")

        except Exception as e:
            self.api_failures += 1
            self.logger.error(f"OpenRouter API error: {str(e)}")
            self.logger.error(traceback.format_exc())
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
        """Generate a text response using direct OpenRouter model key."""
        await self.rate_limiter.acquire()

        try:
            # Ensure we have a session
            session = await self.ensure_session()

            self.logger.info(f"Using OpenRouter model key: {openrouter_model_key}")

            # Prepare the messages
            messages = []
            
            # Add system message
            if system_message:
                messages.append({"role": "system", "content": system_message})
            else:
                messages.append({
                    "role": "system",
                    "content": "You are an advanced AI assistant that helps users with various tasks. Be concise, helpful, and accurate."
                })            # Add context messages if provided
            if context:
                self.logger.info(f"Adding {len(context)} context messages to OpenRouter request")
                for i, msg in enumerate(context):
                    messages.append(msg)
                    if i < 3:  # Log first 3 context messages for debugging
                        self.logger.debug(f"Context message {i+1}: [{msg.get('role', 'unknown')}] {msg.get('content', '')[:100]}...")
            else:
                self.logger.info("No context messages provided")

            # Add the user prompt
            messages.append({"role": "user", "content": prompt})

            # Prepare the request data
            data = {
                "model": openrouter_model_key,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            self.logger.info(f"Sending request to OpenRouter API with model {openrouter_model_key}")

            # Make the API request
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                response_text = await response.text()

                if response.status == 200:
                    response_data = await response.json()
                    
                    if "choices" in response_data and response_data["choices"]:
                        content = response_data["choices"][0]["message"]["content"]
                        self.logger.info(f"Successfully received response from OpenRouter ({len(content)} chars)")
                        
                        # Reset circuit breaker on success
                        self.api_failures = 0
                        return content
                    else:
                        self.logger.error(f"No choices in OpenRouter response: {response_data}")
                        return None
                else:
                    self.logger.error(f"OpenRouter API HTTP error: {response.status} - {response.reason}")
                    self.logger.error(f"Response text: {response_text}")
                    
                    # Increment failure counter
                    self.api_failures += 1
                    self.api_last_failure = time.time()
                    
                    return None

        except asyncio.TimeoutError:
            self.logger.error(f"OpenRouter API timeout after {timeout} seconds")
            self.api_failures += 1
            self.api_last_failure = time.time()
            return None
        except Exception as e:
            self.logger.error(f"OpenRouter API error: {str(e)}")
            self.api_failures += 1
            self.api_last_failure = time.time()
            return None
