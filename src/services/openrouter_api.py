import os
import logging
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI, AuthenticationError, RateLimitError, APIError
from src.services.rate_limiter import RateLimiter, rate_limit
from src.utils.log.telegramlog import telegram_logger
from src.services.model_handlers.model_configs import (
    ModelConfigurations,
    Provider,
    ModelConfig,
)
from src.services.system_message_builder import SystemMessageBuilder
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
        self.client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
        )
        self._load_openrouter_models_from_config()
        self.api_failures = 0
        self.api_last_failure = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300
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
    async def close(self):
        """Close the OpenAI client."""
        await self.client.close()
        self.logger.info("Closed OpenRouter API OpenAI client.")
    def _build_system_message(
        self, model_id: str, context: Optional[List[Dict]] = None, tools: Optional[List] = None
    ) -> str:
        """Return a system message based on model and context, using ModelConfigurations.
        
        Args:
            model_id: The model identifier
            context: Optional conversation context
            tools: Optional tools available to the model. 
                   Note: Base class ignores this parameter. Subclasses (e.g., OpenRouterAPIWithMCP)
                   override this method to include tool-specific instructions.
        
        Returns:
            System message string
        """
        # Add concise hint when there's no context and no custom model config
        model_config = ModelConfigurations.get_all_models().get(model_id)
        add_concise_hint = not context and not model_config
        return SystemMessageBuilder.build_basic_message(model_id, context, add_concise_hint)
    @rate_limit
    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
        quoted_message: Optional[str] = None,
        attachments: Optional[List] = None,
    ) -> Optional[str]:
        """
        Generate response from OpenRouter API.
        
        Args:
            prompt: The user's message/prompt
            context: Optional conversation context
            model: Model identifier to use
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            quoted_message: Optional message being replied to
            attachments: Optional image attachments (currently ignored for OpenRouter text models)
            
        Returns:
            Generated response text or error message
        """
        try:
            if not model or not isinstance(model, str) or not model.strip():
                self.logger.error("Invalid model parameter: must be a non-empty string")
                return "Error: Invalid model specified"
            
            # Log if attachments are provided but will be ignored
            if attachments:
                self.logger.info(f"Attachments provided for OpenRouter model {model} but will be ignored (use generate_vision_response for vision)")
            
            self.logger.info(f"Attempting to get OpenRouter model key for: {model}")
            openrouter_model = ModelConfigurations.get_model_with_fallback(model)
            self.logger.info(f"OpenRouter model key resolved to: {openrouter_model}")
            
            if (
                not openrouter_model
                or not isinstance(openrouter_model, str)
                or not openrouter_model.strip()
            ):
                self.logger.error(
                    f"Invalid OpenRouter model key returned for {model}: {openrouter_model}"
                )
                return f"Error: Could not determine model for {model}"
            if max_tokens is None:
                model_configs = ModelConfigurations.get_all_models()
                model_config = model_configs.get(model)
                if model_config and hasattr(model_config, "max_tokens"):
                    max_tokens = model_config.max_tokens
                else:
                    max_tokens = 32768
            
            system_message = self._build_system_message(model, context)
            
            # Handle quoted message context
            final_prompt = prompt
            if quoted_message:
                final_prompt = f'Replying to: "{quoted_message}"\n\nUser\'s message: {prompt}'
            
            messages = []
            # Use the new capability detection system
            safe_config = ModelConfigurations.get_safe_model_config(model)
            if safe_config["use_system_message"]:
                messages.append({"role": "system", "content": system_message})
            if context:
                messages.extend(
                    [msg for msg in context if "role" in msg and "content" in msg]
                )
            messages.append({"role": "user", "content": final_prompt})

            # Adapt the request parameters based on model capabilities
            request_params = {
                "model": openrouter_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            adapted_params = ModelConfigurations.adapt_request_for_model(model, request_params)

            response = await self.client.chat.completions.create(**adapted_params)
            if response.choices and len(response.choices) > 0:
                message_content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if finish_reason != "stop":
                    self.logger.warning(f"Finish reason: {finish_reason}")
                self.logger.info(
                    f"OpenRouter response length: {len(message_content) if message_content else 0} characters"
                )
                self.api_failures = 0
                return message_content
            self.logger.warning("No valid response from OpenRouter API")
            self.api_failures += 1
            return None
        except AuthenticationError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            error_message = (
                "Authentication error. Please check your OpenRouter API key."
            )
            self.logger.error(f"OpenRouter API authentication error: {str(e)}")
            return f"OpenRouter API error: {error_message}"
        except RateLimitError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            error_message = "Rate limit exceeded. Please try again later."
            self.logger.error(f"OpenRouter API rate limit error: {str(e)}")
            return f"OpenRouter API error: {error_message}"
        except APIError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            error_message = f"API error: {e.message}"
            if hasattr(e, "status") and e.status == 404:
                error_message = (
                    f"Model not found: {model}. Model may be temporarily unavailable."
                )
                self.logger.warning(
                    f"Model {openrouter_model} not found on OpenRouter. This may be temporary."
                )
            elif hasattr(e, "status") and e.status == 400:
                error_message = f"Bad request for model {model}. The model may not support the current request format."
            self.logger.error(
                f"OpenRouter API error for model {model}: {error_message}"
            )
            return f"OpenRouter API error: {error_message}"
        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API error: {str(e)}", exc_info=True)
            return f"Unexpected error when calling OpenRouter API: {str(e)}"
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
            if max_tokens is None:
                max_tokens = 32768
            final_system_message = (
                system_message
                or "You are an advanced AI assistant that helps users with various tasks. Be concise, helpful, and accurate."
            )
            messages = []
            # Use the new capability detection system
            safe_config = ModelConfigurations.get_safe_model_config(openrouter_model_key)
            if safe_config["use_system_message"]:
                messages.append({"role": "system", "content": final_system_message})
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})

            # Adapt the request parameters based on model capabilities
            request_params = {
                "model": openrouter_model_key,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            adapted_params = ModelConfigurations.adapt_request_for_model(openrouter_model_key, request_params)

            response = await self.client.chat.completions.create(**adapted_params)
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                self.api_failures = 0
                return content
            self.api_failures += 1
            return None
        except AuthenticationError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API authentication error: {str(e)}")
            return "Authentication error. Please check your OpenRouter API key."
        except RateLimitError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API rate limit error: {str(e)}")
            return "Rate limit exceeded. Please try again later."
        except APIError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            error_message = f"API error: {e.message}"
            self.logger.error(f"OpenRouter API error: {error_message}")
            return f"OpenRouter API error: {error_message}"
        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API error: {str(e)}")
            return f"Unexpected error when calling OpenRouter API: {str(e)}"
    def debug_model_mapping(self):
        """Debug method to log all available model mappings."""
        self.logger.info("=== OpenRouter Model Mappings ===")
        for model_id, openrouter_key in self.available_models.items():
            self.logger.info(f"  {model_id} -> {openrouter_key}")
        self.logger.info(f"Total models loaded: {len(self.available_models)}")
    def get_system_message(self) -> str:
        """
        Return the system message for OpenRouter models.
        This is used by the prompt formatter for consistent system prompts.
        """
        return (
            "You are an advanced AI assistant that helps users with various tasks. "
            "Be concise, helpful, and accurate."
        )
    @rate_limit
    async def generate_vision_response(
        self,
        prompt: str,
        image_data: bytes,
        model: str = None,
        context: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate response for vision/multimodal content using OpenRouter.
        
        Args:
            prompt: Text prompt for image analysis
            image_data: Raw image bytes
            model: Model to use for generation
            context: Optional conversation context
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout
            
        Returns:
            Generated response text or None if failed
        """
        try:
            if not model or not isinstance(model, str) or not model.strip():
                self.logger.error("Invalid model parameter: must be a non-empty string")
                return "Error: Invalid model specified"
            
            self.logger.info(f"Attempting to get OpenRouter model key for vision: {model}")
            openrouter_model = ModelConfigurations.get_model_with_fallback(model)
            self.logger.info(f"OpenRouter model key resolved to: {openrouter_model}")
            
            if (
                not openrouter_model
                or not isinstance(openrouter_model, str)
                or not openrouter_model.strip()
            ):
                self.logger.error(
                    f"Invalid OpenRouter model key returned for {model}: {openrouter_model}"
                )
                return f"Error: Could not determine model for {model}"
                
            if max_tokens is None:
                model_configs = ModelConfigurations.get_all_models()
                model_config = model_configs.get(model)
                if model_config and hasattr(model_config, "max_tokens"):
                    max_tokens = model_config.max_tokens
                else:
                    max_tokens = 32768
                    
            # Handle BytesIO object and encode image as base64
            import base64
            import io
            
            # Extract bytes from BytesIO if needed
            if isinstance(image_data, io.BytesIO):
                image_data.seek(0)  # Reset position to beginning
                image_bytes = image_data.read()
            else:
                image_bytes = image_data
                
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine image format from data
            image_format = "jpeg"  # default
            if image_bytes.startswith(b'\x89PNG'):
                image_format = "png"
            elif image_bytes.startswith(b'\xff\xd8'):
                image_format = "jpeg"
            elif image_bytes.startswith(b'GIF'):
                image_format = "gif"
            elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
                image_format = "webp"
                
            system_message = self._build_system_message(model, context)
            messages = []
            
            # Use the new capability detection system
            safe_config = ModelConfigurations.get_safe_model_config(model)
            if safe_config["use_system_message"]:
                messages.append({"role": "system", "content": system_message})
                
            if context:
                messages.extend(
                    [msg for msg in context if "role" in msg and "content" in msg]
                )
                
            # Create multimodal message with image
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}"
                        }
                    }
                ]
            }
            messages.append(user_message)
            
            # Adapt the request parameters based on model capabilities
            request_params = {
                "model": openrouter_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            adapted_params = ModelConfigurations.adapt_request_for_model(model, request_params)
            
            self.logger.info(f"Sending vision request to OpenRouter model: {openrouter_model}")
            response = await self.client.chat.completions.create(**adapted_params)
            
            if response.choices and len(response.choices) > 0:
                message_content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                if finish_reason != "stop":
                    self.logger.warning(f"Finish reason: {finish_reason}")
                self.logger.info(
                    f"OpenRouter vision response length: {len(message_content) if message_content else 0} characters"
                )
                self.api_failures = 0
                return message_content
                
            self.logger.warning("No valid response from OpenRouter API for vision request")
            self.api_failures += 1
            return None
            
        except AuthenticationError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            error_message = (
                "Authentication error. Please check your OpenRouter API key."
            )
            self.logger.error(f"OpenRouter API authentication error: {str(e)}")
            return f"OpenRouter API error: {error_message}"
        except RateLimitError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            error_message = "Rate limit exceeded. Please try again later."
            self.logger.error(f"OpenRouter API rate limit error: {str(e)}")
            return f"OpenRouter API error: {error_message}"
        except APIError as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            error_message = f"API error: {e.message}"
            if hasattr(e, "status") and e.status == 404:
                error_message = (
                    f"Model not found: {model}. Model may be temporarily unavailable."
                )
                self.logger.warning(
                    f"Model {openrouter_model} not found on OpenRouter. This may be temporary."
                )
            elif hasattr(e, "status") and e.status == 400:
                error_message = f"Bad request for model {model}. The model may not support vision or the current request format."
            self.logger.error(
                f"OpenRouter API vision error for model {model}: {error_message}"
            )
            return f"OpenRouter API error: {error_message}"
        except Exception as e:
            self.api_failures += 1
            self.api_last_failure = time.time()
            self.logger.error(f"OpenRouter API vision error: {str(e)}", exc_info=True)
            return f"Unexpected error when calling OpenRouter API for vision: {str(e)}"

    def get_model_indicator(self, model: str = None) -> str:
        """
        Return the model indicator for OpenRouter models.
        This is used by text handlers for response formatting.
        
        Args:
            model: Model ID to get indicator for
            
        Returns:
            Formatted string like "🤖 x-ai/grok-4-fast:free"
        """
        if not model:
            return "🤖 OpenRouter"
            
        # Get model configuration
        model_config = ModelConfigurations.get_all_models().get(model)
        if not model_config:
            return f"🤖 {model}"
            
        # Get emoji from configuration system
        emoji = ModelConfigurations._get_indicator_emoji(model_config.provider, model_config.type)
        
        # Get display name or model key
        display_name = model_config.display_name
        if model_config.openrouter_model_key:
            display_name = model_config.openrouter_model_key
            
        return f"{emoji} {display_name}"
