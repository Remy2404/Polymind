"""
Enhanced OpenRouter API with MCP Tool Integration - Improved Version
This module extends the base OpenRouter API to include automatic MCP server
integration, enabling LLMs to use MCP tools seamlessly.

Improvements:
- Better error handling with specific error types
- Optimized performance with caching and reduced redundant operations
- Improved code structure and separation of concerns
- Enhanced logging and monitoring
- Circuit breaker pattern implementation
- Better type hints and documentation
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from functools import lru_cache
from src.api.llm.openrouter_api import OpenRouterAPI
from src.services.mcp import MCPManager
from src.services.model_handlers.model_configs import ModelConfigurations, Provider
from src.utils.log.telegramlog import telegram_logger
from src.api.llm.gemini_api import GeminiAPI
from src.services.system_message_builder import SystemMessageBuilder


class APIError(Exception):
    """Base exception for API-related errors."""

    pass


class ToolUnsupportedError(APIError):
    """Raised when a model doesn't support tool calling."""

    pass


class StreamingConflictError(APIError):
    """Raised when streaming conflicts with tool usage."""

    pass


class ModelRouterError(APIError):
    """Raised when model routing fails."""

    pass


@dataclass
class ModelCapabilities:
    """Encapsulates model capabilities for better type safety."""

    supports_tools: bool
    supports_streaming: bool
    has_streaming_tool_conflict: bool
    provider: Provider
    tool_format: str
    max_tokens: int


class ErrorType(Enum):
    """Categorizes different types of errors for better handling."""

    TOOL_UNSUPPORTED = "tool_unsupported"
    STREAMING_CONFLICT = "streaming_conflict"
    MODEL_NOT_FOUND = "model_not_found"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    UNKNOWN = "unknown"


class OpenRouterAPIWithMCP(OpenRouterAPI):
    """
    Enhanced OpenRouter API with MCP (Model Context Protocol) integration.
    This class extends the base OpenRouterAPI to automatically load and use
    MCP server tools, converting them to OpenAI-compatible format.
    """

    def __init__(self, rate_limiter, mcp_config_path: str = "mcp.json"):
        """
        Initialize the enhanced OpenRouter API with MCP support and Gemini integration.

        Args:
            rate_limiter: Rate limiter instance
            mcp_config_path: Path to MCP configuration file
        """
        super().__init__(rate_limiter)

        # Core components
        self.mcp_manager = MCPManager(mcp_config_path)
        self.logger = logging.getLogger(__name__)

        # Performance optimizations - use sets for O(1) lookups
        self._tool_unsupported_models: Set[str] = set()
        self._streaming_conflict_models: Set[str] = set()

        # Caching for expensive operations
        self._model_capabilities_cache: Dict[str, ModelCapabilities] = {}

        # Circuit breaker state
        self._circuit_breaker_open = False
        self._circuit_breaker_failure_count = 0
        self._circuit_breaker_last_failure_time = 0

        # Initialize Gemini API with proper error handling
        self._initialize_gemini_api(rate_limiter, mcp_config_path)

    def _initialize_gemini_api(self, rate_limiter, mcp_config_path: str) -> None:
        """Initialize Gemini API with proper error handling."""
        try:
            self.gemini_api = GeminiAPI(rate_limiter, mcp_config_path)
            self.logger.info("Gemini API initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Gemini API: {e}")
            self.gemini_api = None

    @lru_cache(maxsize=128)
    def _get_model_capabilities(self, model: str) -> ModelCapabilities:
        """
        Get model capabilities with caching for performance.

        Args:
            model: Model identifier

        Returns:
            ModelCapabilities object with all relevant information
        """
        if model in self._model_capabilities_cache:
            return self._model_capabilities_cache[model]

        model_config = ModelConfigurations.get_all_models().get(model)
        provider = self._determine_provider(model, model_config)

        capabilities = ModelCapabilities(
            supports_tools=ModelConfigurations.model_supports_tool_calls(model),
            supports_streaming=not ModelConfigurations.model_has_streaming_tool_conflict(
                model
            ),
            has_streaming_tool_conflict=ModelConfigurations.model_has_streaming_tool_conflict(
                model
            ),
            provider=provider,
            tool_format=self._determine_tool_format(model),
            max_tokens=model_config.max_tokens if model_config else 128000,
        )

        self._model_capabilities_cache[model] = capabilities
        return capabilities

    def _determine_provider(self, model: str, model_config=None) -> Provider:
        """Determine provider with fallback logic."""
        if model_config:
            return model_config.provider
        return ModelConfigurations._determine_provider_from_id(model)

    def _determine_tool_format(self, model: str) -> str:
        """
        Determine tool format with optimized string matching.

        Args:
            model: Model identifier

        Returns:
            Tool format ("meta" or "openai")
        """
        # Use a single lower() call and check multiple indicators
        model_lower = model.lower()
        meta_indicators = {"meta-llama", "llama", "meta."}

        if any(indicator in model_lower for indicator in meta_indicators):
            self.logger.debug(f"Using Meta tool format for {model}")
            return "meta"
        return "openai"

    def _should_use_gemini(self, model: str) -> bool:
        """Check if we should use Gemini for the given model."""
        if not self.gemini_api:
            return False
        capabilities = self._get_model_capabilities(model)
        return capabilities.provider == Provider.GEMINI

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self._circuit_breaker_open:
            return False

        # Auto-recovery after timeout
        current_time = asyncio.get_event_loop().time()
        if (
            current_time - self._circuit_breaker_last_failure_time
            > self.circuit_breaker_timeout
        ):
            self._circuit_breaker_open = False
            self._circuit_breaker_failure_count = 0
            self.logger.info("Circuit breaker reset - attempting recovery")
            return False

        return True

    def _handle_circuit_breaker_failure(self) -> None:
        """Handle failure for circuit breaker pattern."""
        self._circuit_breaker_failure_count += 1
        self._circuit_breaker_last_failure_time = asyncio.get_event_loop().time()

        if self._circuit_breaker_failure_count >= self.circuit_breaker_threshold:
            self._circuit_breaker_open = True
            self.logger.warning(
                f"Circuit breaker opened after {self._circuit_breaker_failure_count} failures"
            )

    async def initialize_mcp_tools(self) -> bool:
        """
        Initialize and load MCP tools from configured servers.
        Uses singleton pattern - will skip if already initialized.

        Returns:
            True if MCP tools were loaded successfully
        """
        # Use singleton state - avoid redundant initialization
        if self.mcp_manager._initialized:
            self.logger.debug("MCP tools already initialized (using shared instance)")
            return True

        try:
            self.logger.info("Initializing MCP tools...")
            telegram_logger.log_message("Initializing MCP tools...", 0)

            success = await self.mcp_manager.load_servers()
            if success:
                server_info = self.mcp_manager.get_server_info()
                self.logger.info(f"MCP tools initialized successfully: {server_info}")
                telegram_logger.log_message(
                    f"MCP tools initialized: {len(server_info)} servers", 0
                )
                return True
            else:
                self.logger.warning("Failed to initialize MCP tools")
                telegram_logger.log_message("Failed to initialize MCP tools", 0)
                return False

        except Exception as e:
            self.logger.error(f"Error initializing MCP tools: {str(e)}")
            telegram_logger.log_error(f"Error initializing MCP tools: {str(e)}", 0)
            return False

    async def generate_response_with_mcp_tools(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a response using OpenRouter with MCP tools available.

        Args:
            prompt: The user prompt
            context: Conversation context
            model: Optional model override (if None, uses default)
            temperature: Sampling temperature
            timeout: Request timeout

        Returns:
            Generated response or None if failed
        """
        # Circuit breaker check
        if self._is_circuit_breaker_open():
            return "Service temporarily unavailable. Please try again later."

        try:
            # Route to appropriate provider
            if self._should_use_gemini(model):
                return await self._handle_gemini_request(
                    prompt, context, model, temperature, max_tokens, timeout
                )
            else:
                return await self._handle_openrouter_request(
                    prompt, context, model, temperature, max_tokens, timeout
                )

        except Exception as e:
            self._handle_circuit_breaker_failure()
            self.logger.error(f"Error in generate_response_with_mcp_tools: {e}")
            return self._get_error_response(e)

    async def _handle_gemini_request(
        self,
        prompt: str,
        context: Optional[List[Dict]],
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> Optional[str]:
        """Handle requests routed to Gemini API."""
        self.logger.info(f"Using Gemini API for model {model}")
        actual_model = "gemini-2.5-flash" if model == "gemini" else model

        return await self.gemini_api.generate_response_with_mcp_tools(
            prompt=prompt,
            context=context,
            model=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    async def _handle_openrouter_request(
        self,
        prompt: str,
        context: Optional[List[Dict]],
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> Optional[str]:
        """Handle requests routed to OpenRouter API."""
        self.logger.info(f"Using OpenRouter API for model {model}")

        # Get model capabilities and resolve actual model
        actual_model, capabilities = self._resolve_model_and_capabilities(model)

        # Get MCP tools if supported
        mcp_tools = await self._get_mcp_tools_for_model(capabilities)

        # Check for known conflicts
        if self._has_known_conflicts(actual_model, mcp_tools, capabilities):
            return await self._fallback_to_standard_generation(
                prompt, context, actual_model, temperature, timeout
            )

            # Generate with or without tools
            if mcp_tools:
                self.logger.info(f"Using {len(mcp_tools)} MCP tools for {actual_model}")
                return await self.generate_response_with_tools(
                    prompt=prompt,
                    tools=mcp_tools,
                    context=context,
                    model=actual_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            else:
                self.logger.info(f"No MCP tools available for {actual_model}")
                return await self.generate_response(
                    prompt=prompt,
                    context=context,
                    model=actual_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

    def _resolve_model_and_capabilities(
        self, model: Optional[str]
    ) -> Tuple[str, ModelCapabilities]:
        """Resolve actual model name and get capabilities."""
        actual_model = model if model is not None else "gemini"

        # Handle model routing through configuration
        model_config = ModelConfigurations.get_all_models().get(model)
        if model_config and model_config.openrouter_model_key:
            actual_model = model_config.openrouter_model_key

        capabilities = self._get_model_capabilities(actual_model)
        return actual_model, capabilities

    async def _get_mcp_tools_for_model(
        self, capabilities: ModelCapabilities
    ) -> List[Dict[str, Any]]:
        """Get MCP tools formatted for the specific model."""
        if not self.mcp_manager._initialized:
            return []

        return await self.mcp_manager.get_all_tools(provider=capabilities.tool_format)

    def _has_known_conflicts(
        self,
        model: str,
        mcp_tools: List[Dict[str, Any]],
        capabilities: ModelCapabilities,
    ) -> bool:
        """Check for known conflicts that require fallback."""
        return (
            model in self._tool_unsupported_models
            or (mcp_tools and capabilities.has_streaming_tool_conflict)
            or model in self._streaming_conflict_models
        )

    async def _fallback_to_standard_generation(
        self,
        prompt: str,
        context: Optional[List[Dict]],
        model: str,
        temperature: float,
        timeout: float,
    ) -> Optional[str]:
        """Fallback to standard generation without tools."""
        self.logger.info(
            f"Using standard generation for {model} due to known conflicts"
        )

        return await self.generate_response(
            prompt=prompt,
            context=context,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )

    def _build_system_message(
        self,
        model_id: str,
        context: Optional[List[Dict]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system message with dynamic tool usage instructions.

        Args:
            model_id: Model identifier
            context: Optional conversation context
            tools: Optional tools available to the model

        Returns:
            Complete system message
        """
        # Get base system message from parent class
        base_message = super()._build_system_message(model_id, context, tools)

        # Add tool-specific instructions if tools are provided
        if tools:
            tool_names = [tool["function"]["name"] for tool in tools]
            tool_categories = SystemMessageBuilder.categorize_tools_generic(tools)

            # OpenRouter-specific tool instructions
            provider_specific = """- **Documentation & Code Examples**: Use documentation tools when users ask about libraries, frameworks, APIs, or need code examples
- **Search & Research**: Use search tools for finding information, current data, or web content
- **Analysis & Processing**: Use specialized tools for data analysis, file processing, or complex computations
- **External Services**: Use tools that connect to external services or APIs
"""
            tool_instructions = SystemMessageBuilder.build_tool_instructions(
                tool_names, tool_categories, provider_specific
            )
            return base_message + tool_instructions

        return base_message

    async def generate_response_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a response with tool calling support.

        Args:
            prompt: The user prompt
            tools: List of tools in OpenAI format
            context: Conversation context
            model: Optional model to use (if None, uses default)
            temperature: Sampling temperature
            timeout: Request timeout

        Returns:
            Generated response or None if failed
        """
        try:
            # Prepare request parameters
            openrouter_model = self.available_models.get(model, model)
            system_message = self._build_system_message(model, context, tools)
            messages = self._build_messages(system_message, context, prompt, model)

            # Execute tool calling workflow
            response = await self._execute_tool_calling_workflow(
                openrouter_model, messages, tools, temperature, max_tokens, timeout
            )

            return self._clean_response_content(response) if response else None

        except Exception as e:
            return await self._handle_tool_calling_error(
                e, model, prompt, context, temperature, max_tokens, timeout
            )

    def _build_messages(
        self,
        system_message: str,
        context: Optional[List[Dict]],
        prompt: str,
        model: Optional[str],
    ) -> List[Dict]:
        """Build messages array for the request."""
        messages = []

        # Use the new capability detection system
        safe_config = ModelConfigurations.get_safe_model_config(model or "")
        if safe_config["use_system_message"]:
            messages.append({"role": "system", "content": system_message})

        if context:
            messages.extend(context)

        messages.append({"role": "user", "content": prompt})
        return messages

    async def _execute_tool_calling_workflow(
        self,
        openrouter_model: str,
        messages: List[Dict],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> Optional[str]:
        """Execute the main tool calling workflow."""
        # Initial request with tools
        request_params = self._build_request_params(
            openrouter_model, messages, tools, temperature, max_tokens, timeout
        )
        adapted_params = ModelConfigurations.adapt_request_for_model(
            openrouter_model, request_params
        )

        response = await self.client.chat.completions.create(**adapted_params)

        # Handle tool calls if present
        if self._has_tool_calls(response):
            return await self._process_tool_calls(
                response, messages, openrouter_model, temperature, max_tokens, timeout
            )
        else:
            return response.choices[0].message.content

    def _build_request_params(
        self,
        model: str,
        messages: List[Dict],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> Dict[str, Any]:
        """Build request parameters dictionary."""
        params = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
            "timeout": timeout,
            "stream": False,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        return params

    def _has_tool_calls(self, response) -> bool:
        """Check if response contains tool calls."""
        return (
            hasattr(response.choices[0].message, "tool_calls")
            and response.choices[0].message.tool_calls
        )

    async def _process_tool_calls(
        self,
        response,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> Optional[str]:
        """Process tool calls and generate final response."""
        tool_calls = response.choices[0].message.tool_calls
        messages.append(response.choices[0].message.model_dump())

        # Execute all tool calls
        for tool_call in tool_calls:
            await self._execute_single_tool_call(tool_call, messages)

            # Add small delay for multiple tools to prevent overwhelming
            if len(tool_calls) > 1:
                await asyncio.sleep(0.5)

        # Generate final response
        return await self._generate_final_response(
            messages, model, temperature, max_tokens, timeout
        )

    async def _execute_single_tool_call(self, tool_call, messages: List[Dict]) -> None:
        """Execute a single tool call and add result to messages."""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        try:
            # Parse arguments if they're a string
            if isinstance(tool_args, str):
                import json

                tool_args = json.loads(tool_args)

            self.logger.info(f"Executing MCP tool: {tool_name} with args: {tool_args}")

            tool_result = await self.mcp_manager.execute_tool(tool_name, tool_args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result),
                }
            )

            self.logger.info(f"Successfully executed MCP tool: {tool_name}")

        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": f"Error executing tool: {str(e)}",
                }
            )

    async def _generate_final_response(
        self,
        messages: List[Dict],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> Optional[str]:
        """Generate the final response after tool execution."""
        final_request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
            "stream": False,
        }
        if max_tokens is not None:
            final_request_params["max_tokens"] = max_tokens

        final_adapted_params = ModelConfigurations.adapt_request_for_model(
            model, final_request_params
        )
        final_response = await self.client.chat.completions.create(
            **final_adapted_params
        )

        return final_response.choices[0].message.content

    async def _handle_tool_calling_error(
        self,
        error: Exception,
        model: Optional[str],
        prompt: str,
        context: Optional[List[Dict]],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> Optional[str]:
        """Handle errors in tool calling with intelligent fallback."""
        error_str = str(error)
        error_type = self._categorize_error(error_str)

        # Log specific error information
        self._log_categorized_error(error_type, model, error_str)

        # Attempt fallback for recoverable errors
        if self._should_attempt_fallback(error_type):
            return await self._attempt_fallback_generation(
                error_type, model, prompt, context, temperature, max_tokens, timeout
            )

        return self._get_error_response(error)

    def _categorize_error(self, error_str: str) -> ErrorType:
        """Categorize error for better handling."""
        error_lower = error_str.lower()

        if any(
            phrase in error_lower
            for phrase in [
                "no endpoints found that support tool use",
                "does not support tool calling",
                "tool use not supported",
                "tool calling not supported",
            ]
        ):
            return ErrorType.TOOL_UNSUPPORTED

        if "tools are not supported in streaming mode" in error_lower:
            return ErrorType.STREAMING_CONFLICT

        if "not a valid model id" in error_lower:
            return ErrorType.MODEL_NOT_FOUND

        if "authentication" in error_lower:
            return ErrorType.AUTHENTICATION

        if "rate limit" in error_lower:
            return ErrorType.RATE_LIMIT

        return ErrorType.UNKNOWN

    def _log_categorized_error(
        self, error_type: ErrorType, model: Optional[str], error_str: str
    ) -> None:
        """Log errors with appropriate detail level based on type."""
        if error_type == ErrorType.TOOL_UNSUPPORTED:
            self.logger.warning(f"Tool calling not supported for model '{model}'")
        elif error_type == ErrorType.STREAMING_CONFLICT:
            self.logger.warning(f"Streaming mode conflict for model '{model}'")
        elif error_type == ErrorType.MODEL_NOT_FOUND:
            self.logger.warning(f"Model '{model}' not found")
        else:
            self.logger.error(f"Error in tool calling for model '{model}': {error_str}")

    def _should_attempt_fallback(self, error_type: ErrorType) -> bool:
        """Determine if fallback should be attempted for this error type."""
        return error_type in {
            ErrorType.TOOL_UNSUPPORTED,
            ErrorType.STREAMING_CONFLICT,
            ErrorType.UNKNOWN,
        }

    async def _attempt_fallback_generation(
        self,
        error_type: ErrorType,
        model: Optional[str],
        prompt: str,
        context: Optional[List[Dict]],
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
    ) -> Optional[str]:
        """Attempt fallback generation and cache the model's limitations."""
        openrouter_model = self.available_models.get(model, model)

        # Cache the model's limitations for future requests
        if error_type == ErrorType.TOOL_UNSUPPORTED:
            self._tool_unsupported_models.add(openrouter_model)
        elif error_type == ErrorType.STREAMING_CONFLICT:
            self._streaming_conflict_models.add(openrouter_model)

        telegram_logger.log_message(f"Falling back to non-tool mode for {model}", 0)

        try:
            fallback_response = await self.generate_response(
                prompt=prompt,
                context=context,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            if fallback_response:
                self.logger.info(
                    f"Successfully generated fallback response for model {openrouter_model}"
                )
                return fallback_response
            else:
                self.logger.error(
                    f"Fallback generation also failed for model {openrouter_model}"
                )
                return "I apologize, but I'm having trouble processing your request right now. Please try again or use a different model."

        except Exception as fallback_error:
            self.logger.error(f"Fallback generation failed: {str(fallback_error)}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or use a different model."

    def _get_error_response(self, error: Exception) -> str:
        """Get appropriate error response based on error type."""
        error_str = str(error)

        if "overloaded" in error_str.lower():
            return "The API is currently overloaded. Please try again in a moment."
        elif "rate limit" in error_str.lower():
            return "Rate limit exceeded. Please wait a moment before trying again."
        elif "authentication" in error_str.lower():
            return "Authentication error. Please check your API configuration."
        else:
            return "I encountered an error while processing your request. Please try again or use a different model."

    def _clean_response_content(self, content: str) -> str:
        """
        Clean response content by removing thinking tags and tool calls.

        Args:
            content: Raw response content from the model

        Returns:
            Cleaned content suitable for user display
        """
        if not content:
            return content

        import re

        # Remove various thinking and tool call patterns
        patterns_to_remove = [
            r"<think>.*?</think>",
            r"<tool_call>.*?</tool_call>",
            r"<[^>]+>.*?</[^>]+>",
        ]

        for pattern in patterns_to_remove:
            content = re.sub(pattern, "", content, flags=re.DOTALL)

        # Clean up whitespace
        content = content.strip()
        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

        return content

    async def get_available_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available MCP tools in OpenAI format.

        Returns:
            List of available MCP tools
        """
        if self.mcp_manager._initialized:
            return await self.mcp_manager.get_all_tools()
        else:
            return []

    def get_mcp_server_info(self) -> Dict[str, Any]:
        """
        Get information about connected MCP servers.

        Returns:
            Dictionary with server information
        """
        if self.mcp_manager._initialized:
            return self.mcp_manager.get_server_info()
        else:
            return {}

    async def close(self):
        """Close the API client and MCP connections."""
        await super().close()

        if self.mcp_manager._initialized:
            await self.mcp_manager.disconnect_all()

        if self.gemini_api:
            await self.gemini_api.close()

        # Clear caches
        self._model_capabilities_cache.clear()
        self._get_model_capabilities.cache_clear()
