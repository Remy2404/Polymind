import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool
from src.utils.log.telegramlog import telegram_logger
from .tool_converter import MCPToolConverter


class MCPServerClient:
    """Client for connecting to and managing MCP servers with enhanced capabilities."""

    def __init__(self, server_config: Dict[str, Any], server_name: str = "unknown"):
        """
        Initialize MCP server client with enhanced capabilities.
        Args:
            server_config: MCP server configuration from mcp.json
            server_name: Name of the MCP server
        """
        self.server_config = server_config
        self.server_name = server_name
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.logger = logging.getLogger(__name__)
        self.available_tools: List[MCPTool] = []
        self.openai_tools: List[Dict[str, Any]] = []
        self.meta_tools: List[Dict[str, Any]] = []
        self._connection_task = None
        self._cleanup_done = False

        # Enhanced capabilities
        self._progress_callbacks: List[Callable[[str, float, str], None]] = []
        self._context_providers: List[Callable[[], Dict[str, Any]]] = []

        # Circuit breaker for reliability
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0
        self._circuit_breaker_timeout = 60.0  # 1 minute circuit breaker
        self._max_consecutive_failures = 5

        # Adaptive timeouts based on operation type
        self._timeout_multipliers = {
            "search": 2.0,
            "fetch": 1.5,
            "process": 3.0,
            "analyze": 2.0,
            "default": 1.0,
        }

    def add_progress_callback(self, callback: Callable[[str, float, str], None]):
        """
        Add a progress callback function.
        Args:
            callback: Function called with (operation, progress, message)
        """
        self._progress_callbacks.append(callback)

    def add_context_provider(self, provider: Callable[[], Dict[str, Any]]):
        """
        Add a context provider function.
        Args:
            provider: Function that returns context dictionary
        """
        self._context_providers.append(provider)

    def _report_progress(self, operation: str, progress: float, message: str = ""):
        """Report progress to all registered callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(operation, progress, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def _get_context(self) -> Dict[str, Any]:
        """Get combined context from all providers."""
        context = {}
        for provider in self._context_providers:
            try:
                provider_context = provider()
                context.update(provider_context)
            except Exception as e:
                self.logger.warning(f"Context provider failed: {e}")
        return context

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open (failing)."""
        if self._circuit_breaker_failures >= self._max_consecutive_failures:
            current_time = asyncio.get_event_loop().time()
            if (
                current_time - self._circuit_breaker_last_failure
                < self._circuit_breaker_timeout
            ):
                return True
            else:
                # Reset circuit breaker after timeout
                self._circuit_breaker_failures = 0
                self.logger.info(f"Circuit breaker reset for server {self.server_name}")
        return False

    def _record_failure(self):
        """Record a failure for circuit breaker."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = asyncio.get_event_loop().time()

    def _record_success(self):
        """Record a success to reset circuit breaker."""
        self._circuit_breaker_failures = 0

    def _get_adaptive_timeout(self, tool_name: str) -> float:
        """Get adaptive timeout based on tool type."""
        base_timeout = 60.0 if os.getenv("INSIDE_DOCKER") else 30.0

        # Determine operation type from tool name
        tool_lower = tool_name.lower()
        multiplier = self._timeout_multipliers["default"]

        for op_type, mult in self._timeout_multipliers.items():
            if op_type in tool_lower:
                multiplier = mult
                break

        return base_timeout * multiplier

    async def connect(self) -> bool:
        """
        Connect to the MCP server and retrieve available tools.
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.server_config.get("type") == "stdio":
                primary_success = await self._try_connect_with_config(
                    self.server_config
                )
                if primary_success:
                    return True
                fallback_config = self.server_config.get("fallback")
                if fallback_config:
                    self.logger.info(
                        f"Trying fallback command for MCP server '{self.server_name}'"
                    )
                    fallback_success = await self._try_connect_with_config(
                        fallback_config
                    )
                    if fallback_success:
                        return True
                return False
            elif self.server_config.get("type") == "sse":
                self.logger.warning("SSE connections not yet implemented")
                return False
            elif self.server_config.get("type") == "http":
                self.logger.warning("HTTP connections not yet implemented")
                return False
            else:
                self.logger.error(
                    f"Unsupported server type: {self.server_config.get('type')}"
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Failed to connect to MCP server '{self.server_name}': {str(e)}"
            )
            telegram_logger.log_error(
                f"Failed to connect to MCP server '{self.server_name}': {str(e)}", 0
            )
            return False

    async def _try_connect_with_config(self, config: Dict[str, Any]) -> bool:
        """
        Try to connect using a specific configuration.
        Args:
            config: Server configuration to try
        Returns:
            True if connection successful
        """
        try:
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env={**os.environ, **config.get("env", {})},
            )
            try:
                conn_timeout = 30.0 if os.getenv("INSIDE_DOCKER") else 15.0
                async with asyncio.timeout(conn_timeout):
                    async with stdio_client(server_params) as (stdio, write):
                        self.stdio, self.write = stdio, write
                        async with ClientSession(self.stdio, self.write) as session:
                            self.session = session
                            try:
                                init_timeout = (
                                    15.0 if os.getenv("INSIDE_DOCKER") else 10.0
                                )
                                async with asyncio.timeout(init_timeout):
                                    await session.initialize()
                            except asyncio.TimeoutError:
                                timeout_msg = f"Timeout initializing MCP server session '{self.server_name}' after {init_timeout} seconds"
                                self.logger.warning(timeout_msg)
                                telegram_logger.log_error(timeout_msg, 0)
                                return False
                            try:
                                list_timeout = (
                                    10.0 if os.getenv("INSIDE_DOCKER") else 5.0
                                )
                                async with asyncio.timeout(list_timeout):
                                    response = await session.list_tools()
                                    self.available_tools = response.tools
                            except asyncio.TimeoutError:
                                timeout_msg = f"Timeout listing tools from MCP server '{self.server_name}' after {list_timeout} seconds"
                                self.logger.warning(timeout_msg)
                                telegram_logger.log_error(timeout_msg, 0)
                                return False
                            self.openai_tools = (
                                MCPToolConverter.convert_mcp_tools_to_openai(
                                    self.available_tools
                                )
                            )
                            self.meta_tools = (
                                MCPToolConverter.convert_mcp_tools_to_meta(
                                    self.available_tools
                                )
                            )
                            self.logger.info(
                                f"Connected to MCP server '{self.server_name}' with {len(self.available_tools)} tools"
                            )
                            telegram_logger.log_message(
                                f"Connected to MCP server '{self.server_name}' with {len(self.available_tools)} tools",
                                0,
                            )
                            self._server_params = server_params
                            return True
            except asyncio.TimeoutError:
                timeout_msg = f"Timeout connecting to MCP server '{self.server_name}' after {conn_timeout} seconds"
                self.logger.warning(timeout_msg)
                telegram_logger.log_error(timeout_msg, 0)
                return False
            except Exception as conn_error:
                self.logger.error(
                    f"Error establishing stdio connection to '{self.server_name}': {str(conn_error)}"
                )
                return False
        except Exception as e:
            self.logger.error(
                f"Failed to connect to MCP server '{self.server_name}': {str(e)}"
            )
            telegram_logger.log_error(
                f"Failed to connect to MCP server '{self.server_name}': {str(e)}", 0
            )
            return False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server with enhanced capabilities.
        Implements retry mechanism with exponential backoff for reliability.
        Supports progress reporting and context injection for long-running operations.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
        Returns:
            Tool execution result
        """
        max_retries = 3
        retry_delays = [1.0, 2.0, 4.0]  # Exponential backoff

        # Inject context if providers are available
        enhanced_args = dict(arguments)
        context = self._get_context()
        if context:
            enhanced_args["_context"] = context

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    self.logger.info(
                        f"Retry attempt {attempt + 1}/{max_retries} for tool '{tool_name}'"
                    )
                    self._report_progress(
                        f"retry_{tool_name}",
                        attempt / max_retries,
                        f"Retrying tool execution (attempt {attempt + 1})",
                    )

                # Check circuit breaker
                if self._is_circuit_breaker_open():
                    raise RuntimeError(
                        f"Circuit breaker is open for server '{self.server_name}' due to consecutive failures"
                    )

                # Get adaptive timeout for this tool
                adaptive_timeout = self._get_adaptive_timeout(tool_name)

                # Report initial progress
                self._report_progress(tool_name, 0.0, f"Starting {tool_name} execution")

                result = await self._call_tool_with_config(
                    self.server_config, tool_name, enhanced_args, adaptive_timeout
                )

                # Record success for circuit breaker
                self._record_success()

                # Report completion
                self._report_progress(
                    tool_name, 1.0, f"Completed {tool_name} execution"
                )

                return result

            except Exception as primary_error:
                self.logger.warning(
                    f"Primary tool call failed for '{tool_name}': {str(primary_error)}"
                )

                # Report error progress
                self._report_progress(
                    tool_name, -1.0, f"Primary execution failed: {str(primary_error)}"
                )

                fallback_config = self.server_config.get("fallback")
                if fallback_config:
                    self.logger.info(
                        f"Trying fallback configuration for tool '{tool_name}'"
                    )
                    self._report_progress(
                        tool_name, 0.5, "Trying fallback configuration"
                    )
                    try:
                        result = await self._call_tool_with_config(
                            fallback_config, tool_name, enhanced_args
                        )
                        self._report_progress(
                            tool_name, 1.0, "Fallback execution successful"
                        )
                        return result
                    except Exception as fallback_error:
                        self.logger.error(
                            f"Fallback tool call also failed for '{tool_name}': {str(fallback_error)}"
                        )
                        # Continue to retry logic below

                # Retry logic for both primary and fallback failures
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    self.logger.info(f"Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                    continue

                # Final failure - record and raise
                error_msg = f"Tool '{tool_name}' failed after {max_retries} attempts"
                if fallback_config:
                    error_msg += " (including fallback)"
                error_msg += f": {str(primary_error)}"

                self.logger.error(error_msg)
                self._record_failure()
                raise RuntimeError(error_msg) from primary_error

    async def _call_tool_with_config(
        self,
        config: Dict[str, Any],
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Call a tool using a specific configuration.
        Args:
            config: Server configuration to use
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
        Returns:
            Tool execution result
        """
        if config.get("type") == "stdio":
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env={**os.environ, **config.get("env", {})},
            )
            try:
                # Extended timeouts for document operations (Word, PDF, etc.)
                # which require more processing time for large content
                connection_timeout = 180.0
                init_timeout = 60.0
                execution_timeout = timeout or 120.0
                async with asyncio.timeout(connection_timeout):
                    async with stdio_client(server_params) as (stdio, write):
                        async with ClientSession(stdio, write) as session:
                            try:
                                async with asyncio.timeout(init_timeout):
                                    await session.initialize()
                            except asyncio.TimeoutError:
                                raise RuntimeError(
                                    f"Timeout initializing session for tool call '{tool_name}' after {init_timeout}s"
                                )
                            try:
                                async with asyncio.timeout(execution_timeout):
                                    result = await session.call_tool(
                                        tool_name, arguments
                                    )
                            except asyncio.TimeoutError:
                                raise RuntimeError(
                                    f"Timeout executing tool '{tool_name}' after {execution_timeout}s"
                                )
                            self.logger.info(
                                f"Tool '{tool_name}' executed successfully"
                            )
                            return result.content
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Timeout during tool call '{tool_name}' after {connection_timeout}s"
                )
            except Exception as conn_error:
                raise RuntimeError(
                    f"Connection error during tool call '{tool_name}': {str(conn_error)}"
                )
        else:
            raise RuntimeError(
                f"Unsupported server type for tool calls: {config.get('type')}"
            )

    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            if hasattr(self, "session") and self.session:
                try:
                    await self.session.__aexit__(None, None, None)
                except Exception:
                    pass
            if hasattr(self, "stdio") and self.stdio:
                try:
                    await self.stdio.__aexit__(None, None, None)
                except Exception:
                    pass
            self.session = None
            self.stdio = None
            self.write = None
            self.available_tools = []
            self.openai_tools = []
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {str(e)}")
