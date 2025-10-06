"""
MCP (Model Context Protocol) Integration for AI Agent Tool Calling
This module provides automatic MCP server integration with OpenRouter,
enabling LLMs to use MCP tools without hardcoded definitions.
"""
import json
import logging
import os
import asyncio
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as MCPTool
from src.utils.log.telegramlog import telegram_logger
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substitute environment variables in MCP configuration.
    Handles $VAR_NAME syntax in configuration values.
    Args:
        config: MCP configuration dictionary
    Returns:
        Configuration with environment variables substituted
    """
    def substitute_value(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("$"):
                env_var = value[1:]
                env_value = os.getenv(env_var)
                if env_value is None:
                    logging.warning(
                        f"Environment variable '{env_var}' not found, using original value"
                    )
                    return value
                return env_value
            return value
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value
    return substitute_value(config)
def validate_mcp_environment() -> bool:
    """
    Validate that required MCP environment variables are available.
    Returns:
        True if all required variables are present
    """
    required_vars = ["MCP_API_KEY"]
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    if missing_vars:
        logging.error(
            f"Missing required MCP environment variables: {', '.join(missing_vars)}"
        )
        telegram_logger.log_error(
            f"Missing required MCP environment variables: {', '.join(missing_vars)}", 0
        )
        return False
    logging.info("All required MCP environment variables are present")
    return True
class MCPToolConverter:
    """Converts MCP tool definitions to OpenAI-compatible format with structured schemas."""
    
    @staticmethod
    def convert_mcp_tool_to_openai(mcp_tool: MCPTool) -> Dict[str, Any]:
        """
        Convert an MCP tool definition to OpenAI tool format with enhanced schemas.
        Args:
            mcp_tool: MCP tool definition
        Returns:
            OpenAI-compatible tool definition with structured schemas
        """
        openai_tool = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description or f"Execute {mcp_tool.name} tool",
                "parameters": MCPToolConverter._convert_input_schema(mcp_tool.inputSchema),
            },
        }
        
        # Add output schema if available for better validation
        if hasattr(mcp_tool, "outputSchema") and mcp_tool.outputSchema:
            openai_tool["function"]["output_schema"] = MCPToolConverter._convert_output_schema(mcp_tool.outputSchema)
        
        # Add additional metadata for robustness
        openai_tool["function"]["strict"] = True  # Enable strict mode for better validation
        
        return openai_tool
    
    @staticmethod
    def _convert_input_schema(input_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert MCP input schema to OpenAI format with validation.
        """
        if not input_schema:
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        # Ensure required fields
        schema = {
            "type": input_schema.get("type", "object"),
            "properties": input_schema.get("properties", {}),
            "required": input_schema.get("required", [])
        }
        
        return schema
    
    @staticmethod
    def _convert_output_schema(output_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert MCP output schema to structured format.
        """
        if not output_schema:
            return {
                "type": "object",
                "properties": {
                    "result": {"type": "string", "description": "Tool execution result"},
                    "success": {"type": "boolean", "description": "Whether the operation succeeded"}
                }
            }
        
        # Ensure the output schema is well-structured
        schema = dict(output_schema)
        
        # Add standard fields for consistency
        if schema.get("type") == "object" and "properties" in schema:
            properties = schema["properties"]
            # Ensure success indicator
            if "success" not in properties:
                properties["success"] = {"type": "boolean", "description": "Operation success status"}
            # Ensure error field for failed operations
            if "error" not in properties:
                properties["error"] = {"type": "string", "description": "Error message if operation failed"}
        
        return schema
    @staticmethod
    def convert_mcp_tools_to_openai(mcp_tools: List[MCPTool]) -> List[Dict[str, Any]]:
        """
        Convert multiple MCP tools to OpenAI format.
        Args:
            mcp_tools: List of MCP tool definitions
        Returns:
            List of OpenAI-compatible tool definitions
        """
        return [MCPToolConverter.convert_mcp_tool_to_openai(tool) for tool in mcp_tools]
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
            "search": 2.0,  # Search operations take longer
            "fetch": 1.5,   # Network fetch operations
            "process": 3.0, # Document processing
            "analyze": 2.0, # Analysis operations
            "default": 1.0
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
            if current_time - self._circuit_breaker_last_failure < self._circuit_breaker_timeout:
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
                conn_timeout = 120.0 if os.getenv("INSIDE_DOCKER") else 60.0
                async with asyncio.timeout(conn_timeout):
                    async with stdio_client(server_params) as (stdio, write):
                        self.stdio, self.write = stdio, write
                        async with ClientSession(self.stdio, self.write) as session:
                            self.session = session
                            try:
                                # Extended initialization timeout for heavy servers like office-word-mcp-server
                                init_timeout = (
                                    90.0 if os.getenv("INSIDE_DOCKER") else 45.0
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
                                    60.0 if os.getenv("INSIDE_DOCKER") else 30.0
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
                        f"Retrying tool execution (attempt {attempt + 1})"
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
                self._report_progress(tool_name, 1.0, f"Completed {tool_name} execution")
                        
                return result
                        
            except Exception as primary_error:
                self.logger.warning(
                    f"Primary tool call failed for '{tool_name}': {str(primary_error)}"
                )
                        
                # Report error progress
                self._report_progress(
                    tool_name, 
                    -1.0, 
                    f"Primary execution failed: {str(primary_error)}"
                )
                        
                fallback_config = self.server_config.get("fallback")
                if fallback_config:
                    self.logger.info(
                        f"Trying fallback configuration for tool '{tool_name}'"
                    )
                    self._report_progress(
                        tool_name, 
                        0.5, 
                        "Trying fallback configuration"
                    )
                    try:
                        result = await self._call_tool_with_config(
                            fallback_config, tool_name, enhanced_args
                        )
                        self._report_progress(
                            tool_name, 
                            1.0, 
                            "Fallback execution successful"
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
        self, config: Dict[str, Any], tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
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
class MCPManager:
    """Manages multiple MCP server connections and tool aggregation."""
    def __init__(self, mcp_config_path: str = "mcp.json"):
        """
        Initialize MCP manager.
        Args:
            mcp_config_path: Path to MCP configuration file
        """
        self.mcp_config_path = Path(mcp_config_path)
        self.servers: Dict[str, MCPServerClient] = {}
        self.logger = logging.getLogger(__name__)
        self.all_openai_tools: List[Dict[str, Any]] = []
        self.tool_to_server_map: Dict[str, str] = {}
    async def load_servers(self) -> bool:
        """
        Load and connect to all MCP servers from configuration.
        Returns:
            True if any servers connected successfully
        """
        if not validate_mcp_environment():
            self.logger.error("MCP environment validation failed")
            return False
        if not self.mcp_config_path.exists():
            self.logger.warning(f"MCP config file not found: {self.mcp_config_path}")
            return False
        try:
            with open(self.mcp_config_path, "r") as f:
                config = json.load(f)
            config = substitute_env_vars(config)
            self.logger.info("Environment variables substituted in MCP configuration")
            telegram_logger.log_message(
                "Environment variables substituted in MCP configuration", 0
            )
            servers_config = config.get("servers", {})
            connection_tasks = []
            for server_name, server_config in servers_config.items():
                task = asyncio.create_task(
                    self._connect_single_server(server_name, server_config),
                    name=f"mcp_connect_{server_name}",
                )
                connection_tasks.append(task)
            if connection_tasks:
                try:
                    global_timeout = 300.0 if os.getenv("INSIDE_DOCKER") else 120.0
                    done, pending = await asyncio.wait(
                        connection_tasks,
                        timeout=global_timeout,
                        return_when=asyncio.ALL_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    successful_connections = 0
                    for i, task in enumerate(done):
                        server_name = list(servers_config.keys())[i]
                        try:
                            result = await task
                            if result is True:
                                successful_connections += 1
                        except Exception as e:
                            self.logger.error(
                                f"Failed to connect to MCP server '{server_name}': {str(e)}"
                            )
                    if successful_connections > 0:
                        self.logger.info(
                            f"Successfully connected to {successful_connections}/{len(connection_tasks)} MCP servers"
                        )
                        telegram_logger.log_message(
                            f"Successfully connected to {successful_connections}/{len(connection_tasks)} MCP servers",
                            0,
                        )
                        return True
                    else:
                        self.logger.warning("No MCP servers connected successfully")
                        return False
                except Exception as e:
                    self.logger.error(f"Error during MCP server connections: {str(e)}")
                    return False
            else:
                self.logger.warning("No MCP servers configured")
                return False
        except Exception as e:
            self.logger.error(f"Error loading MCP servers: {str(e)}")
            telegram_logger.log_error(f"Error loading MCP servers: {str(e)}", 0)
            return False
    async def _connect_single_server(
        self, server_name: str, server_config: Dict[str, Any]
    ) -> bool:
        """
        Connect to a single MCP server with timeout.
        Args:
            server_name: Name of the server
            server_config: Server configuration
        Returns:
            True if connection successful
        """
        try:
            self.logger.info(f"Connecting to MCP server: {server_name}")
            if isinstance(server_config, dict) and "type" in server_config:
                client = MCPServerClient(server_config, server_name)
            else:
                client = MCPServerClient({server_name: server_config}, server_name)
            try:
                server_timeout = 120.0 if os.getenv("INSIDE_DOCKER") else 60.0
                async with asyncio.timeout(server_timeout):
                    success = await client.connect()
            except asyncio.TimeoutError:
                timeout_msg = f"Timeout connecting to MCP server '{server_name}' after {server_timeout} seconds"
                self.logger.warning(timeout_msg)
                telegram_logger.log_error(timeout_msg, 0)
                return False
            except Exception as conn_error:
                self.logger.error(
                    f"Error connecting to MCP server '{server_name}': {str(conn_error)}"
                )
                return False
            if success:
                self.servers[server_name] = client
                for tool in client.openai_tools:
                    tool_name = tool["function"]["name"]
                    self.tool_to_server_map[tool_name] = server_name
                self.all_openai_tools.extend(client.openai_tools)
                return True
            else:
                return False
        except Exception as e:
            self.logger.error(
                f"Error connecting to MCP server '{server_name}': {str(e)}"
            )
            return False
    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools from connected MCP servers in OpenAI format.
        Returns:
            List of OpenAI-compatible tool definitions
        """
        return self.all_openai_tools.copy()
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with robust error handling and fallback mechanisms.
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
        Returns:
            Structured response with consistent format for success and error cases
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            if tool_name not in self.tool_to_server_map:
                return self._create_error_response(
                    tool_name, 
                    "TOOL_NOT_FOUND", 
                    f"Tool '{tool_name}' not found in any connected MCP server",
                    arguments
                )
            
            server_name = self.tool_to_server_map[tool_name]
            server = self.servers[server_name]
            
            # Execute the tool with timeout and error handling
            result = await server.call_tool(tool_name, arguments)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Return structured success response
            return {
                "success": True,
                "tool_name": tool_name,
                "server": server_name,
                "result": result,
                "execution_time": round(execution_time, 3),
                "status": "completed"
            }
            
        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            return self._create_error_response(
                tool_name,
                "TIMEOUT",
                f"Tool execution timed out after {execution_time:.1f}s",
                arguments,
                execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_type = self._classify_error(e)
            
            # Try fallback if available and this is a recoverable error
            if error_type in ["CONNECTION_ERROR", "SERVER_ERROR", "TIMEOUT"] and self._has_fallback_available(tool_name):
                try:
                    self.logger.info(f"Attempting fallback execution for tool '{tool_name}'")
                    fallback_result = await self._execute_tool_with_fallback(tool_name, arguments)
                    if fallback_result:
                        fallback_result["used_fallback"] = True
                        return fallback_result
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback execution also failed for '{tool_name}': {str(fallback_error)}")
            
            return self._create_error_response(
                tool_name,
                error_type,
                str(e),
                arguments,
                execution_time
            )
    
    def _create_error_response(
        self, 
        tool_name: str, 
        error_type: str, 
        error_message: str, 
        arguments: Dict[str, Any],
        execution_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        Create a consistent error response structure.
        """
        return {
            "success": False,
            "tool_name": tool_name,
            "error_type": error_type,
            "error_message": error_message,
            "arguments": arguments,
            "execution_time": round(execution_time, 3),
            "status": "failed",
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def _classify_error(self, error: Exception) -> str:
        """
        Classify the type of error for better handling.
        """
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return "TIMEOUT"
        elif "connection" in error_str or "connect" in error_str:
            return "CONNECTION_ERROR"
        elif "not found" in error_str or "404" in error_str:
            return "NOT_FOUND"
        elif "unauthorized" in error_str or "403" in error_str or "401" in error_str:
            return "AUTHENTICATION_ERROR"
        elif "rate limit" in error_str or "429" in error_str:
            return "RATE_LIMIT_ERROR"
        elif "server" in error_str or "500" in error_str:
            return "SERVER_ERROR"
        elif "validation" in error_str or "invalid" in error_str:
            return "VALIDATION_ERROR"
        else:
            return "UNKNOWN_ERROR"
    
    def _has_fallback_available(self, tool_name: str) -> bool:
        """
        Check if a fallback is available for the given tool.
        """
        if tool_name not in self.tool_to_server_map:
            return False
        
        server_name = self.tool_to_server_map[tool_name]
        server = self.servers.get(server_name)
        if not server:
            return False
        
        # Check if server config has fallback
        return hasattr(server, 'server_config') and 'fallback' in server.server_config
    
    async def _execute_tool_with_fallback(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a tool using fallback configuration.
        """
        if tool_name not in self.tool_to_server_map:
            return None
        
        server_name = self.tool_to_server_map[tool_name]
        server = self.servers.get(server_name)
        if not server or not hasattr(server, 'server_config'):
            return None
        
        fallback_config = server.server_config.get('fallback')
        if not fallback_config:
            return None
        
        try:
            # Create a temporary client for fallback execution
            temp_client = MCPServerClient(fallback_config, f"{server_name}_fallback")
            
            # Try to connect and execute with shorter timeout for fallback
            connect_timeout = 30.0 if os.getenv("INSIDE_DOCKER") else 15.0
            async with asyncio.timeout(connect_timeout):
                if await temp_client.connect():
                    # Get adaptive timeout for fallback
                    adaptive_timeout = temp_client._get_adaptive_timeout(tool_name)
                    result = await temp_client._call_tool_with_config(
                        fallback_config, tool_name, arguments, adaptive_timeout
                    )
                    await temp_client.disconnect()
                    
                    return {
                        "success": True,
                        "tool_name": tool_name,
                        "server": f"{server_name}_fallback",
                        "result": result,
                        "execution_time": 0.0,  # Would need to track this properly
                        "status": "completed"
                    }
            
            await temp_client.disconnect()
            return None
            
        except Exception as e:
            self.logger.warning(f"Fallback execution failed: {str(e)}")
            return None
    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        disconnect_tasks = []
        for server in self.servers.values():
            task = asyncio.create_task(server.disconnect())
            disconnect_tasks.append(task)
        if disconnect_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*disconnect_tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout during server disconnection")
            except Exception as e:
                self.logger.error(f"Error during server disconnection: {str(e)}")
        self.servers.clear()
        self.all_openai_tools.clear()
        self.tool_to_server_map.clear()
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about connected servers and their tools.
        Returns:
            Dictionary with server and tool information
        """
        info = {}
        for server_name, server in self.servers.items():
            info[server_name] = {
                "tool_count": len(server.available_tools),
                "tools": [tool.name for tool in server.available_tools],
            }
        return info
