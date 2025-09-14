"""
MCP (Model Context Protocol) Integration for AI Agent Tool Calling
This module provides automatic MCP server integration with OpenRouter,
enabling LLMs to use MCP tools without hardcoded definitions.
"""
import json
import logging
import os
import asyncio
from typing import Dict, List, Optional, Any
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
    """Converts MCP tool definitions to OpenAI-compatible format."""
    @staticmethod
    def convert_mcp_tool_to_openai(mcp_tool: MCPTool) -> Dict[str, Any]:
        """
        Convert an MCP tool definition to OpenAI tool format.
        Args:
            mcp_tool: MCP tool definition
        Returns:
            OpenAI-compatible tool definition
        """
        openai_tool = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
            if "properties" in mcp_tool.inputSchema:
                openai_tool["function"]["parameters"]["properties"] = (
                    mcp_tool.inputSchema["properties"]
                )
            if "required" in mcp_tool.inputSchema:
                openai_tool["function"]["parameters"]["required"] = (
                    mcp_tool.inputSchema["required"]
                )
        return openai_tool
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
    """Client for connecting to and managing MCP servers."""
    def __init__(self, server_config: Dict[str, Any], server_name: str = "unknown"):
        """
        Initialize MCP server client.
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
                conn_timeout = 60.0 if os.getenv("INSIDE_DOCKER") else 30.0
                async with asyncio.timeout(conn_timeout):
                    async with stdio_client(server_params) as (stdio, write):
                        self.stdio, self.write = stdio, write
                        async with ClientSession(self.stdio, self.write) as session:
                            self.session = session
                            try:
                                init_timeout = (
                                    30.0 if os.getenv("INSIDE_DOCKER") else 15.0
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
                                    20.0 if os.getenv("INSIDE_DOCKER") else 10.0
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
        Call a tool on the MCP server using reconnection for each call.
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
        Returns:
            Tool execution result
        """
        try:
            self.logger.info(
                f"Calling tool '{tool_name}' on server '{self.server_name}' with arguments: {arguments}"
            )
            try:
                return await self._call_tool_with_config(
                    self.server_config, tool_name, arguments
                )
            except Exception as primary_error:
                self.logger.warning(
                    f"Primary tool call failed for '{tool_name}': {str(primary_error)}"
                )
                fallback_config = self.server_config.get("fallback")
                if fallback_config:
                    self.logger.info(
                        f"Trying fallback configuration for tool '{tool_name}'"
                    )
                    try:
                        return await self._call_tool_with_config(
                            fallback_config, tool_name, arguments
                        )
                    except Exception as fallback_error:
                        self.logger.error(
                            f"Fallback tool call also failed for '{tool_name}': {str(fallback_error)}"
                        )
                        raise RuntimeError(
                            f"Both primary and fallback tool calls failed for '{tool_name}'"
                        )
                raise primary_error
        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}' on server '{self.server_name}': {str(e) or 'Unknown error'}"
            self.logger.error(error_msg)
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(
                f"Exception args: {e.args if hasattr(e, 'args') else 'No args'}"
            )
            raise RuntimeError(error_msg) from e
    async def _call_tool_with_config(
        self, config: Dict[str, Any], tool_name: str, arguments: Dict[str, Any]
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
                async with asyncio.timeout(30.0):
                    async with stdio_client(server_params) as (stdio, write):
                        async with ClientSession(stdio, write) as session:
                            try:
                                async with asyncio.timeout(10.0):
                                    await session.initialize()
                            except asyncio.TimeoutError:
                                raise RuntimeError(
                                    f"Timeout initializing session for tool call '{tool_name}'"
                                )
                            try:
                                async with asyncio.timeout(20.0):
                                    result = await session.call_tool(
                                        tool_name, arguments
                                    )
                            except asyncio.TimeoutError:
                                raise RuntimeError(
                                    f"Timeout executing tool '{tool_name}' after 20 seconds"
                                )
                            self.logger.info(
                                f"Tool '{tool_name}' executed successfully"
                            )
                            return result.content
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Timeout during tool call '{tool_name}' after 30 seconds"
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
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool by name.
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
        Returns:
            Tool execution result
        """
        if tool_name not in self.tool_to_server_map:
            raise ValueError(
                f"Tool '{tool_name}' not found in any connected MCP server"
            )
        server_name = self.tool_to_server_map[tool_name]
        server = self.servers[server_name]
        return await server.call_tool(tool_name, arguments)
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
