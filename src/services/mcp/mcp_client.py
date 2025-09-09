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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # Fallback if python-dotenv is not available
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
            # Handle $VAR_NAME syntax
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
        # Convert MCP tool to OpenAI format
        openai_tool = {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

        # Convert input schema properties
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
            # Create server parameters based on config type
            if self.server_config.get("type") == "stdio":
                server_params = StdioServerParameters(
                    command=self.server_config["command"],
                    args=self.server_config.get("args", []),
                    env={**os.environ, **self.server_config.get("env", {})},
                )

                # Use proper async context manager
                try:
                    async with asyncio.timeout(30.0):  # 30 second timeout
                        async with stdio_client(server_params) as (stdio, write):
                            self.stdio, self.write = stdio, write

                            # Create session within the same context
                            async with ClientSession(self.stdio, self.write) as session:
                                self.session = session

                                # Initialize the session with timeout
                                try:
                                    async with asyncio.timeout(15.0):
                                        await session.initialize()
                                except asyncio.TimeoutError:
                                    self.logger.warning(
                                        f"Timeout initializing MCP server session '{self.server_name}' after 15 seconds"
                                    )
                                    return False

                                # List available tools with timeout
                                try:
                                    async with asyncio.timeout(10.0):
                                        response = await session.list_tools()
                                        self.available_tools = response.tools
                                except asyncio.TimeoutError:
                                    self.logger.warning(
                                        f"Timeout listing tools from MCP server '{self.server_name}' after 10 seconds"
                                    )
                                    return False

                                # Convert to OpenAI format
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

                                # Store server config for later reconnection
                                self._server_params = server_params

                                return True

                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Timeout connecting to MCP server '{self.server_name}' after 30 seconds"
                    )
                    return False
                except Exception as conn_error:
                    self.logger.error(
                        f"Error establishing stdio connection to '{self.server_name}': {str(conn_error)}"
                    )
                    return False

            elif self.server_config.get("type") == "sse":
                # SSE connection would require different handling
                self.logger.warning("SSE connections not yet implemented")
                return False

            elif self.server_config.get("type") == "http":
                # HTTP connection would require different handling
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
            self.logger.info(f"Calling tool '{tool_name}' on server '{self.server_name}' with arguments: {arguments}")

            # Reconnect for each tool call to avoid connection persistence issues
            if self.server_config.get("type") == "stdio":
                server_params = StdioServerParameters(
                    command=self.server_config["command"],
                    args=self.server_config.get("args", []),
                    env={**os.environ, **self.server_config.get("env", {})},
                )

                # Use proper async context manager for tool call
                try:
                    async with asyncio.timeout(30.0):  # 30 second timeout for tool call
                        async with stdio_client(server_params) as (stdio, write):
                            async with ClientSession(stdio, write) as session:
                                # Initialize session
                                try:
                                    async with asyncio.timeout(10.0):
                                        await session.initialize()
                                except asyncio.TimeoutError:
                                    raise RuntimeError(f"Timeout initializing session for tool call '{tool_name}'")

                                # Call the tool
                                try:
                                    async with asyncio.timeout(20.0):  # 20 second timeout for tool execution
                                        result = await session.call_tool(tool_name, arguments)
                                except asyncio.TimeoutError:
                                    raise RuntimeError(f"Timeout executing tool '{tool_name}' after 20 seconds")

                                self.logger.info(f"Tool '{tool_name}' executed successfully")
                                return result.content

                except asyncio.TimeoutError:
                    raise RuntimeError(f"Timeout during tool call '{tool_name}' after 30 seconds")
                except Exception as conn_error:
                    raise RuntimeError(f"Connection error during tool call '{tool_name}': {str(conn_error)}")

            else:
                raise RuntimeError(f"Unsupported server type for tool calls: {self.server_config.get('type')}")

        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}' on server '{self.server_name}': {str(e) or 'Unknown error'}"
            self.logger.error(error_msg)
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception args: {e.args if hasattr(e, 'args') else 'No args'}")
            raise RuntimeError(error_msg) from e

    async def disconnect(self):
        """Disconnect from the MCP server."""
        try:
            # Clean up any remaining resources
            if hasattr(self, 'session') and self.session:
                try:
                    await self.session.__aexit__(None, None, None)
                except Exception:
                    pass  # Ignore cleanup errors

            if hasattr(self, 'stdio') and self.stdio:
                try:
                    await self.stdio.__aexit__(None, None, None)
                except Exception:
                    pass  # Ignore cleanup errors

            # Clear all references
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
        # Validate environment variables first
        if not validate_mcp_environment():
            self.logger.error("MCP environment validation failed")
            return False

        if not self.mcp_config_path.exists():
            self.logger.warning(f"MCP config file not found: {self.mcp_config_path}")
            return False

        try:
            with open(self.mcp_config_path, "r") as f:
                config = json.load(f)

            # Substitute environment variables in configuration
            config = substitute_env_vars(config)

            # Log successful environment variable substitution
            self.logger.info("Environment variables substituted in MCP configuration")
            telegram_logger.log_message(
                "Environment variables substituted in MCP configuration", 0
            )

            servers_config = config.get("servers", {})
            connection_tasks = []

            # Create connection tasks for all servers
            for server_name, server_config in servers_config.items():
                task = asyncio.create_task(
                    self._connect_single_server(server_name, server_config),
                    name=f"mcp_connect_{server_name}",
                )
                connection_tasks.append(task)

            # Wait for all connections to complete with a global timeout
            if connection_tasks:
                try:
                    # Use asyncio.wait instead of gather to handle exceptions better
                    done, pending = await asyncio.wait(
                        connection_tasks,
                        timeout=120,  # 2 minute total timeout
                        return_when=asyncio.ALL_COMPLETED,
                    )

                    # Cancel any pending tasks
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

            # Handle different config formats
            if isinstance(server_config, dict) and "type" in server_config:
                # Direct server config
                client = MCPServerClient(server_config, server_name)
            else:
                # Nested config format
                client = MCPServerClient({server_name: server_config}, server_name)

            # Connect with timeout
            try:
                async with asyncio.timeout(60.0):  # 60 second timeout per server
                    success = await client.connect()
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timeout connecting to MCP server '{server_name}' after 60 seconds"
                )
                return False
            except Exception as conn_error:
                self.logger.error(
                    f"Error connecting to MCP server '{server_name}': {str(conn_error)}"
                )
                return False

            if success:
                self.servers[server_name] = client

                # Map tools to server
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
            # Wait for all disconnections with timeout
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
