import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils.log.telegramlog import telegram_logger
from .config import substitute_env_vars, validate_mcp_environment
from .server_client import MCPServerClient


class MCPManager:
    """Manages multiple MCP server connections and tool aggregation (Singleton)."""

    _instance: Optional["MCPManager"] = None
    _lock = asyncio.Lock()
    _initialized = False

    def __new__(cls, mcp_config_path: str = "mcp.json"):
        """
        Singleton pattern: Ensure only one instance exists.
        """
        if cls._instance is None:
            cls._instance = super(MCPManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, mcp_config_path: str = "mcp.json"):
        """
        Initialize MCP manager (only once due to singleton).
        Args:
            mcp_config_path: Path to MCP configuration file
        """
        # Skip re-initialization if already initialized
        if self.__class__._initialized:
            return

        self.mcp_config_path = Path(mcp_config_path)
        self.servers: Dict[str, MCPServerClient] = {}
        self.logger = logging.getLogger(__name__)
        self.all_openai_tools: List[Dict[str, Any]] = []
        self.tool_to_server_map: Dict[str, str] = {}

    @classmethod
    def get_instance(cls, mcp_config_path: str = "mcp.json") -> "MCPManager":
        """
        Get the singleton instance of MCPManager.
        Args:
            mcp_config_path: Path to MCP configuration file
        Returns:
            The singleton MCPManager instance
        """
        if cls._instance is None:
            cls._instance = cls(mcp_config_path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance (useful for testing).
        """
        cls._instance = None
        cls._initialized = False

    async def load_servers(self) -> bool:
        """
        Load and connect to all MCP servers from configuration.
        Uses singleton pattern to prevent re-initialization.
        Returns:
            True if any servers connected successfully
        """
        # Use async lock to prevent concurrent initialization
        async with self.__class__._lock:
            # If already initialized, return success
            if self.__class__._initialized:
                self.logger.info(
                    "MCP servers already initialized, skipping re-initialization"
                )
                return len(self.servers) > 0

            if not validate_mcp_environment():
                self.logger.error("MCP environment validation failed")
                return False
            if not self.mcp_config_path.exists():
                self.logger.warning(
                    f"MCP config file not found: {self.mcp_config_path}"
                )
                return False
            try:
                with open(self.mcp_config_path, "r") as f:
                    config = json.load(f)
                config = substitute_env_vars(config)
                self.logger.info(
                    "Environment variables substituted in MCP configuration"
                )
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
                        # Reduced timeout for faster startup - broken servers should fail quickly
                        global_timeout = 30.0  # 30 seconds total for all servers
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
                            # Mark as initialized only after successful connection
                            self.__class__._initialized = True
                            return True
                        else:
                            self.logger.warning("No MCP servers connected successfully")
                            return False
                    except Exception as e:
                        self.logger.error(
                            f"Error during MCP server connections: {str(e)}"
                        )
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
                # Reduced timeout for faster startup - broken servers should fail quickly
                server_timeout = 10.0  # 10 seconds per server
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

    async def get_all_tools(self, provider: str = "openai") -> List[Dict[str, Any]]:
        """
        Get all available tools from connected MCP servers in the specified format.

        Args:
            provider: Tool format to return ("openai" or "meta")

        Returns:
            List of tool definitions in the requested format
        """
        if provider.lower() == "meta":
            # Aggregate Meta-format tools from all servers
            return [
                tool for server in self.servers.values() for tool in server.meta_tools
            ]
        else:
            # Default to OpenAI format
            return self.all_openai_tools.copy()

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                    arguments,
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
                "status": "completed",
            }

        except asyncio.TimeoutError:
            execution_time = asyncio.get_event_loop().time() - start_time
            return self._create_error_response(
                tool_name,
                "TIMEOUT",
                f"Tool execution timed out after {execution_time:.1f}s",
                arguments,
                execution_time,
            )

        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            error_type = self._classify_error(e)

            # Try fallback if available and this is a recoverable error
            if error_type in [
                "CONNECTION_ERROR",
                "SERVER_ERROR",
                "TIMEOUT",
            ] and self._has_fallback_available(tool_name):
                try:
                    self.logger.info(
                        f"Attempting fallback execution for tool '{tool_name}'"
                    )
                    fallback_result = await self._execute_tool_with_fallback(
                        tool_name, arguments
                    )
                    if fallback_result:
                        fallback_result["used_fallback"] = True
                        return fallback_result
                except Exception as fallback_error:
                    self.logger.warning(
                        f"Fallback execution also failed for '{tool_name}': {str(fallback_error)}"
                    )

            return self._create_error_response(
                tool_name, error_type, str(e), arguments, execution_time
            )

    def _create_error_response(
        self,
        tool_name: str,
        error_type: str,
        error_message: str,
        arguments: Dict[str, Any],
        execution_time: float = 0.0,
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
            "timestamp": asyncio.get_event_loop().time(),
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
        return hasattr(server, "server_config") and "fallback" in server.server_config

    async def _execute_tool_with_fallback(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a tool using fallback configuration.
        """
        if tool_name not in self.tool_to_server_map:
            return None

        server_name = self.tool_to_server_map[tool_name]
        server = self.servers.get(server_name)
        if not server or not hasattr(server, "server_config"):
            return None

        fallback_config = server.server_config.get("fallback")
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
                        "status": "completed",
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
