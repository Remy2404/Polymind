"""
MCP Client for Pydantic AI Integration

This module provides a proper MCP client that connects to MCP servers
and exposes their tools to Pydantic AI agents.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from .mcp_registry import MCPRegistry

logger = logging.getLogger(__name__)


@dataclass
class MCPToolCall:
    """Represents a call to an MCP tool."""
    tool_name: str
    arguments: Dict[str, Any]
    server_name: str


class MCPToolResult(BaseModel):
    """Result from an MCP tool call."""
    success: bool
    content: str
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class MCPClient:
    """
    Client for managing MCP server connections and tool invocations.
    
    This client manages multiple MCP servers and provides a unified interface
    for calling tools across all connected servers.
    """
    
    def __init__(self, registry: Optional[MCPRegistry] = None):
        self.registry = registry or MCPRegistry()
        self._sessions: Dict[str, ClientSession] = {}
        self._tools_map: Dict[str, str] = {}  # tool_name -> server_name
        self._connected_servers: set[str] = set()
        
    async def connect_servers(self) -> None:
        """Connect to all enabled MCP servers."""
        enabled_servers = self.registry.enabled_servers()
        logger.info(f"Connecting to MCP servers: {enabled_servers}")
        
        for server_name in enabled_servers:
            try:
                await self._connect_server(server_name)
            except Exception as e:
                logger.error(f"Failed to connect to server {server_name}: {e}")
    
    async def _connect_server(self, server_name: str) -> None:
        """Connect to a specific MCP server."""
        server_config = self.registry.server_config(server_name)
        if not server_config:
            logger.warning(f"No config found for server: {server_name}")
            return
            
        server_type = server_config.get("type", "stdio")
        if server_type != "stdio":
            logger.warning(f"Server type '{server_type}' not supported yet for {server_name}")
            return
            
        # Prepare command and arguments
        command = server_config.get("command")
        args = server_config.get("args", [])
        
        if not command:
            logger.warning(f"No command specified for server: {server_name}")
            return
            
        # Expand environment variables in args
        expanded_args = []
        for arg in args:
            if isinstance(arg, str) and "${" in arg:
                # Simple environment variable expansion
                for env_var in ["SMITHERY_API_KEY"]:
                    placeholder = f"${{{env_var}}}"
                    if placeholder in arg:
                        env_value = os.getenv(env_var)
                        if env_value:
                            arg = arg.replace(placeholder, env_value)
                        else:
                            logger.warning(f"Environment variable {env_var} not set for {server_name}")
                            return
            expanded_args.append(arg)
        
        try:
            # Create server parameters
            server_params = StdioServerParameters(
                command=command,
                args=expanded_args,
                env=dict(os.environ)  # Pass current environment
            )
            
            # Connect using stdio client
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the server
                    await session.initialize()
                    
                    # List available tools
                    tools_result = await session.list_tools()
                    
                    # Store session and tools mapping
                    self._sessions[server_name] = session
                    
                    # Map tools to server
                    for tool in tools_result.tools:
                        self._tools_map[tool.name] = server_name
                        logger.debug(f"Registered tool '{tool.name}' from server '{server_name}'")
                    
                    self._connected_servers.add(server_name)
                    logger.info(f"Connected to server '{server_name}' with {len(tools_result.tools)} tools")
                    
                    # Keep connection alive
                    # Note: In a real implementation, we'd manage the connection lifecycle differently
                    # For now, we'll store the session reference
                    
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            raise
    
    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> MCPToolResult:
        """
        Call an MCP tool by name.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            MCPToolResult containing the result of the tool call
        """
        if not arguments:
            arguments = {}
            
        # Find which server hosts this tool
        server_name = self._tools_map.get(tool_name)
        if not server_name:
            return MCPToolResult(
                success=False,
                content="",
                error=f"Tool '{tool_name}' not found in any connected server"
            )
        
        # Get the session for this server
        session = self._sessions.get(server_name)
        if not session:
            return MCPToolResult(
                success=False,
                content="",
                error=f"No active session for server '{server_name}'"
            )
        
        try:
            # Call the tool
            result = await session.call_tool(tool_name, arguments)
            
            # Extract content from result
            content_parts = []
            for content_item in result.content:
                if hasattr(content_item, 'text'):
                    content_parts.append(content_item.text)
                elif hasattr(content_item, 'content'):
                    content_parts.append(str(content_item.content))
                else:
                    content_parts.append(str(content_item))
            
            content = "\n".join(content_parts) if content_parts else ""
            
            return MCPToolResult(
                success=True,
                content=content,
                metadata={
                    "server": server_name,
                    "tool": tool_name,
                    "is_error": result.isError if hasattr(result, 'isError') else False
                }
            )
            
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}' on server '{server_name}': {e}")
            return MCPToolResult(
                success=False,
                content="",
                error=f"Tool call failed: {str(e)}"
            )
    
    def list_available_tools(self) -> List[str]:
        """List all available tools across connected servers."""
        return list(self._tools_map.keys())
    
    def get_tool_server(self, tool_name: str) -> Optional[str]:
        """Get the server name that hosts a specific tool."""
        return self._tools_map.get(tool_name)
    
    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for server_name in list(self._sessions.keys()):
            try:
                # Note: In a real implementation, we'd properly close sessions
                # For now, just remove from our tracking
                del self._sessions[server_name]
                self._connected_servers.discard(server_name)
                logger.info(f"Disconnected from server: {server_name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {server_name}: {e}")
        
        self._tools_map.clear()


# Singleton instance for the application
_mcp_client: Optional[MCPClient] = None


async def get_mcp_client() -> MCPClient:
    """Get or create the global MCP client instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
        await _mcp_client.connect_servers()
    return _mcp_client


async def initialize_mcp_client() -> MCPClient:
    """Initialize the MCP client and connect to servers."""
    client = MCPClient()
    await client.connect_servers()
    return client


__all__ = ["MCPClient", "MCPToolCall", "MCPToolResult", "get_mcp_client", "initialize_mcp_client"]
