"""
MCP Server Registry - Manages multiple MCP servers and their tools.
Provides a centralized registry for all MCP servers in the Polymind AI system.
"""

import json
import logging
import os
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from urllib.parse import urlencode
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    
    name: str
    type: str  # "stdio", "streamable_http", etc.
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None
    enabled: bool = True
    description: str = ""
    tools: List[str] = field(default_factory=list)


@dataclass
class MCPTool:
    """Represents an MCP tool."""
    
    name: str
    server: str
    description: str
    schema: Optional[Dict[str, Any]] = None


class MCPRegistry:
    """
    Registry for managing multiple MCP servers and their tools.
    Handles dynamic loading, configuration, and tool access.
    """
    
    def __init__(self, config_path: str = "mcp.json"):
        """Initialize the MCP registry."""
        self.config_path = config_path
        self.servers: Dict[str, MCPServerConfig] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.active_connections: Dict[str, Any] = {}
        self.logger = logger
        
    async def load_config(self) -> bool:
        """Load MCP configuration from file."""
        try:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"MCP config file not found: {self.config_path}")
                return False
                
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                
            # Load servers
            servers_config = config.get("servers", {})
            for name, server_data in servers_config.items():
                server_config = MCPServerConfig(
                    name=name,
                    type=server_data.get("type", "stdio"),
                    command=server_data.get("command"),
                    args=server_data.get("args", []),
                    env=server_data.get("env", {}),
                    url=server_data.get("url"),
                    enabled=server_data.get("enabled", True),
                    description=server_data.get("description", "")
                )
                self.servers[name] = server_config
                
            # Load tools
            tools_config = config.get("tools", {})
            for tool_name, tool_data in tools_config.items():
                tool = MCPTool(
                    name=tool_name,
                    server=tool_data.get("server", ""),
                    description=tool_data.get("description", ""),
                    schema=tool_data.get("schema")
                )
                self.tools[tool_name] = tool
                
            self.logger.info(f"Loaded {len(self.servers)} MCP servers and {len(self.tools)} tools")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load MCP config: {e}")
            return False
            
    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get list of enabled MCP servers."""
        return [server for server in self.servers.values() if server.enabled]
        
    def get_server_tools(self, server_name: str) -> List[MCPTool]:
        """Get tools for a specific server."""
        return [tool for tool in self.tools.values() if tool.server == server_name]
        
    def get_all_tools(self) -> List[MCPTool]:
        """Get all available tools from enabled servers."""
        enabled_servers = {server.name for server in self.get_enabled_servers()}
        return [tool for tool in self.tools.values() if tool.server in enabled_servers]
        
    async def connect_to_smithery_server(self, server_name: str, api_key: str, profile: str = None) -> Optional[Any]:
        """
        Connect to a Smithery.ai MCP server.
        
        Args:
            server_name: Name of the server to connect to
            api_key: Smithery API key
            profile: Optional profile parameter
            
        Returns:
            Connection object or None if failed
        """
        try:
            server_config = self.servers.get(server_name)
            if not server_config or not server_config.enabled:
                self.logger.warning(f"Server {server_name} not found or disabled")
                return None
                
            if server_config.type == "streamable_http" and server_config.url:
                # Build URL with parameters
                params = {"api_key": api_key}
                if profile:
                    params["profile"] = profile
                    
                url = f"{server_config.url}?{urlencode(params)}"
                
                # Store connection info for later use
                self.active_connections[server_name] = {
                    "url": url,
                    "type": "streamable_http",
                    "api_key": api_key,
                    "profile": profile
                }
                
                self.logger.info(f"Connected to Smithery server: {server_name}")
                return self.active_connections[server_name]
                
        except Exception as e:
            self.logger.error(f"Failed to connect to server {server_name}: {e}")
            
        return None
        
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute a tool on its associated MCP server.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result or None if failed
        """
        try:
            tool = self.tools.get(tool_name)
            if not tool:
                self.logger.warning(f"Tool {tool_name} not found")
                return None
                
            server_connection = self.active_connections.get(tool.server)
            if not server_connection:
                self.logger.warning(f"No active connection for server {tool.server}")
                return None
                
            # For now, return a mock response for demonstration
            # In a real implementation, this would make the actual MCP call
            result = {
                "tool": tool_name,
                "server": tool.server,
                "status": "success",
                "result": f"Executed {tool_name} with parameters: {parameters}",
                "mock": True  # Indicates this is a mock response
            }
            
            self.logger.info(f"Executed tool {tool_name} on server {tool.server}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute tool {tool_name}: {e}")
            return None
            
    async def search_with_exa(self, query: str, search_type: str = "web") -> Optional[Dict[str, Any]]:
        """
        Perform search using Exa via MCP.
        
        Args:
            query: Search query
            search_type: Type of search ("web", "company", etc.)
            
        Returns:
            Search results or None if failed
        """
        try:
            if search_type == "company":
                tool_name = "exa_search_company_research_exa"
            else:
                tool_name = "exa_search_web_search_exa"
                
            parameters = {
                "query": query,
                "type": search_type,
                "num_results": 10,
                "include_domains": [],
                "exclude_domains": []
            }
            
            return await self.execute_tool(tool_name, parameters)
            
        except Exception as e:
            self.logger.error(f"Failed to perform Exa search: {e}")
            return None
            
    async def crawl_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Crawl and extract content from a URL using MCP.
        
        Args:
            url: URL to crawl
            
        Returns:
            Crawled content or None if failed
        """
        try:
            tool_name = "exa_search_content_crawler"
            parameters = {
                "url": url,
                "extract_text": True,
                "extract_links": False,
                "extract_images": False
            }
            
            return await self.execute_tool(tool_name, parameters)
            
        except Exception as e:
            self.logger.error(f"Failed to crawl URL {url}: {e}")
            return None
            
    def register_server(self, name: str, server_config: MCPServerConfig):
        """Register a new MCP server."""
        self.servers[name] = server_config
        self.logger.info(f"Registered MCP server: {name}")
        
    def unregister_server(self, name: str):
        """Unregister an MCP server."""
        if name in self.servers:
            del self.servers[name]
            # Clean up associated tools
            self.tools = {k: v for k, v in self.tools.items() if v.server != name}
            # Clean up connections
            if name in self.active_connections:
                del self.active_connections[name]
            self.logger.info(f"Unregistered MCP server: {name}")
            
    async def initialize(self) -> bool:
        """Initialize the MCP registry and establish connections."""
        try:
            # Load configuration
            if not await self.load_config():
                return False
                
            # Get environment variables for Smithery
            api_key = os.getenv("SMITHERY_API_KEY")
            profile = os.getenv("EXA_PROFILE")
            
            if api_key:
                # Connect to enabled Smithery servers
                for server_name, server_config in self.servers.items():
                    if server_config.enabled and "smithery" in server_config.description.lower():
                        await self.connect_to_smithery_server(server_name, api_key, profile)
            else:
                self.logger.warning("SMITHERY_API_KEY not set - Smithery servers will not be available")
                
            self.logger.info("MCP Registry initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP registry: {e}")
            return False
            
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers."""
        status = {}
        for name, server in self.servers.items():
            status[name] = {
                "enabled": server.enabled,
                "connected": name in self.active_connections,
                "type": server.type,
                "description": server.description,
                "tools": len(self.get_server_tools(name))
            }
        return status