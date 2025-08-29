"""
MCP (Model Context Protocol) Registry and Server Management
Dynamically loads and manages multiple MCP servers from mcp.json configuration.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.mcp import MCPServerStdio


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    type: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    enabled: bool = True
    tool_prefix: Optional[str] = None


class MCPRegistry:
    """
    Registry for managing multiple MCP servers.
    Loads server configurations from mcp.json and provides toolsets for Pydantic AI agents.
    """
    
    def __init__(self, config_file: str = "mcp.json", base_path: Optional[str] = None):
        """
        Initialize MCP Registry.
        
        Args:
            config_file: Path to MCP configuration file
            base_path: Base path for resolving relative paths (defaults to project root)
        """
        self.logger = logging.getLogger(__name__)
        self.servers: Dict[str, MCPServerStdio] = {}
        self.server_configs: Dict[str, MCPServerConfig] = {}
        
        # Determine base path (project root)
        if base_path is None:
            current_file = Path(__file__)
            # Navigate up to project root (src/services/mcp.py -> ../../)
            self.base_path = current_file.parent.parent.parent
        else:
            self.base_path = Path(base_path)
            
        self.config_file_path = self.base_path / config_file
        
        self.logger.info(f"Initializing MCP Registry with config: {self.config_file_path}")
        
    async def initialize(self) -> None:
        """Initialize all MCP servers from configuration."""
        try:
            await self._load_config()
            await self._initialize_servers()
            self.logger.info(f"Successfully initialized {len(self.servers)} MCP servers")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP Registry: {e}")
            raise
            
    async def _load_config(self) -> None:
        """Load MCP server configurations from mcp.json."""
        try:
            if not self.config_file_path.exists():
                raise FileNotFoundError(f"MCP config file not found: {self.config_file_path}")
                
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            servers_config = config_data.get('servers', {})
            
            for server_name, server_data in servers_config.items():
                # Process environment variables in args
                processed_args = []
                for arg in server_data.get('args', []):
                    if isinstance(arg, str) and arg.startswith('${') and arg.endswith('}'):
                        env_var = arg[2:-1]  # Remove ${ and }
                        env_value = os.getenv(env_var)
                        if env_value:
                            processed_args.append(env_value)
                        else:
                            self.logger.warning(f"Environment variable {env_var} not found for server {server_name}")
                            processed_args.append(arg)  # Keep original if env var not found
                    else:
                        processed_args.append(arg)
                
                config = MCPServerConfig(
                    name=server_name,
                    type=server_data.get('type', 'stdio'),
                    command=server_data.get('command', ''),
                    args=processed_args,
                    env=server_data.get('env'),
                    enabled=server_data.get('enabled', True),
                    tool_prefix=server_data.get('tool_prefix')
                )
                
                self.server_configs[server_name] = config
                self.logger.info(f"Loaded config for MCP server: {server_name}")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in MCP config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load MCP config: {e}")
            
    async def _initialize_servers(self) -> None:
        """Initialize MCP server instances."""
        for server_name, config in self.server_configs.items():
            if not config.enabled:
                self.logger.info(f"Skipping disabled MCP server: {server_name}")
                continue
                
            try:
                # Currently only supporting STDIO transport
                if config.type != 'stdio':
                    self.logger.warning(f"Unsupported transport type '{config.type}' for server {server_name}")
                    continue
                    
                # Prepare environment variables
                env = os.environ.copy()
                if config.env:
                    env.update(config.env)
                    
                # Create MCPServerStdio instance
                server = MCPServerStdio(
                    command=config.command,
                    args=config.args,
                    env=env
                )
                
                self.servers[server_name] = server
                self.logger.info(f"Initialized MCP server: {server_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize MCP server {server_name}: {e}")
                # Continue with other servers even if one fails
                continue
                
    def get_server(self, server_name: str) -> Optional[MCPServerStdio]:
        """Get a specific MCP server by name."""
        return self.servers.get(server_name)
        
    def get_all_servers(self) -> Dict[str, MCPServerStdio]:
        """Get all initialized MCP servers."""
        return self.servers.copy()
        
    def get_toolsets(self) -> List[MCPServerStdio]:
        """Get all MCP servers as toolsets for Pydantic AI agents."""
        return list(self.servers.values())
        
    def get_server_names(self) -> List[str]:
        """Get names of all initialized servers."""
        return list(self.servers.keys())
        
    def is_server_available(self, server_name: str) -> bool:
        """Check if a specific server is available."""
        return server_name in self.servers
        
    def get_server_config(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get configuration for a specific server."""
        return self.server_configs.get(server_name)
        
    async def reload_config(self) -> None:
        """Reload configuration and reinitialize servers."""
        self.logger.info("Reloading MCP configuration...")
        
        # Clear existing servers
        await self.shutdown()
        
        # Reload configuration
        await self.initialize()
        
    async def shutdown(self) -> None:
        """Shutdown all MCP servers and clean up resources."""
        self.logger.info("Shutting down MCP Registry...")
        
        for server_name, server in self.servers.items():
            try:
                # MCP servers will be cleaned up when the agent context exits
                self.logger.info(f"Shutdown MCP server: {server_name}")
            except Exception as e:
                self.logger.error(f"Error shutting down MCP server {server_name}: {e}")
                
        self.servers.clear()
        self.server_configs.clear()
        
    def get_available_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available tools from all servers.
        Note: This is informational only - actual tool schemas are retrieved at runtime.
        """
        tools_info = {}
        
        for server_name, config in self.server_configs.items():
            if server_name in self.servers:
                prefix = config.tool_prefix
                tools_info[server_name] = {
                    "server_name": server_name,
                    "command": config.command,
                    "args": config.args,
                    "tool_prefix": prefix,
                    "status": "initialized"
                }
            else:
                tools_info[server_name] = {
                    "server_name": server_name,
                    "status": "not_initialized" if config.enabled else "disabled"
                }
                
        return tools_info
        
    async def discover_available_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Dynamically discover available tools from all initialized MCP servers.
        Returns a mapping of server names to their available tools.
        """
        discovered_tools = {}
        
        for server_name, server in self.servers.items():
            try:
                # Create a temporary agent to discover tools
                from pydantic_ai import Agent
                from pydantic_ai.models.openai import OpenAIModel
                from pydantic_ai.providers.openai import OpenAIProvider
                
                # Use a minimal model for tool discovery
                provider = OpenAIProvider(
                    base_url="https://openrouter.ai/api/v1",
                    api_key="dummy_key"  # Won't be used for discovery
                )
                model = OpenAIModel(
                    model_name="qwen/qwen3-235b-a22b:free",
                    provider=provider
                )
                
                # Create agent with this server only
                agent = Agent(
                    model=model,
                    system_prompt="You are a tool discovery assistant.",
                    toolsets=[server]
                )
                
                # Try to get tool information by inspecting the agent
                # This is a simplified approach - in practice, you might need
                # to use MCP protocol directly to get tool schemas
                discovered_tools[server_name] = []
                
                # For now, we'll use known tool patterns based on server type
                if server_name == "Exa Search":
                    discovered_tools[server_name] = [
                        {"name": "web_search_exa", "description": "Web search using Exa AI"},
                        {"name": "company_research", "description": "Company research and analysis"},
                        {"name": "research_paper_search", "description": "Academic paper search"},
                        {"name": "crawling", "description": "Web page crawling and content extraction"},
                        {"name": "competitor_finder", "description": "Find company competitors"},
                        {"name": "linkedin_search", "description": "LinkedIn profile and company search"},
                        {"name": "wikipedia_search_exa", "description": "Wikipedia search"},
                        {"name": "github_search", "description": "GitHub repository and code search"}
                    ]
                elif server_name == "Context7":
                    discovered_tools[server_name] = [
                        {"name": "resolve-library-id", "description": "Resolve library/package names"},
                        {"name": "get-library-docs", "description": "Get library documentation"}
                    ]
                elif server_name == "sequentialthinking":
                    discovered_tools[server_name] = [
                        {"name": "sequentialthinking", "description": "Step-by-step problem solving"}
                    ]
                elif server_name == "Docfork":
                    discovered_tools[server_name] = [
                        {"name": "get-library-docs", "description": "Document analysis and insights"}
                    ]
                    
                self.logger.info(f"Discovered {len(discovered_tools[server_name])} tools from {server_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to discover tools from {server_name}: {e}")
                discovered_tools[server_name] = []
                
        return discovered_tools


# Global MCP registry instance
_mcp_registry: Optional[MCPRegistry] = None


async def get_mcp_registry() -> MCPRegistry:
    """Get the global MCP registry instance, initializing if necessary."""
    global _mcp_registry
    
    if _mcp_registry is None:
        _mcp_registry = MCPRegistry()
        await _mcp_registry.initialize()
        
    return _mcp_registry


async def shutdown_mcp_registry() -> None:
    """Shutdown the global MCP registry."""
    global _mcp_registry
    
    if _mcp_registry is not None:
        await _mcp_registry.shutdown()
        _mcp_registry = None
