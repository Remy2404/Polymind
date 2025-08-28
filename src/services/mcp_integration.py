"""
MCP Integration service for initializing and managing MCP clients with Pydantic AI.
"""

import logging
from typing import Optional

from .agent import EnhancedAgent
from .mcp_client import MCPClient, initialize_mcp_client
from .mcp_registry import MCPRegistry

logger = logging.getLogger(__name__)


class MCPIntegrationService:
    """Service for managing MCP integration with Pydantic AI agents."""
    
    def __init__(self):
        self.registry: Optional[MCPRegistry] = None
        self.client: Optional[MCPClient] = None
        self.agent: Optional[EnhancedAgent] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the MCP integration service."""
        try:
            # Initialize registry
            self.registry = MCPRegistry()
            logger.info("MCP Registry initialized")
            
            # Initialize MCP client and connect to servers
            self.client = await initialize_mcp_client()
            logger.info(f"MCP Client initialized with tools: {self.client.list_available_tools()}")
            
            # Initialize agent
            self.agent = EnhancedAgent(registry=self.registry)
            self.agent.set_mcp_client(self.client)
            logger.info("Enhanced Agent initialized with MCP tools")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP integration: {e}")
            self._initialized = False
            return False
    
    def is_initialized(self) -> bool:
        """Check if the service is properly initialized."""
        return self._initialized and self.agent is not None and self.client is not None
    
    def get_agent(self) -> Optional[EnhancedAgent]:
        """Get the initialized agent."""
        return self.agent if self.is_initialized() else None
    
    def get_available_tools(self) -> list[str]:
        """Get list of available MCP tools."""
        if self.client:
            return self.client.list_available_tools()
        return []
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.disconnect_all()
        self._initialized = False


# Global service instance
_mcp_service: Optional[MCPIntegrationService] = None


async def get_mcp_service() -> MCPIntegrationService:
    """Get or create the global MCP integration service."""
    global _mcp_service
    if _mcp_service is None:
        _mcp_service = MCPIntegrationService()
        await _mcp_service.initialize()
    return _mcp_service


async def initialize_mcp_service() -> MCPIntegrationService:
    """Initialize the MCP integration service."""
    service = MCPIntegrationService()
    await service.initialize()
    return service


__all__ = ["MCPIntegrationService", "get_mcp_service", "initialize_mcp_service"]
