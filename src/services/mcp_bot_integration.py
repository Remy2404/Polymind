"""
MCP Integration for Telegram Bot
This module provides MCP (Model Context Protocol) integration for the Telegram bot,
enabling automatic tool calling and server management.
"""
import os
import logging
from typing import Dict, List, Optional, Any
from src.services.mcp.mcp_client import MCPManager
from src.services.openrouter_api_with_mcp import OpenRouterAPIWithMCP
from src.services.rate_limiter import RateLimiter
class MCPBotIntegration:
    """MCP integration for Telegram bot."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mcp_manager: Optional[MCPManager] = None
        self.openrouter_with_mcp: Optional[OpenRouterAPIWithMCP] = None
        self.is_initialized = False
        self.user_mcp_enabled: Dict[int, bool] = {}
    async def initialize_mcp(self) -> bool:
        """
        Initialize MCP manager and OpenRouter with MCP support.
        Returns:
            True if initialization successful, False otherwise
        """
        import asyncio
        try:
            self.logger.info("Initializing MCP integration...")
            self.mcp_manager = MCPManager()
            try:
                load_timeout = 300.0 if os.getenv("INSIDE_DOCKER") else 120.0
                async with asyncio.timeout(load_timeout):
                    success = await self.mcp_manager.load_servers()
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"MCP server loading timed out after {load_timeout} seconds - continuing without MCP"
                )
                self.mcp_manager = None
                return False
            if not success:
                self.logger.warning(
                    "Failed to load MCP servers - MCP functionality will be disabled"
                )
                self.mcp_manager = None
                return False
            server_info = self.mcp_manager.get_server_info()
            self.logger.info(f"Connected to {len(server_info)} MCP servers")
            if os.getenv("OPENROUTER_API_KEY"):
                rate_limiter = RateLimiter(requests_per_minute=20)
                self.openrouter_with_mcp = OpenRouterAPIWithMCP(rate_limiter)
                try:
                    tools_timeout = 60.0 if os.getenv("INSIDE_DOCKER") else 30.0
                    async with asyncio.timeout(tools_timeout):
                        mcp_success = (
                            await self.openrouter_with_mcp.initialize_mcp_tools()
                        )
                    if not mcp_success:
                        self.logger.warning(
                            "Failed to initialize MCP tools in OpenRouter"
                        )
                        return False
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Timeout initializing MCP tools in OpenRouter after {tools_timeout} seconds - continuing without MCP"
                    )
                    self.openrouter_with_mcp = None
                    return False
                self.logger.info("OpenRouter MCP integration initialized")
            else:
                self.logger.warning("OPENROUTER_API_KEY not set - MCP tools disabled")
            self.is_initialized = True
            self.logger.info("✅ MCP integration initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP integration: {str(e)}")
            if self.mcp_manager:
                await self.mcp_manager.disconnect_all()
                self.mcp_manager = None
            if self.openrouter_with_mcp:
                await self.openrouter_with_mcp.close()
                self.openrouter_with_mcp = None
            return False
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available MCP tools in OpenAI format.
        Returns:
            List of tool definitions
        """
        if not self.is_initialized or not self.mcp_manager:
            return []
        try:
            return await self.mcp_manager.get_all_tools()
        except Exception as e:
            self.logger.error(f"Error getting MCP tools: {str(e)}")
            return []
    async def is_mcp_available_for_user(self, user_id: int) -> bool:
        """
        Check if MCP is available and enabled for a user.
        Args:
            user_id: Telegram user ID
        Returns:
            True if MCP is available for the user
        """
        if not self.is_initialized:
            return False
        if user_id in self.user_mcp_enabled:
            return self.user_mcp_enabled[user_id]
        return True
    async def set_mcp_enabled_for_user(self, user_id: int, enabled: bool):
        """
        Enable or disable MCP for a specific user.
        Args:
            user_id: Telegram user ID
            enabled: Whether to enable MCP for this user
        """
        self.user_mcp_enabled[user_id] = enabled
        status = "enabled" if enabled else "disabled"
        self.logger.info(f"MCP {status} for user {user_id}")
    def is_model_mcp_compatible(self, model: str) -> bool:
        """
        Check if a model is compatible with MCP (tool calling via OpenRouter).
        Args:
            model: Model identifier
        Returns:
            True if model supports MCP integration, False otherwise
        """
        if not model:
            return False
        from src.services.model_handlers.model_configs import ModelConfigurations
        return ModelConfigurations.model_supports_tool_calls(model)
    async def generate_response_with_mcp(
        self,
        prompt: str,
        user_id: int,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> Optional[str]:
        """
        Generate a response using MCP tools if available.
        Args:
            prompt: User prompt
            user_id: Telegram user ID
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments
        Returns:
            Generated response or None if failed
        """
        if not await self.is_mcp_available_for_user(user_id):
            self.logger.debug(f"MCP not available for user {user_id}")
            return None
        if not self.openrouter_with_mcp:
            self.logger.debug("OpenRouter with MCP not initialized")
            return None
        if model and not self.is_model_mcp_compatible(model):
            self.logger.debug(f"Model {model} is not MCP compatible, skipping MCP")
            return None
        try:
            self.logger.info(
                f"Generating MCP-enhanced response for user {user_id} with model {model}"
            )
            response = await self.openrouter_with_mcp.generate_response_with_mcp_tools(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            if response:
                self.logger.info(
                    f"MCP response generated successfully for user {user_id}"
                )
            else:
                self.logger.warning(f"No MCP response generated for user {user_id}")
            return response
        except Exception as e:
            self.logger.error(
                f"Error generating MCP response for user {user_id}: {str(e)}"
            )
            if "not a valid model ID" in str(e) or "400" in str(e):
                self.logger.warning(
                    f"Model {model} appears incompatible with OpenRouter MCP integration"
                )
            return None
    async def get_mcp_status(self) -> Dict[str, Any]:
        """
        Get MCP integration status.
        Returns:
            Dictionary with MCP status information
        """
        status = {
            "initialized": self.is_initialized,
            "servers": {},
            "tools_count": 0,
            "openrouter_available": self.openrouter_with_mcp is not None,
        }
        if self.is_initialized and self.mcp_manager:
            server_info = self.mcp_manager.get_server_info()
            status["servers"] = server_info
            status["tools_count"] = sum(
                info["tool_count"] for info in server_info.values()
            )
        return status
    async def cleanup(self):
        """Clean up MCP resources."""
        if self.mcp_manager:
            await self.mcp_manager.disconnect_all()
            self.mcp_manager = None
        if self.openrouter_with_mcp:
            await self.openrouter_with_mcp.close()
            self.openrouter_with_mcp = None
        self.is_initialized = False
        self.logger.info("MCP integration cleaned up")
mcp_integration = MCPBotIntegration()
async def initialize_mcp_for_bot():
    """Initialize MCP integration for the bot."""
    return await mcp_integration.initialize_mcp()
async def get_mcp_tools_for_user(user_id: int) -> List[Dict[str, Any]]:
    """Get available MCP tools for a user."""
    if await mcp_integration.is_mcp_available_for_user(user_id):
        return await mcp_integration.get_available_tools()
    return []
async def generate_mcp_response(
    prompt: str, user_id: int, model: Optional[str] = None, **kwargs
) -> Optional[str]:
    """Generate a response with MCP tools."""
    return await mcp_integration.generate_response_with_mcp(
        prompt=prompt, user_id=user_id, model=model, **kwargs
    )
def is_model_mcp_compatible(model: str) -> bool:
    """
    Check if a model is compatible with MCP integration.
    Args:
        model: Model identifier
    Returns:
        True if model supports MCP, False otherwise
    """
    return mcp_integration.is_model_mcp_compatible(model)
