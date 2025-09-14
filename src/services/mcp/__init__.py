"""
MCP (Model Context Protocol) Integration Package
This package provides automatic MCP server integration with OpenRouter,
enabling LLMs to use MCP tools without hardcoded definitions.
"""
from .mcp_client import MCPManager, MCPToolConverter, MCPServerClient
__all__ = ["MCPManager", "MCPToolConverter", "MCPServerClient"]
