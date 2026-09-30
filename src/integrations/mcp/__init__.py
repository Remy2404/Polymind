"""
MCP (Model Context Protocol) Integration Package
This package provides automatic MCP server integration with OpenRouter,
enabling LLMs to use MCP tools without hardcoded definitions.
"""

from .manager import MCPManager
from .server_client import MCPServerClient
from .tool_converter import MCPToolConverter

__all__ = ["MCPManager", "MCPToolConverter", "MCPServerClient"]
