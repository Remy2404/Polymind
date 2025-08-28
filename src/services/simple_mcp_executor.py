"""
Simple MCP Tool Executor

This module provides a simplified way to execute MCP tools without maintaining persistent connections.
Instead, it spawns MCP processes on-demand for each tool call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from .mcp_registry import MCPRegistry
from .direct_search_executor import get_direct_search_executor

logger = logging.getLogger(__name__)


class SimpleMCPExecutor:
    """
    Simple executor for MCP tools that spawns processes on-demand.
    
    This avoids the complexity of managing persistent connections and works
    more reliably for occasional tool calls.
    """
    
    def __init__(self, registry: Optional[MCPRegistry] = None):
        self.registry = registry or MCPRegistry()
    
    async def execute_exa_search(self, query: str) -> str:
        """Execute Exa web search."""
        try:
            result = await self._execute_smithery_tool("exa", "web_search", {"query": query})
            if result and not result.startswith("Error:"):
                return result
        except Exception as e:
            logger.warning(f"Smithery Exa search failed, using fallback: {e}")
        
        # Fallback to direct search
        logger.info("Using direct search fallback for web search")
        executor = get_direct_search_executor()
        return await executor.search_with_serpapi_fallback(query)
    
    async def execute_exa_company_research(self, company_name: str) -> str:
        """Execute Exa company research."""
        try:
            result = await self._execute_smithery_tool("exa", "company_research", {"company_name": company_name})
            if result and not result.startswith("Error:"):
                return result
        except Exception as e:
            logger.warning(f"Smithery company research failed, using fallback: {e}")
        
        # Fallback to direct search
        logger.info("Using direct search fallback for company research")
        executor = get_direct_search_executor()
        return await executor.research_company_fallback(company_name)
    
    async def execute_exa_crawl(self, url: str) -> str:
        """Execute Exa URL crawling."""
        try:
            result = await self._execute_smithery_tool("exa", "crawl", {"url": url})
            if result and not result.startswith("Error:"):
                return result
        except Exception as e:
            logger.warning(f"Smithery URL crawling failed, using fallback: {e}")
        
        # Fallback to direct extraction
        logger.info("Using direct extraction fallback for URL crawling")
        executor = get_direct_search_executor()
        return await executor.extract_url_content_fallback(url)
    
    async def execute_duckduckgo_search(self, query: str) -> str:
        """Execute DuckDuckGo search."""
        try:
            result = await self._execute_smithery_tool("duckduckgo", "search", {"query": query})
            if result and not result.startswith("Error:"):
                return result
        except Exception as e:
            logger.warning(f"Smithery DuckDuckGo search failed, using fallback: {e}")
        
        # Fallback to direct search
        logger.info("Using direct DuckDuckGo search fallback")
        executor = get_direct_search_executor()
        return await executor.search_duckduckgo(query)
    
    async def _execute_smithery_tool(self, service: str, action: str, params: Dict[str, Any]) -> str:
        """Execute a Smithery-based tool."""
        smithery_key = os.getenv("SMITHERY_API_KEY")
        if not smithery_key:
            return "Error: SMITHERY_API_KEY not configured"

        try:
            # Build the command
            cmd = [
                "npx", "-y", "@smithery/cli@latest", "run", service,
                "--key", smithery_key
            ]

            # Add action and parameters
            if action:
                cmd.extend(["--action", action])

            # Add parameters as JSON - ensure proper escaping
            if params:
                try:
                    params_json = json.dumps(params, ensure_ascii=True)
                    cmd.extend(["--params", params_json])
                except (TypeError, ValueError) as e:
                    logger.error(f"Failed to serialize parameters: {e}")
                    return f"Error: Invalid parameters for {service} {action}"

            logger.info(f"Executing Smithery command: {' '.join(cmd[:5])}...")  # Log partial command for security

            # Execute the command using subprocess.run for Windows compatibility
            import subprocess

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    encoding='utf-8',
                    errors='replace'  # Replace invalid characters
                )

                stdout = result.stdout
                stderr = result.stderr
                returncode = result.returncode

            except subprocess.TimeoutExpired:
                return "Error: Tool execution timed out after 30 seconds"
            except UnicodeDecodeError as e:
                logger.error(f"Unicode decoding error in tool output: {e}")
                return "Error: Tool returned invalid character encoding"

            logger.info(f"Smithery command completed with return code: {returncode}")
            if stderr:
                logger.warning(f"Smithery stderr: {stderr[:500]}...")  # Log first 500 chars

            if returncode == 0:
                # Validate that stdout is not empty and contains valid content
                if not stdout or stdout.strip() == "":
                    return "Error: Tool returned empty result"

                # Clean the result to ensure it's safe for JSON processing
                result = stdout.strip()

                # Remove any potential control characters that might break JSON parsing
                result = ''.join(char for char in result if ord(char) >= 32 or char in '\n\r\t')

                # Check if result looks like valid JSON and try to parse it
                if result.startswith('{') or result.startswith('['):
                    try:
                        # Try to parse as JSON to validate it's well-formed
                        json.loads(result)
                        logger.info("Tool returned valid JSON response")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Tool returned malformed JSON: {e}")
                        # Don't fail completely, just log the issue

                logger.info(f"Smithery tool result length: {len(result)}")
                return result
            else:
                error_msg = stderr.strip() if stderr else "Unknown error"
                logger.error(f"Smithery tool failed with code {returncode}: {error_msg[:500]}...")
                return f"Error: Tool execution failed - {error_msg[:200]}"

        except asyncio.TimeoutError:
            return "Error: Tool execution timed out"
        except Exception as e:
            logger.error(f"Error executing Smithery tool: {e}", exc_info=True)
            return f"Error: Tool execution failed - {str(e)[:200]}"
    
    async def execute_context7_search(self, query: str) -> str:
        """Execute Context7 documentation search."""
        try:
            # Build the command for Context7
            cmd = ["npx", "-y", "@upstash/context7-mcp@latest"]
            
            # Execute the command with query as input using subprocess.run for Windows compatibility
            import subprocess
            
            # Prepare input data
            input_data = json.dumps({
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {"query": query}
                }
            }) + "\n"
            
            try:
                result = subprocess.run(
                    cmd,
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                stdout = result.stdout
                stderr = result.stderr
                returncode = result.returncode
                
            except subprocess.TimeoutExpired:
                return "Error: Context7 search timed out"
            
            if returncode == 0:
                return stdout
            else:
                error_msg = stderr if stderr else "Unknown error"
                logger.error(f"Context7 tool failed: {error_msg}")
                return f"Error: {error_msg}"
                
        except asyncio.TimeoutError:
            return "Error: Context7 search timed out"
        except Exception as e:
            logger.error(f"Error executing Context7 search: {e}")
            return f"Error: {str(e)}"


# Global instance
_executor: Optional[SimpleMCPExecutor] = None


def get_mcp_executor() -> SimpleMCPExecutor:
    """Get or create the global MCP executor."""
    global _executor
    if _executor is None:
        _executor = SimpleMCPExecutor()
    return _executor


__all__ = ["SimpleMCPExecutor", "get_mcp_executor"]
