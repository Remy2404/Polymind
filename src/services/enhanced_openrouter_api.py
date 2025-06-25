"""
Integration module for the OpenRouter API with Enhanced MCP support

This module provides an improved OpenRouter API client with robust MCP server integration.
"""

import os
import json
import aiohttp
import asyncio
import logging
import traceback
import time
import subprocess
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from src.services.rate_limiter import RateLimiter, rate_limit
from src.utils.log.telegramlog import telegram_logger
from src.services.mcp_enhanced import EnhancedMCPServerManager

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

if not OPENROUTER_API_KEY:
    telegram_logger.log_error(
        "OPENROUTER_API_KEY not found in environment variables.", 0
    )

if not EXA_API_KEY:
    telegram_logger.log_error(
        "EXA_API_KEY not found in environment variables.", 0
    )


class EnhancedOpenRouterAPI:
    def __init__(self, rate_limiter: RateLimiter):
        """
        Initialize the Enhanced OpenRouter API client with improved MCP support.
        
        Args:
            rate_limiter: The rate limiter to use for API calls
        """
        self.rate_limiter = rate_limiter
        self.logger = logging.getLogger(__name__)
        telegram_logger.log_message("Initializing Enhanced OpenRouter API with MCP support", 0)

        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found or empty")
            
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        # Initialize available models
        # This is a condensed sample - in production code you would include the full model list
        self.available_models = {
            # Core models
            "claude-3-haiku": "anthropic/claude-3-haiku:free",
            "claude-3-sonnet": "anthropic/claude-3-sonnet:free",
            "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet:free",
            "gpt-3.5-turbo": "openai/gpt-3.5-turbo:free",
            "gpt-4o": "openai/gpt-4o:free",
            "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct:free",
            "llama-3-70b": "meta-llama/llama-3-70b-instruct:free",
            "gemini-flash-1.5": "google/gemini-flash-1.5:free",
            "gemini-pro-1.5": "google/gemini-pro-1.5:free",
            
            # Add other models as needed
        }

        # Circuit breaker properties
        self.api_failures = 0
        self.api_last_failure = 0
        self.circuit_breaker_threshold = 5  # Number of failures before opening circuit
        self.circuit_breaker_timeout = 300  # Seconds to keep circuit open (5 minutes)
        
        # Initialize Enhanced MCP server manager
        self.logger.info("Initializing Enhanced MCP Server Manager")
        self.mcp_manager = EnhancedMCPServerManager()
        
        # Initialize session
        self.session = None

    async def ensure_session(self):
        """Create or reuse aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            self.logger.info("Created new OpenRouter API aiohttp session")
        return self.session

    async def close(self):
        """Close the aiohttp session and stop all MCP servers"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("Closed OpenRouter API aiohttp session")
            self.session = None
        
        # Stop all MCP servers
        try:
            await self.mcp_manager.stop_all_servers()
            self.logger.info("All MCP servers stopped")
        except Exception as e:
            self.logger.error(f"Error stopping MCP servers: {e}")

    @rate_limit
    async def generate_response(
        self,
        prompt: str,
        context: List[Dict] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a text response using the OpenRouter API.
        
        Args:
            prompt: The prompt to generate a response for
            context: Optional list of prior conversation messages
            model: The model ID to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            timeout: API call timeout in seconds
            
        Returns:
            str: Generated response text or None if failed
        """
        await self.rate_limiter.acquire()

        try:
            # Ensure we have a session
            session = await self.ensure_session()

            # Map model ID to OpenRouter model path
            openrouter_model = self.available_models.get(model, model)
            self.logger.info(f"Using OpenRouter model: {model} -> {openrouter_model}")

            # Prepare system message based on model and context
            if context:
                system_message = "You are an advanced AI assistant that helps users with various tasks. You have access to the conversation history, so please refer to previous messages when relevant."
            else:
                system_message = "You are an advanced AI assistant that helps users with various tasks."

            # Prepare the messages
            messages = []
            messages.append({"role": "system", "content": system_message})
            
            # Add conversation context if available
            if context:
                for msg in context:
                    messages.append(msg)
            
            # Add the current user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Prepare the API request payload
            payload = {
                "model": openrouter_model,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "text"}
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens

            # Log request details
            self.logger.info(f"Sending request to OpenRouter API with model {openrouter_model}")
            self.logger.debug(f"Payload: {json.dumps(payload)}")

            # Make the API call
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    self.api_failures += 1
                    self.api_last_failure = time.time()
                    return None

                # Parse response
                try:
                    result = await response.json()
                    self.logger.debug(f"OpenRouter API response: {json.dumps(result)}")
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        response_content = result["choices"][0]["message"]["content"]
                        return response_content
                    else:
                        self.logger.error("Invalid response format from OpenRouter API")
                        return None
                except Exception as e:
                    self.logger.error(f"Error parsing OpenRouter API response: {e}")
                    return None

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout while calling OpenRouter API (after {timeout}s)")
            self.api_failures += 1
            self.api_last_failure = time.time()
            return None
        except Exception as e:
            self.logger.error(f"Error calling OpenRouter API: {e}")
            self.logger.error(traceback.format_exc())
            self.api_failures += 1
            self.api_last_failure = time.time()
            return None
    
    @rate_limit
    async def generate_response_with_tool_calls(
        self,
        prompt: str,
        context: List[Dict] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: List[Dict] = None,
        timeout: float = 300.0,
    ) -> Optional[Dict]:
        """
        Generate a response that can include tool calls using the OpenRouter API.
        
        Args:
            prompt: The prompt to generate a response for
            context: Optional list of prior conversation messages
            model: The model ID to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens to generate
            tools: List of tool definitions to make available to the model
            timeout: API call timeout in seconds
            
        Returns:
            Dict: Response object containing text and/or tool calls
        """
        await self.rate_limiter.acquire()

        try:
            # Make sure MCP servers are running if needed
            if tools:
                # Check if we need Exa for any tools
                exa_tools = [tool for tool in tools if tool.get("function", {}).get("name", "").startswith("mcp_exa_")]
                if exa_tools:
                    self.logger.info("Exa tools requested, ensuring Exa MCP server is running")
                    await self.ensure_mcp_server("exa")

            # Ensure we have a session
            session = await self.ensure_session()

            # Map model ID to OpenRouter model path
            openrouter_model = self.available_models.get(model, model)
            self.logger.info(f"Using OpenRouter model with tools: {model} -> {openrouter_model}")

            # Prepare system message
            system_message = "You are an advanced AI assistant with tool use capabilities. Use the provided tools when appropriate to help the user."

            # Prepare the messages
            messages = []
            messages.append({"role": "system", "content": system_message})
            
            # Add conversation context if available
            if context:
                for msg in context:
                    messages.append(msg)
            
            # Add the current user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Prepare the API request payload
            payload = {
                "model": openrouter_model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if max_tokens:
                payload["max_tokens"] = max_tokens
                
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            # Log request details
            self.logger.info(f"Sending tool-enabled request to OpenRouter API")
            self.logger.debug(f"Payload: {json.dumps(payload)}")

            # Make the API call
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"OpenRouter API error: {response.status} - {error_text}")
                    self.api_failures += 1
                    self.api_last_failure = time.time()
                    return None

                # Parse response
                try:
                    result = await response.json()
                    self.logger.debug(f"OpenRouter API response: {json.dumps(result)}")
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]
                    else:
                        self.logger.error("Invalid response format from OpenRouter API")
                        return None
                except Exception as e:
                    self.logger.error(f"Error parsing OpenRouter API response: {e}")
                    return None

        except asyncio.TimeoutError:
            self.logger.error(f"Timeout while calling OpenRouter API (after {timeout}s)")
            self.api_failures += 1
            self.api_last_failure = time.time()
            return None
        except Exception as e:
            self.logger.error(f"Error calling OpenRouter API: {e}")
            self.logger.error(traceback.format_exc())
            self.api_failures += 1
            self.api_last_failure = time.time()
            return None
    
    async def execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        """
        Execute a list of tool calls returned by the OpenRouter API.
        
        Args:
            tool_calls: List of tool call objects from OpenRouter response
            
        Returns:
            List[Dict]: List of tool call results
        """
        results = []
        
        for tool_call in tool_calls:
            try:
                function = tool_call.get("function", {})
                tool_id = tool_call.get("id")
                tool_name = function.get("name")
                
                if not tool_name:
                    self.logger.error("Tool call missing function name")
                    continue
                
                # Parse arguments
                args_str = function.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    self.logger.error(f"Invalid arguments JSON for tool {tool_name}: {args_str}")
                    args = {}
                
                self.logger.info(f"Executing tool call: {tool_name}")
                
                # Determine which MCP server to use based on tool name
                server_id = None
                if tool_name.startswith("mcp_exa_"):
                    server_id = "exa"
                elif tool_name.startswith("mcp_duckduckgo_"):
                    server_id = "exa"  # For now, assuming all are in Exa
                    
                if not server_id:
                    self.logger.error(f"Cannot determine server for tool {tool_name}")
                    results.append({
                        "tool_call_id": tool_id,
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps({"error": "Unknown tool server"})
                    })
                    continue
                
                # Make sure the server is running
                await self.ensure_mcp_server(server_id)
                
                # Execute the tool call
                tool_result = await self.mcp_manager.execute_tool_call(server_id, tool_name, args)
                
                # Format the result
                result_content = ""
                if tool_result.get("type") == "function_result":
                    result_content = tool_result.get("result", "{}")
                elif tool_result.get("type") == "function_error":
                    self.logger.error(f"Tool error: {tool_result.get('error')}")
                    result_content = json.dumps({"error": tool_result.get("error")})
                else:
                    self.logger.warning(f"Unknown tool result type: {tool_result.get('type')}")
                    result_content = json.dumps(tool_result)
                
                # Add to results
                results.append({
                    "tool_call_id": tool_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": result_content
                })
                
            except Exception as e:
                self.logger.error(f"Error executing tool call: {e}")
                self.logger.error(traceback.format_exc())
                results.append({
                    "tool_call_id": tool_call.get("id"),
                    "role": "tool",
                    "name": tool_call.get("function", {}).get("name", "unknown"),
                    "content": json.dumps({"error": str(e)})
                })
        
        return results
    
    async def ensure_mcp_server(self, server_id: str) -> bool:
        """
        Ensure that an MCP server is running and ready.
        
        Args:
            server_id: The identifier of the server to ensure
            
        Returns:
            bool: True if server is running and ready
        """
        try:
            # Check if server is already running and healthy
            status = await self.mcp_manager.get_server_status(server_id)
            server_status = status.get(server_id, {})
            
            if server_status.get("running", False) and server_status.get("ready", False):
                self.logger.info(f"MCP server {server_id} is already running and ready")
                return True
            
            # Start server if not running or not ready
            self.logger.info(f"Starting MCP server {server_id}")
            await self.mcp_manager.start_server(server_id)
            
            # Wait for server to become ready
            max_retries = 10
            for i in range(max_retries):
                await asyncio.sleep(1)
                
                status = await self.mcp_manager.get_server_status(server_id)
                server_status = status.get(server_id, {})
                
                if server_status.get("running", False) and server_status.get("ready", False):
                    self.logger.info(f"MCP server {server_id} is ready")
                    return True
            
            self.logger.warning(f"MCP server {server_id} did not become ready within timeout")
            return False
            
        except Exception as e:
            self.logger.error(f"Error ensuring MCP server {server_id}: {e}")
            return False
    
    async def get_mcp_server_status(self, server_id: str = None) -> Dict:
        """
        Get the status of MCP servers.
        
        Args:
            server_id: Optional server ID to check specific server
            
        Returns:
            Dict: Status information for servers
        """
        return await self.mcp_manager.get_server_status(server_id)
    
    async def get_mcp_server_logs(self, server_id: str, max_lines: int = 100) -> Dict:
        """
        Get logs from an MCP server for diagnostics.
        
        Args:
            server_id: The server identifier
            max_lines: Maximum number of log lines to retrieve
            
        Returns:
            Dict: Log information
        """
        return await self.mcp_manager.get_server_logs(server_id, max_lines)
    
    def get_available_tools(self) -> List[Dict]:
        """
        Get a list of available tools for OpenRouter API tool calls.
        
        Returns:
            List[Dict]: List of tool definitions
        """
        # Define tools - these match the MCP tool signatures
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_exa_search_web_search_exa",
                    "description": "Search the web using Exa AI",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "numResults": {
                                "type": "number",
                                "description": "Number of search results to return (default: 5)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp_exa_search_crawling_exa",
                    "description": "Extract content from a specific URL using Exa AI",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to crawl and extract content from"
                            },
                            "maxCharacters": {
                                "type": "number",
                                "description": "Maximum characters to extract (default: 3000)"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp_duckduckgo_se_search",
                    "description": "Search DuckDuckGo and return formatted results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 10)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp_duckduckgo_se_fetch_content",
                    "description": "Fetch and parse content from a webpage URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The webpage URL to fetch content from"
                            }
                        },
                        "required": ["url"]
                    }
                }
            }
        ]
        
        return tools
