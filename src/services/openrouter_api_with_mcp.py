"""
Enhanced OpenRouter API with MCP Tool Integration

This module extends the base OpenRouter API to include automatic MCP server
integration, enabling LLMs to use MCP tools seamlessly.
"""

import logging
from typing import Dict, List, Optional, Any
from src.services.openrouter_api import OpenRouterAPI
from src.services.mcp import MCPManager
from src.services.model_handlers.model_configs import ModelConfigurations
from src.utils.log.telegramlog import telegram_logger


class OpenRouterAPIWithMCP(OpenRouterAPI):
    """
    Enhanced OpenRouter API with MCP (Model Context Protocol) integration.

    This class extends the base OpenRouterAPI to automatically load and use
    MCP server tools, converting them to OpenAI-compatible format.
    """

    def __init__(self, rate_limiter, mcp_config_path: str = "mcp.json"):
        """
        Initialize the enhanced OpenRouter API with MCP support.

        Args:
            rate_limiter: Rate limiter instance
            mcp_config_path: Path to MCP configuration file
        """
        super().__init__(rate_limiter)
        self.mcp_manager = MCPManager(mcp_config_path)
        self.mcp_tools_loaded = False
        self.logger = logging.getLogger(__name__)

    async def initialize_mcp_tools(self) -> bool:
        """
        Initialize and load MCP tools from configured servers.

        Returns:
            True if MCP tools were loaded successfully
        """
        try:
            self.logger.info("Initializing MCP tools...")
            telegram_logger.log_message("Initializing MCP tools...", 0)

            success = await self.mcp_manager.load_servers()

            if success:
                self.mcp_tools_loaded = True
                server_info = self.mcp_manager.get_server_info()
                self.logger.info(f"MCP tools initialized successfully: {server_info}")
                telegram_logger.log_message(f"MCP tools initialized: {len(server_info)} servers", 0)
                return True
            else:
                self.logger.warning("Failed to initialize MCP tools")
                telegram_logger.log_message("Failed to initialize MCP tools", 0)
                return False

        except Exception as e:
            self.logger.error(f"Error initializing MCP tools: {str(e)}")
            telegram_logger.log_error(f"Error initializing MCP tools: {str(e)}", 0)
            return False

    async def generate_response_with_mcp_tools(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a response using OpenRouter with MCP tools available.

        Args:
            prompt: The user prompt
            context: Conversation context
            model: Optional model override (if None, uses default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout

        Returns:
            Generated response or None if failed
        """
        # Ensure MCP tools are loaded
        if not self.mcp_tools_loaded:
            await self.initialize_mcp_tools()

        # Use provided model or default
        actual_model = model if model is not None else "gemini"

        # Get available MCP tools in OpenAI format
        mcp_tools = await self.mcp_manager.get_all_tools() if self.mcp_tools_loaded else []

        if mcp_tools:
            self.logger.info(f"Using {len(mcp_tools)} MCP tools for generation with model {actual_model}")
            return await self.generate_response_with_tools(
                prompt=prompt,
                tools=mcp_tools,
                context=context,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )
        else:
            self.logger.info(f"No MCP tools available, using standard generation with model {actual_model}")
            return await self.generate_response(
                prompt=prompt,
                context=context,
                model=actual_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

    def _build_system_message(
        self, model_id: str, context: Optional[List[Dict]] = None, tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Return a system message based on model and context, with tool usage instructions."""
        model_config = ModelConfigurations.get_all_models().get(model_id)
        if model_config and model_config.system_message:
            base_message = model_config.system_message
        else:
            base_message = "You are an advanced AI assistant that helps users with various tasks."

        context_hint = " Use conversation history/context when relevant." if context else ""

        # Add tool usage instructions if tools are available
        tool_instructions = ""
        if tools:
            tool_names = [tool["function"]["name"] for tool in tools]
            tool_instructions = f"""

You have access to the following tools: {', '.join(tool_names)}

When users ask for documentation, code examples, or information about libraries/frameworks, you should:
1. Use 'resolve-library-id' to find the correct library identifier
2. Use 'get-library-docs' to fetch comprehensive documentation with code examples
3. Provide detailed, accurate information based on the tool results

Always use these tools when appropriate rather than giving generic responses or links.

IMPORTANT: Do not include your internal thinking process, <think> tags, or tool call details in your final response to the user. Only provide the final, clean answer that directly addresses their question."""

        return base_message + context_hint + tool_instructions

    async def generate_response_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        context: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: float = 300.0,
    ) -> Optional[str]:
        """
        Generate a response with tool calling support.

        Args:
            prompt: The user prompt
            tools: List of tools in OpenAI format
            context: Conversation context
            model: Optional model to use (if None, uses default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout

        Returns:
            Generated response or None if failed
        """
        try:
            # Get model with fallback support
            openrouter_model = self.available_models.get(model, model)

            system_message = self._build_system_message(model, context, tools)
            messages = [{"role": "system", "content": system_message}]

            if context:
                messages.extend(context)

            messages.append({"role": "user", "content": prompt})

            # Make initial request with tools
            response = await self.client.chat.completions.create(
                model=openrouter_model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            # Handle tool calls if present
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                tool_calls = response.choices[0].message.tool_calls

                # Add assistant's message with tool calls to context
                messages.append(response.choices[0].message.model_dump())

                # Execute tools and add results
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    try:
                        # Parse arguments if they're a string
                        if isinstance(tool_args, str):
                            import json
                            tool_args = json.loads(tool_args)
                            self.logger.info(f"Parsed tool arguments for {tool_name}: {tool_args}")

                        # Execute the tool
                        self.logger.info(f"Executing MCP tool: {tool_name} with args: {tool_args}")
                        tool_result = await self.mcp_manager.execute_tool(tool_name, tool_args)

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(tool_result)
                        })

                        self.logger.info(f"Successfully executed MCP tool: {tool_name}")

                    except json.JSONDecodeError as json_error:
                        self.logger.error(f"Failed to parse tool arguments for {tool_name}: {str(json_error)}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error parsing tool arguments: {str(json_error)}"
                        })
                    except Exception as e:
                        self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
                        self.logger.error(f"Tool arguments were: {tool_args}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error executing tool: {str(e)}"
                        })

                # Make final request with tool results
                final_response = await self.client.chat.completions.create(
                    model=openrouter_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )

                # Clean the response content to remove thinking tags and tool calls
                cleaned_content = self._clean_response_content(final_response.choices[0].message.content)
                return cleaned_content

            else:
                # No tool calls, return the direct response
                cleaned_content = self._clean_response_content(response.choices[0].message.content)
                return cleaned_content

        except Exception as e:
            self.logger.error(f"Error in generate_response_with_tools: {str(e)}")
            return f"Error generating response with tools: {str(e)}"

    def _clean_response_content(self, content: str) -> str:
        """
        Clean response content by removing thinking tags and tool calls.

        Args:
            content: Raw response content from the model

        Returns:
            Cleaned content suitable for user display
        """
        if not content:
            return content

        import re

        # Remove thinking tags (common in reasoning models like DeepSeek)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # Remove tool call tags
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)

        # Remove any remaining XML-like tags that might be in the response
        content = re.sub(r'<[^>]+>.*?</[^>]+>', '', content, flags=re.DOTALL)

        # Clean up extra whitespace
        content = content.strip()

        # Remove multiple consecutive newlines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        return content

    async def get_available_mcp_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available MCP tools in OpenAI format.

        Returns:
            List of available MCP tools
        """
        if not self.mcp_tools_loaded:
            await self.initialize_mcp_tools()

        if self.mcp_tools_loaded:
            return await self.mcp_manager.get_all_tools()
        else:
            return []

    def get_mcp_server_info(self) -> Dict[str, Any]:
        """
        Get information about connected MCP servers.

        Returns:
            Dictionary with server information
        """
        if self.mcp_tools_loaded:
            return self.mcp_manager.get_server_info()
        else:
            return {}

    async def close(self):
        """Close the API client and MCP connections."""
        await super().close()
        if self.mcp_tools_loaded:
            await self.mcp_manager.disconnect_all()
